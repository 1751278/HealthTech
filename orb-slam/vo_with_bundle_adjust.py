import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

# =====================================================================
# 1. BUNDLE ADJUSTMENT HELPERS
# =====================================================================

def project(points_3d, camera_params, K):
    """Vectorized projection of 3D points into 2D pixel coordinates."""
    rvecs = camera_params[:, :3]
    tvecs = camera_params[:, 3:]
    
    # Rodrigues rotation vector to matrix (vectorized)
    theta = np.linalg.norm(rvecs, axis=1, keepdims=True)
    theta_safe = np.where(theta == 0, 1e-10, theta)
    k = rvecs / theta_safe
    
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    k_cross_p = np.cross(k, points_3d)
    k_dot_p = np.sum(k * points_3d, axis=1, keepdims=True)
    
    points_rot = points_3d * cos_t + k_cross_p * sin_t + k * k_dot_p * (1 - cos_t)
    points_cam = points_rot + tvecs
    
    # Intrinsic projection
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Prevent division by zero / points behind camera
    z = np.maximum(points_cam[:, 2], 1e-5)
    u = fx * (points_cam[:, 0] / z) + cx
    v = fy * (points_cam[:, 1] / z) + cy
    
    return np.column_stack([u, v])


def residual_function(params, n_cameras, n_points, camera_indices, point_indices, points_2d, K):
    """Computes reprojection residual errors for SciPy optimization."""
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))
    
    obs_cameras = camera_params[camera_indices]
    obs_points_3d = points_3d[point_indices]
    
    points_proj = project(obs_points_3d, obs_cameras, K)
    return (points_proj - points_2d).ravel()


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    """Constructs the sparse Jacobian pattern matrix."""
    m = len(camera_indices) * 2
    n = n_cameras * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(len(camera_indices))
    
    for d in range(6):
        A[2 * i, camera_indices * 6 + d] = 1
        A[2 * i + 1, camera_indices * 6 + d] = 1

    for d in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + d] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + d] = 1

    return A


# =====================================================================
# 2. VISUAL ODOMETRY WITH SLIDING WINDOW BUNDLE ADJUSTMENT
# =====================================================================

class MonocularVOWithBA:
    def __init__(self, K: np.ndarray, window_size: int = 5):
        self.K = K
        self.window_size = window_size
        
        # Feature Extractor and Matcher
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Trajectory Pose Tracking
        self.R_global = np.eye(3)
        self.t_global = np.zeros((3, 1))
        
        # Caching current frame state
        self.prev_kp = None
        self.prev_des = None
        
        # Sliding Window Buffers
        self.window_poses = []       # List of 6D camera parameter arrays [rvec, tvec]
        self.window_points_3d = []   # Accumulated 3D points
        self.camera_indices = []     # Index of camera observing point i
        self.point_indices = []      # Index of 3D point i
        self.points_2d = []          # 2D pixel coordinates of observations

    def _triangulate_points(self, R_rel, t_rel, pts1, pts2):
        """Triangulates 3D points between two views given relative motion."""
        P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = self.K @ np.hstack((R_rel, t_rel))
        
        pts1_h = pts1.T
        pts2_h = pts2.T
        
        pts4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
        pts3d = (pts4d[:3] / pts4d[3]).T
        
        # Filter points behind camera (z <= 0)
        valid_mask = pts3d[:, 2] > 0
        return pts3d, valid_mask

    def run_local_bundle_adjustment(self):
        """Executes Bundle Adjustment over the current sliding window buffer."""
        n_cameras = len(self.window_poses)
        n_points = len(self.window_points_3d)
        
        if n_cameras < 2 or n_points == 0:
            return

        initial_cameras = np.array(self.window_poses)
        initial_points = np.array(self.window_points_3d)
        
        camera_indices = np.array(self.camera_indices)
        point_indices = np.array(self.point_indices)
        points_2d = np.array(self.points_2d)
        
        # Initial state vector
        x0 = np.hstack((initial_cameras.ravel(), initial_points.ravel()))
        
        # Compute Sparse Structure Matrix
        A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
        
        # Optimize using Huber robust loss to eliminate outlier matches
        res = least_squares(
            fun=residual_function,
            x0=x0,
            jac_sparsity=A,
            method='trf',
            loss='huber',
            f_scale=1.0,
            ftol=1e-3,
            max_nfev=20,  # Fast local refinement step
            args=(n_cameras, n_points, camera_indices, point_indices, points_2d, self.K)
        )
        
        # Extract optimized camera poses
        optimized_cameras = res.x[:n_cameras * 6].reshape((n_cameras, 6))
        
        # Update last camera pose in sliding window
        latest_rvec = optimized_cameras[-1, :3]
        latest_tvec = optimized_cameras[-1, 3:].reshape(3, 1)
        
        R_opt, _ = cv2.Rodrigues(latest_rvec)
        self.R_global = R_opt
        self.t_global = latest_tvec

    def process_frame(self, frame: np.ndarray):
        """Processes an incoming camera frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        kp, des = self.orb.detectAndCompute(gray, None)

        if self.prev_des is None:
            self.prev_kp, self.prev_des = kp, des
            rvec, _ = cv2.Rodrigues(self.R_global)
            self.window_poses.append(np.hstack([rvec.ravel(), self.t_global.ravel()]))
            return self.R_global, self.t_global

        # 1. Feature Matching
        matches = sorted(self.bf.match(self.prev_des, des), key=lambda x: x.distance)
        pts_prev = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
        pts_curr = np.float32([kp[m.trainIdx].pt for m in matches])

        # 2. Relative Pose Estimation
        E, mask = cv2.findEssentialMat(pts_curr, pts_prev, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask_pose = cv2.recoverPose(E, pts_curr, pts_prev, self.K)

        # 3. Trajectory Accumulation
        self.t_global = self.t_global + (self.R_global @ t)
        self.R_global = R @ self.R_global
        
        rvec_curr, _ = cv2.Rodrigues(self.R_global)
        current_pose_param = np.hstack([rvec_curr.ravel(), self.t_global.ravel()])
        self.window_poses.append(current_pose_param)

        # 4. Triangulate Points & Store Observations
        pts3d, valid_mask = self._triangulate_points(R, t, pts_prev, pts_curr)
        
        cam_idx_prev = len(self.window_poses) - 2
        cam_idx_curr = len(self.window_poses) - 1

        for i, is_valid in enumerate(valid_mask):
            if is_valid:
                pt_idx = len(self.window_points_3d)
                self.window_points_3d.append(pts3d[i])
                
                # Observation from previous frame
                self.camera_indices.append(cam_idx_prev)
                self.point_indices.append(pt_idx)
                self.points_2d.append(pts_prev[i])
                
                # Observation from current frame
                self.camera_indices.append(cam_idx_curr)
                self.point_indices.append(pt_idx)
                self.points_2d.append(pts_curr[i])

        # 5. Run Local BA when window buffer reaches capacity
        if len(self.window_poses) >= self.window_size:
            self.run_local_bundle_adjustment()
            
            # Slide window: slide buffer forward
            self.window_poses.pop(0)
            self.camera_indices = [c - 1 for c in self.camera_indices if c > 0]

        # Cache keypoints for next step
        self.prev_kp, self.prev_des = kp, des
        return self.R_global, self.t_global


# =====================================================================
# 3. EXAMPLE EXECUTION LOOP
# =====================================================================

if __name__ == "__main__":
    K = np.array([
        [718.856,   0.0,   607.1928],
        [  0.0,   718.856, 185.2157],
        [  0.0,     0.0,     1.0   ]
    ])

    vo = MonocularVOWithBA(K, window_size=5)
    cap = cv2.VideoCapture("path_to_video.mp4")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        R, t = vo.process_frame(frame)
        print(f"Frame {frame_idx:04d} -> Global Camera Position (x, y, z): {t.ravel()}")
        frame_idx += 1

    cap.release()