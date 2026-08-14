#Use Instructions#
# This class represents a loop closure mechanism that uses a matching function to identify potential loop closures.
# Initialize this before the main loop
    # Match_function represents a function such as xFeat that handles matching between features in the current frame and previously stored frames to detect loop closures.
    # In the initialization of the VO class, you can create an instance of the LoopClosure class and pass the matching function to it.
    # Example: self.lc = lc(self._match_xFeat,self.K,self.device)
    # Then at the end of the process_frame function, you can call the process_loop_check method of the LoopClosure instance to check for loop closures.
    # Example: 
    # traj = self.lc.process_loop_check(self.cur_R, self.cur_t, frame_count, kp_full, feats_full,self.trajectory)
    # if traj:
        #self.trajectory = traj


import torch
import cv2
import numpy as np
from .VLAD import VLAD
from .key_frame import KeyFrame
# --------------------------------------------------------------------------- #
# VLAD (Vector of Locally Aggregated Descriptors) implementation
# --------------------------------------------------------------------------- #
class LoopClosure:
    def __init__(self, match_function, K, device=torch.device("cpu")):
        self.match_function = match_function
        self.device = device
        self.K = K
        # VLAD provides a fixed-length global descriptor for keyframe
        # retrieval during loop-closure detection.
        self.vlad = VLAD(descriptor_dim=64, n_clusters=32, device=self.device)
        self.prev_keyframe = None
        self.next_keyframe_id = 0
        self.trajectory = []
        self.keyframes = [] #stores all the keyframes in this list, to be looked out during loop closure checks
    def _create_keyframe(self, frame_count, kp, feats):
            # Keyframes keep descriptors on the CPU so large descriptor banks do
            # not remain permanently allocated in GPU memory.
            if isinstance(feats, torch.Tensor):
                feats_cpu = feats.detach().cpu()
            else:
                feats_cpu = torch.tensor(feats)
    
            # Once VLAD has been fitted, use it to create the keyframe's global
            # descriptor. Before that point, use mean-pooling as a temporary
            # fallback descriptor.
            if self.vlad.centroids is not None:
                global_descriptor = self.vlad.encode(feats_cpu.to(self.device))
            else:
                global_descriptor = feats_cpu.mean(dim=0)
    
            kf = KeyFrame(
                self.next_keyframe_id,
                frame_count,
                self.cur_R,
                self.cur_t,
                kp,
                feats_cpu,
                global_descriptor=global_descriptor,
            )
            self.keyframes.append(kf)
            self.prev_keyframe = kf
            self.next_keyframe_id += 1
            self._maybe_fit_vlad()
    def _process_keyframes(self, n_candidates=3, similarity_thresh=0.85):
            current = self.prev_keyframe
            eligible = [kf for kf in self.keyframes if current.id - kf.id >= 50]
            if not eligible:
                return []
    
            # Stack all global descriptors into one matrix so cosine similarities
            # against the current keyframe can be computed in a single batched
            # matrix multiplication instead of one Python loop per keyframe.
            bank = torch.stack([kf.global_descriptor for kf in eligible])   # (M, k*d), already L2-normed
            scores = bank @ current.global_descriptor                       # (M,) cosine sim in one matmul
            scores_np = scores.cpu()
            
            keep = scores_np >= similarity_thresh
            idxs = keep.nonzero(as_tuple=True)[0]
            if len(idxs) == 0:
                
                return []
            ranked = sorted(((scores_np[i].item(), eligible[i]) for i in idxs), key=lambda x: x[0], reverse=True)
            return ranked[:n_candidates]
    
    def _prefilter_candidates(self, candidates, current_frame, sim_thresh=0.9, min_count=100):
        """
        Cheap one-shot ranking of candidates using a single batched matmul.
        Returns candidates re-sorted by rough inlier count, keeping only those
        that clear min_count. This does not replace the full geometric check.
        """
        # Normalize local descriptors so their dot product represents cosine
        # similarity during the cheap candidate prefilter.
        cur_desc = current_frame.descriptors.to(self.device)
        cur_desc = torch.nn.functional.normalize(cur_desc, dim=1)

        cat_desc = []
        boundaries = []  # (candidate, score, start, end)
        offset = 0
        for score, cand in candidates:
            d = torch.nn.functional.normalize(cand.descriptors.to(self.device), dim=1)
            boundaries.append((cand, score, offset, offset + d.shape[0]))
            cat_desc.append(d)
            offset += d.shape[0]

        if not cat_desc:
            return []

        # Combine every candidate's descriptors into one matrix so all
        # candidate-to-current-frame similarities can be evaluated together.
        cat_desc = torch.cat(cat_desc, dim=0)

        with torch.inference_mode():
            sim = cat_desc @ cur_desc.T

            # For every candidate descriptor, keep only its strongest
            # similarity against any descriptor in the current frame.
            best_sim, _ = sim.max(dim=1)

        ranked = []
        for cand, score, start, end in boundaries:
            # This is only a rough match count used to reject weak candidates
            # before running the expensive XFeat + RANSAC verification.
            rough_count = int((best_sim[start:end] > sim_thresh).sum().item())
            if rough_count >= min_count:
                ranked.append((rough_count, score, cand))

        # Process candidates with the strongest rough descriptor support first.
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [(score, cand) for _, score, cand in ranked]
    def _process_candidates(self,candidates):
            
            current_frame = self.prev_keyframe  # Grab the current keyframe

            # Cheap descriptor-level filtering happens before expensive XFeat
            # matching and geometric verification, reducing the number of
            # candidates that reach the RANSAC stage.
            candidates = self._prefilter_candidates(candidates, current_frame)
            if not candidates:
                return None
            
            for (score, candidate) in candidates:
                print("Score: ", score, " Candidate KF: ", candidate.id, " Current KF: ", current_frame.id)
                matches = self.match_function(
                    candidate.descriptors,
                    current_frame.descriptors
                )
                if matches.shape[0] < 100:
                    continue

                # Matches are raw descriptor-index pairs, so NumPy fancy indexing
                # directly retrieves the corresponding keypoint coordinates.
                pts_old = np.float32(candidate.keypoints[matches[:, 0]])
                pts_new = np.float32(current_frame.keypoints[matches[:, 1]])
                E, mask = cv2.findEssentialMat(
                    pts_new,
                    pts_old,
                    self.K,
                    method=cv2.RANSAC,
                    prob=0.999,
                    threshold=1.0
                )
                if E is None:
                    continue
                num_inliers, R, t, pose_mask = cv2.recoverPose(
                    E,
                    pts_new,
                    pts_old,
                    self.K,
                    mask=mask
                )

                ratio = num_inliers / len(matches)

                if num_inliers > 150 and ratio > 0.5:

                    print(
                        f"LOOP FOUND! "
                        f"KF {current_frame.id} -> KF {candidate.id} "
                        f"({num_inliers}/{len(matches)})"
                        f"(Score: {score:.3f})"
                    )

                    return candidate, R, t

            return None
    def _update_trajectory(self, candidate, current_frame_number): #Gradually ramps up the amount of correction in order to correct the trajectory
    
    
            # Save the current pose
    
    
            old_R = self.cur_R.copy()
            old_t = self.cur_t.copy()
    
            # Calculate the rotational error
            R_error = candidate.pose_R @ old_R.T
            # Calculate the translational error
            t_error = candidate.pose_T - R_error @ old_t
    
            # Start from candidate, end at current frame, as those are the most relevant frames to correct. The rest of the trajectory is left unchanged.
    
            start = max(0, candidate.frame_number)
            end = min(current_frame_number, len(self.trajectory) - 1)
    
            # Check if the range is valid
            if start >= end:
                print("Loop correction skipped: invalid trajectory range.")
                return
    
            #  Apply the correction gradually
            total_frames = end - start
    
    
            for i in range(start, end + 1):
    
                alpha = (i - start) / total_frames # How far along the trajectory we are, from 0.0 (candidate) to 1.0 (current frame)
    
    
                # Translation correction
                original_position = self.trajectory[i]
    
                corrected_position = (
                    R_error @ original_position
                    + t_error
                )
    
                # Blend between original and corrected position.
    
                self.trajectory[i] = (
                    (1.0 - alpha) * original_position
                    + alpha * corrected_position
                )
    
            # Correct keyframes using the same gradual correction
    
            for kf in self.keyframes:
    
                if kf.id < candidate.id:
                    continue
    
                
                if kf.frame_number > current_frame_number:
                    continue
    
                # Candidate itself should remain fixed.
                if kf.id == candidate.id:
                    continue
    
                # Determine where this keyframe lies between the candidate and current frame.
                alpha = (
                    kf.frame_number - candidate.frame_number
                ) / (
                    current_frame_number - candidate.frame_number
                )
    
                alpha = np.clip(alpha, 0.0, 1.0)
    
                # Calculate correction
    
    
                corrected_t = (
                    R_error @ kf.pose_T
                    + t_error
                )
    
                corrected_R = (
                    R_error @ kf.pose_R
                )
                #Translation correction
    
                kf.pose_T = (
                    (1.0 - alpha) * kf.pose_T
                    + alpha * corrected_t
                )
    
                # Rotation
                R_delta = corrected_R @ kf.pose_R.T
    
                rvec, _ = cv2.Rodrigues(R_delta)
    
                interpolated_rvec = rvec * alpha
    
                R_interpolated, _ = cv2.Rodrigues(interpolated_rvec)
    
                kf.pose_R = (
                    R_interpolated @ kf.pose_R
                )
    
            #Current pose should match candidate's pose
    
            self.cur_t = self.trajectory[end].copy()
            self.cur_R = R_error @ old_R
    
            #Debug Information:
    
            print("\n========== LOOP CLOSURE ==========")
            print("Candidate KF:", candidate.id)
            print("Candidate frame:", candidate.frame_number)
            print("Current frame:", current_frame_number)
            print("==================================\n")
    def _maybe_fit_vlad(self):
            # Wait until enough keyframes exist to build a representative VLAD
            # codebook, then fit it once and re-encode all existing keyframes.
            if self.vlad.centroids is not None or len(self.keyframes) < 20:
                return
            pool = torch.cat([kf.descriptors.to(self.device) for kf in self.keyframes], dim=0)
            self.vlad.fit(pool)
    
            # Existing keyframes were initially encoded using mean-pooling.
            # Re-encode them now that the shared VLAD codebook exists.
            for kf in self.keyframes:
                kf.global_descriptor = self.vlad.encode(kf.descriptors.to(self.device))

    def process_loop_check(self, cur_R, cur_t, frame_count, kp_full, feats_full, trajectory):
        self.cur_t = cur_t
        self.cur_R = cur_R
        self.trajectory = trajectory

         # Create new keyframes and store them when the camera has moved or
        # rotated far enough from the previous keyframe.
        if len(self.keyframes) > 0:
            translation = np.linalg.norm(
                self.cur_t - self.prev_keyframe.pose_T
            )
            R_rel = self.cur_R @ self.prev_keyframe.pose_R.T

            trace = np.trace(R_rel)

            # Prevents errors
            value = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)

            rotation = np.degrees(np.arccos(value))  # Rotation in degrees

            if translation > 0.6 or rotation > 15:
                print("Keyframe created: ", self.next_keyframe_id, " translation: ", translation, " rotation: ", int(rotation))
                self._create_keyframe(frame_count, kp_full, feats_full)
                candidates = self._process_keyframes(50,0.1)
                if candidates:
                    result = self._process_candidates(candidates)
                    if result:
                        self._update_trajectory(result[0], frame_count)
                        return self.trajectory
        else:
            print("Keyframe created: ", self.next_keyframe_id)

            # The first keyframe also uses the full dense 4096-feature
            # detection so it is not less descriptive than later keyframes.
            self._create_keyframe(frame_count, kp_full, feats_full)

