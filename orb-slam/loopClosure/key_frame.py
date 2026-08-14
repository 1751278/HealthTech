#A KeyFrame, storing all the data from a single frame.
class KeyFrame:
    def __init__(
        self,
        id,
        frame_number,
        pose_R,
        pose_T,
        keypoints,
        descriptors,
        global_descriptor=None,
    ):
        self.id = id

        self.frame_number = frame_number
        self.keypoints = keypoints
        self.descriptors = descriptors

        # Use VLAD as the global descriptor when one has already been
        # computed; otherwise fall back to mean-pooling until VLAD is fitted.
        self.global_descriptor = (
            global_descriptor
            if global_descriptor is not None
            else descriptors.mean(dim=0)
        )

        self.pose_R = pose_R.copy()  # Rotation
        self.pose_T = pose_T.copy()  # Translation