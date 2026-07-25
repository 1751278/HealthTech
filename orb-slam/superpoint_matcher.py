import cv2
import torch
import numpy as np
from lightglue import SuperPoint, LightGlue
from lightglue.utils import load_image, rbd


class SuperPointMatcher:
    def __init__(self, max_keypoints: int = 2048, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # SuperPoint: replaces cv2.ORB_create()
        self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(self.device)
        # LightGlue: replaces cv2.BFMatcher(...) + knnMatch + ratio test
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)

    def detect_and_compute(self, image):
        """Equivalent to orb.detectAndCompute(img, None).

        Returns:
            keypoints_cv: list of cv2.KeyPoint (pixel coords, for drawMatches etc.)
            feats: dict of raw torch feature tensors, needed for matching
        """
        tensor_image = torch.from_numpy(image)

        # 3. Rearrange dimensions from HxWxC to CxHxW (required by PyTorch models)
        tensor_image = tensor_image.permute(2, 0, 1)

        # 4. Convert type to float and scale pixel values to [0.0, 1.0]
        tensor_image = tensor_image.float() / 255.0

        # 5. Add a batch dimension (BxCxHxW) if feeding into a Neural Network
        tensor_image = tensor_image.unsqueeze(0)

        # 6. Load the optimized tensor directly to your target device
        tensor_image = tensor_image.to(self.device)
        with torch.no_grad():
            feats = self.extractor.extract(tensor_image)
        keypoints_np = feats["keypoints"][0].cpu().numpy()
        keypoints_cv = [cv2.KeyPoint(x=float(x), y=float(y), size=1.0) for x, y in keypoints_np]
        return keypoints_cv, feats

    def match(self, feats0, feats1, min_confidence: float = 0.90):
        """Equivalent to bf.knnMatch(des1, des2, k=2) + the 0.75 ratio test.

        LightGlue's raw output is already the confident, mutually-agreed
        match set (its analogue of "good" matches), so no separate ratio
        test is applied. `min_confidence` optionally tightens that further.

        Returns:
            good_matches: list of cv2.DMatch, sorted by distance ascending
                          (distance = 1 - match confidence, so lower is better,
                          matching cv2's "lower distance = better match" convention)
        """
        with torch.no_grad():
            result = self.matcher({"image0": feats0, "image1": feats1})
        result = rbd(result)  # remove batch dimension

        idx_pairs = result["matches"].cpu().numpy()  # (M, 2): (queryIdx, trainIdx)
        scores = result["scores"].cpu().numpy()  # (M,) confidence in [0, 1]

        if min_confidence > 0:
            keep = scores >= min_confidence
            idx_pairs = idx_pairs[keep]
            scores = scores[keep]

        good_matches = [
            cv2.DMatch(_queryIdx=int(i0), _trainIdx=int(i1), _distance=float(1.0 - s))
            for (i0, i1), s in zip(idx_pairs, scores)
        ]
        good_matches = sorted(good_matches, key=lambda m: m.distance)
        return good_matches

    def match_images(self, image_path0: str, image_path1: str, min_confidence: float = 0.0):
        """Full pipeline: detect + describe + match for two images.

        Returns:
            kp0, kp1: lists of cv2.KeyPoint (same shape as orb.detectAndCompute output)
            good_matches: list of cv2.DMatch (same shape as the post-ratio-test `good` list)
        """
        kp0, feats0 = self.detect_and_compute(image_path0)
        kp1, feats1 = self.detect_and_compute(image_path1)
        good_matches = self.match(feats0, feats1, min_confidence=min_confidence)
        return kp0, kp1, good_matches


def draw_matches(image_path0, image_path1, kp0, kp1, good_matches, out_path="matches.png", max_lines=200):
    """Same call shape as the classic:
        cv2.drawMatches(img1, kp1, img2, kp2, good[:N], None, flags=2)
    """
    img0 = cv2.imread(image_path0)
    img1 = cv2.imread(image_path1)
    canvas = cv2.drawMatches(
        img0, kp0, img1, kp1, good_matches[:max_lines], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(out_path, canvas)
    return out_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python superpoint_matcher.py <image1> <image2>")
        sys.exit(1)

    img_path0, img_path1 = sys.argv[1], sys.argv[2]

    sp_matcher = SuperPointMatcher(max_keypoints=2048)
    # kp0, kp1: list[cv2.KeyPoint]      -- like orb.detectAndCompute output
    # good:    list[cv2.DMatch]        -- like the post-ratio-test `good` list
    kp0, kp1, good = sp_matcher.match_images(img_path0, img_path1)

    print(f"Found {len(good)} good matches")
    out = draw_matches(img_path0, img_path1, kp0, kp1, good, out_path="matches.png")
    print(f"Saved visualization to {out}")