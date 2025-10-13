"""
ASIFT Calibration helper

Purpose:
- When the UI button "ASIFT Calibration" is pressed, call run_asift_calibration_and_send(...)
- It computes a 3x3 homography H from a reference (projector pattern) and a camera capture,
  then sends H to the projector app over its H ZMQ endpoint (REP at tcp://*:5560).

Notes:
- This file currently uses OpenCV SIFT + RANSAC to estimate H. To use true Affine-SIFT, you can
  integrate the MATLAB/Octave implementation at `https://github.com/rijn/Affine-SIFT.git` by
  wrapping it via octave-cli and then estimating H from the matched features.

Usage (CLI):
  python3 asift_calibration.py --ref ref.png --cam cam.png \
      --endpoint tcp://127.0.0.1:5560 --save H_asift.txt

UI hook (pseudo):
  from ZMQ_sender_mask.asift_calibration import run_asift_calibration_and_send
  run_asift_calibration_and_send(ref_path, cam_path, endpoint="tcp://127.0.0.1:5560", save_txt=H_path)
"""

from __future__ import annotations

import argparse
import sys
import struct
from typing import Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover
    cv2 = None

try:
    import zmq  # type: ignore
except Exception as e:  # pragma: no cover
    zmq = None


def _require_cv2():
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required. Install: pip install opencv-python-headless")


def _require_zmq():
    if zmq is None:
        raise RuntimeError("pyzmq is required. Install: pip install pyzmq")


def compute_homography_sift(
    ref_gray: np.ndarray,
    cam_gray: np.ndarray,
    ratio_test: float = 0.75,
    ransac_thresh: float = 3.0,
    max_feats: int = 4000,
) -> Optional[np.ndarray]:
    """Estimate H (3x3) mapping ref->cam using SIFT + RANSAC.

    Returns H (float64 3x3) or None if insufficient matches.
    """
    _require_cv2()

    if ref_gray.ndim != 2 or cam_gray.ndim != 2:
        raise ValueError("Inputs must be grayscale images (H,W)")

    sift = cv2.SIFT_create(nfeatures=max_feats)
    kps1, des1 = sift.detectAndCompute(ref_gray, None)
    kps2, des2 = sift.detectAndCompute(cam_gray, None)

    if des1 is None or des2 is None or len(kps1) < 4 or len(kps2) < 4:
        return None

    index_params = dict(algorithm=1, trees=5)  # FLANN KDTree
    search_params = dict(checks=64)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < ratio_test * n.distance:
            good.append(m)

    if len(good) < 4:
        return None

    pts1 = np.float32([kps1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kps2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)
    if H is None or not np.isfinite(H).all():
        return None
    # Normalize so H[2,2] == 1 when possible
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H.astype(np.float64)


def _affine_variant(image: np.ndarray, angle_deg: float, tilt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Create a simple affine-approximated variant via rotation + anisotropic scale.

    Returns (variant_image, inverse_2x3_matrix_to_original_coords)
    """
    h, w = image.shape[:2]
    # rotation
    M_rot = cv2.getRotationMatrix2D((w/2.0, h/2.0), angle_deg, 1.0)
    rotated = cv2.warpAffine(image, M_rot, (w, h), flags=cv2.INTER_LINEAR)

    # anisotropic scale to approximate tilt: scale x by 1/tilt, y by tilt (keeps area similar)
    S = np.array([[1.0/tilt, 0.0, 0.0],
                  [0.0,      tilt, 0.0]], dtype=np.float32)
    variant = cv2.warpAffine(rotated, S, (w, h), flags=cv2.INTER_LINEAR)

    # Compose total forward affine A: first rotate, then scale: A = S * [M_rot; 0 0 1]
    A = S @ np.vstack([M_rot, [0, 0, 1]])
    # Inverse 2x3 to map variant keypoints back to original
    A3 = np.vstack([A, [0, 0, 1]])
    A_inv = np.linalg.inv(A3)[:2, :]
    return variant, A_inv.astype(np.float32)


def compute_homography_asift(
    ref_gray: np.ndarray,
    cam_gray: np.ndarray,
    ratio_test: float = 0.75,
    ransac_thresh: float = 3.0,
    max_feats: int = 2000,
    angles_deg = (0.0, 45.0, 90.0, 135.0),
    tilts = (1.0, 1.6),
) -> Optional[np.ndarray]:
    """Approximate ASIFT by synthesizing a few affine variants, then SIFT+RANSAC.

    This is a pragmatic, lightweight approximation; for full ASIFT, integrate
    the MATLAB/Octave repo (`https://github.com/rijn/Affine-SIFT.git`) and
    import its matched features before RANSAC homography estimation.
    """
    _require_cv2()
    if ref_gray.ndim != 2 or cam_gray.ndim != 2:
        raise ValueError("Inputs must be grayscale images (H,W)")

    sift = cv2.SIFT_create(nfeatures=max_feats)

    matches_xy1 = []
    matches_xy2 = []

    # Precompute variants
    ref_variants = []  # list of (img_var, invA)
    cam_variants = []
    for a in angles_deg:
        for t in tilts:
            r_img, r_invA = _affine_variant(ref_gray, a, t)
            c_img, c_invA = _affine_variant(cam_gray, a, t)
            ref_variants.append((r_img, r_invA))
            cam_variants.append((c_img, c_invA))

    # Detect features per variant
    ref_kd = []  # list of (kps, des, invA)
    cam_kd = []
    for (img, invA) in ref_variants:
        kps, des = sift.detectAndCompute(img, None)
        if des is not None and len(kps) >= 4:
            ref_kd.append((kps, des, invA))
    for (img, invA) in cam_variants:
        kps, des = sift.detectAndCompute(img, None)
        if des is not None and len(kps) >= 4:
            cam_kd.append((kps, des, invA))

    if not ref_kd or not cam_kd:
        return None

    index_params = dict(algorithm=1, trees=5)  # FLANN KDTree
    search_params = dict(checks=64)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match across a subset of variant pairs (same index pairing to limit cost)
    for i in range(min(len(ref_kd), len(cam_kd))):
        kps1, des1, invA1 = ref_kd[i]
        kps2, des2, invA2 = cam_kd[i]
        if des1 is None or des2 is None:
            continue
        ms = flann.knnMatch(des1, des2, k=2)
        good = []
        for m, n in ms:
            if m.distance < ratio_test * n.distance:
                good.append(m)
        if len(good) < 4:
            continue

        # Back-map keypoints to original image coordinates
        for m in good:
            x1, y1 = kps1[m.queryIdx].pt
            x2, y2 = kps2[m.trainIdx].pt
            p1 = np.array([x1, y1, 1.0], dtype=np.float32)
            p2 = np.array([x2, y2, 1.0], dtype=np.float32)
            q1 = invA1 @ p1
            q2 = invA2 @ p2
            matches_xy1.append([float(q1[0]), float(q1[1])])
            matches_xy2.append([float(q2[0]), float(q2[1])])

    if len(matches_xy1) < 4:
        return None

    pts1 = np.float32(matches_xy1).reshape(-1, 1, 2)
    pts2 = np.float32(matches_xy2).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)
    if H is None or not np.isfinite(H).all():
        return None
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H.astype(np.float64)


def send_h_over_zmq(H: np.ndarray, endpoint: str = "tcp://127.0.0.1:5560", timeout_ms: int = 500):
    """Send H to projector REP endpoint as multipart: ["H", 9*double]."""
    _require_zmq()
    if H.shape != (3, 3):
        raise ValueError("H must be 3x3")

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    try:
        sock.RCVTIMEO = timeout_ms
        sock.SNDTIMEO = timeout_ms
    except Exception:
        pass
    sock.connect(endpoint)

    payload = struct.pack("<9d", *H.reshape(-1).tolist())
    sock.send_multipart([b"H", payload])
    try:
        reply = sock.recv()
        # Expect b"OK"
    except Exception:
        reply = b""
    finally:
        sock.close(0)


def save_h_to_text(H: np.ndarray, path: str):
    """Write 9 doubles (row-major) to a text file, space-separated."""
    if H.shape != (3, 3):
        raise ValueError("H must be 3x3")
    vals = " ".join(f"{float(x):.17g}" for x in H.reshape(-1))
    with open(path, "w") as f:
        f.write(vals + "\n")


def run_asift_calibration_and_send(
    ref_path: str,
    cam_path: str,
    endpoint: str = "tcp://127.0.0.1:5560",
    save_txt: Optional[str] = None,
) -> Tuple[bool, Optional[np.ndarray]]:
    """High-level entry point for the UI button.

    - Loads ref and cam images (grayscale),
    - computes H via SIFT+RANSAC (ASIFT-ready hook),
    - sends H to projector over ZMQ,
    - optionally saves H to text file.

    Returns (ok, H or None)
    """
    _require_cv2()

    ref_gray = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    cam_gray = cv2.imread(cam_path, cv2.IMREAD_GRAYSCALE)
    if ref_gray is None or cam_gray is None:
        raise FileNotFoundError(f"Could not load ref='{ref_path}' or cam='{cam_path}'")

    # Prefer ASIFT-style matching; fall back to plain SIFT if needed
    H = compute_homography_asift(ref_gray, cam_gray)
    if H is None:
        H = compute_homography_sift(ref_gray, cam_gray)
    if H is None:
        return False, None

    try:
        send_h_over_zmq(H, endpoint=endpoint)
    except Exception:
        # ZMQ send is best-effort; continue to save file if requested
        pass

    if save_txt:
        save_h_to_text(H, save_txt)
    return True, H


def run_asift_ui_action(
    ref_path: str,
    cam_path: str,
    h_txt_path: str,
    endpoint: str = "tcp://127.0.0.1:5560",
) -> Tuple[bool, Optional[np.ndarray]]:
    """Convenience function for UI button callback.

    - Computes H with ASIFT approximation (fallback SIFT),
    - Sends H to projector (so it is applied immediately),
    - Saves H to the same text file your regular Calibration uses.
    """
    ok, H = run_asift_calibration_and_send(ref_path, cam_path, endpoint=endpoint, save_txt=h_txt_path)
    return ok, H


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="ASIFT Calibration helper (SIFT+RANSAC backend)")
    p.add_argument("--ref", required=True, help="Reference (projected pattern) image path")
    p.add_argument("--cam", required=True, help="Camera-captured image path")
    p.add_argument("--endpoint", default="tcp://127.0.0.1:5560", help="Projector H REP endpoint")
    p.add_argument("--save", default=None, help="Optional path to save 9-text-doubles H file")
    return p.parse_args(argv)


def main(argv=None):  # pragma: no cover
    args = _parse_args(argv)
    ok, H = run_asift_calibration_and_send(args.ref, args.cam, endpoint=args.endpoint, save_txt=args.save)
    if not ok:
        print("ASIFT calibration failed: insufficient matches or homography not found", file=sys.stderr)
        return 2
    print("ASIFT calibration OK. H=\n", H)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())



