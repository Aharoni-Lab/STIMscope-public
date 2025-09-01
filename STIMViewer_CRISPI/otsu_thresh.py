
from __future__ import annotations
import os
import sys
import gc
import time
from pathlib import Path
from typing import Optional, Tuple, List, Union

import numpy as np
import cv2




try:
    import cupy as cp
    _CUPY = True
    print("✅ CuPy available for mean projection acceleration")
except Exception:
    cp = None
    _CUPY = False
    print("ℹ️ CuPy not available; mean projection will use CPU")


_HAS_UMAT = hasattr(cv2, "UMat")


try:
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    from scipy import ndimage as ndi
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False
    print("skimage/scipy not found; large-ROI splitting will be limited")





def load_movie(movie_path: str, dataset_name: Optional[str] = None):
    """
    Load a movie from various formats.

    Supported formats:
      - NumPy (.npy, .npz): returns a numpy memmap/ndarray
      - Video (.avi, .mp4, .mov, .mkv): returns a cv2.VideoCapture stream

    Returns
    -------
    movie_handle : ndarray (mmap) or cv2.VideoCapture
    """
    ext = os.path.splitext(movie_path)[1].lower()

    if ext in (".npy", ".npz"):

        try:
            return np.load(movie_path, mmap_mode="r", allow_pickle=False)
        except Exception:
            print("Falling back to allow_pickle=True for numpy load")
            return np.load(movie_path, mmap_mode="r", allow_pickle=True)

    if ext in (".avi", ".mp4", ".mov", ".mkv", ".m4v", ".mjpeg", ".mpg", ".mpeg"):
        cap = cv2.VideoCapture(movie_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {movie_path}")
        return cap

    raise ValueError(f"Unsupported movie format: {ext}")





class _Perf:
    def __init__(self, name: str):
        self.name = name
        self.t0 = None
        self.mem0 = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        try:
            import psutil
            self.mem0 = psutil.Process().memory_info().rss / 1024 / 1024
        except Exception:
            self.mem0 = 0.0
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            import psutil
            mem1 = psutil.Process().memory_info().rss / 1024 / 1024
            dmem = mem1 - self.mem0
            print(f"⏱️ {self.name}: {time.perf_counter() - self.t0:.3f}s, ΔMem {dmem:+.1f} MB")
        except Exception:
            print(f"⏱️ {self.name}: {time.perf_counter() - self.t0:.3f}s")





def compute_mean_projection(
    movie,
    calib_frames: int = 900,
    chunk_size: int = 50,
    use_gpu: bool = True
) -> np.ndarray:
    """
    Compute the mean image over the first `calib_frames` frames.

    Supports:
      - cv2.VideoCapture
      - numpy ndarray / memmap (N,H,W) or (N,H,W,1) or (N,H,W,3)

    Returns: float32 array (H, W)
    """
    with _Perf("Mean projection"):
        try:

            if hasattr(movie, "read"):
                cap = movie
                acc = None
                n = 0
                while n < calib_frames:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame.ndim == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    f = frame.astype(np.float32)
                    if acc is None:
                        acc = f
                    else:
                        acc += f
                    n += 1
                    if n and (n % 200 == 0):
                        print(f"  processed {n}/{calib_frames} frames…")
                try:
                    cap.release()
                except Exception:
                    pass
                if acc is None or n == 0:
                    raise RuntimeError("No frames read from video for projection")
                return (acc / float(n)).astype(np.float32, copy=False)


            arr = np.asarray(movie)
            if arr.ndim == 4 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            elif arr.ndim == 4 and arr.shape[-1] == 3:

                arr = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in arr], axis=0)

            if arr.ndim != 3:
                raise ValueError(f"Unsupported movie array shape: {arr.shape}")

            total = int(arr.shape[0])
            count = min(int(calib_frames), total)
            print(f"📊 Mean projection using {count}/{total} frames")

            if use_gpu and _CUPY:
                acc = cp.zeros(arr.shape[1:], dtype=cp.float32)
                for start in range(0, count, chunk_size):
                    stop = min(start + chunk_size, count)
                    acc += cp.asarray(arr[start:stop], dtype=cp.float32).sum(axis=0)
                out = cp.asnumpy(acc / float(count))
                return out.astype(np.float32, copy=False)


            acc = np.zeros(arr.shape[1:], dtype=np.float64)
            for start in range(0, count, chunk_size):
                stop = min(start + chunk_size, count)
                acc += arr[start:stop].astype(np.float32, copy=False).sum(axis=0, dtype=np.float64)
            return (acc / float(count)).astype(np.float32, copy=False)

        finally:
            gc.collect()





def save_rois(masks: List[np.ndarray], sizes: List[int], output_npz: str = "rois.npz") -> None:
    """
    Save ROI masks and sizes to disk in compressed format.
    """
    try:
        stack = np.stack([m.astype(np.uint8, copy=False) for m in masks])
        np.savez_compressed(output_npz, masks=stack, sizes=np.asarray(sizes, dtype=np.int32))
        print(f"Saved ROIs → {output_npz} (count={len(masks)})")
    except MemoryError:
        base, _ = os.path.splitext(output_npz)
        os.makedirs(base, exist_ok=True)
        for i, (mask, size) in enumerate(zip(masks, sizes)):
            np.savez_compressed(os.path.join(base, f"mask_{i:04d}.npz"),
                                mask=mask.astype(np.uint8, copy=False),
                                size=int(size))
        print(f"Large ROI set; saved individual masks in directory: {base}")





def denoise_and_threshold_gpu(
    mean_img: np.ndarray,
    gauss_ksize: Tuple[int, int] = (3, 3),
    gauss_sigma: float = 0.5,
    min_area: int = 5,
    max_area: int = 200,
    use_gpu: bool = True,
    threshold_method: str = "otsu",  
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Segment ROIs from a mean image.

    Steps:
      1) Gaussian blur (UMat GPU if available)
      2) Normalize to 8-bit
      3) Threshold (Otsu by default; 'adaptive' also available)
      4) Morphological cleanup
      5) Connected components
      6) Optionally split large regions via distance-transform/watershed (CPU)

    Returns:
      masks: list of boolean HxW arrays
      sizes: list of pixel counts
    """
    with _Perf("ROI segmentation"):

        img = np.asarray(mean_img)
        if img.ndim != 2:
            raise ValueError(f"mean_img must be 2D; got {img.shape}")
        if img.dtype != np.float32:
            img = img.astype(np.float32, copy=False)


        src = cv2.UMat(img) if (use_gpu and _HAS_UMAT) else img
        blur = cv2.GaussianBlur(src, gauss_ksize, sigmaX=float(gauss_sigma))


        norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)


        if threshold_method.lower() == "adaptive":
            bw = cv2.adaptiveThreshold(
                norm, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                blockSize=21, C=0
            )
        else:

            if hasattr(norm, "get"):
                norm_cpu = norm.get()
            else:
                norm_cpu = norm
            _, bw = cv2.threshold(norm_cpu, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k3)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k5)


        if hasattr(bw, "get"):
            bw = bw.get()
        bw = bw.astype(np.uint8, copy=False)


        num_labels, labels_img = cv2.connectedComponents(bw, connectivity=8)
        print(f"🔍 Found {max(0, num_labels - 1)} initial ROIs")

        masks: List[np.ndarray] = []
        sizes: List[int] = []

        if num_labels <= 1:
            print("No ROIs found (post-threshold).")
            return masks, sizes


        for lab in range(1, num_labels):
            mask = (labels_img == lab)
            area = int(mask.sum())
            if area < int(min_area):
                continue

            if max_area and area > int(max_area) and _HAS_SKIMAGE:

                dist = cv2.distanceTransform((mask.astype(np.uint8) * 255), cv2.DIST_L2, 5)
                coords = peak_local_max(dist, min_distance=5, labels=mask)
                if coords.size == 0:

                    masks.append(mask)
                    sizes.append(area)
                    continue
                peaks = np.zeros_like(dist, dtype=bool)
                peaks[coords[:, 0], coords[:, 1]] = True
                markers = ndi.label(peaks)[0]
                labels_ws = watershed(-dist, markers, mask=mask)
                for lab_ws in np.unique(labels_ws):
                    if lab_ws == 0:
                        continue
                    submask = (labels_ws == lab_ws)
                    s = int(submask.sum())
                    if s >= int(min_area):
                        masks.append(submask)
                        sizes.append(s)
            else:
                masks.append(mask)
                sizes.append(area)

        print(f"✅ Extracted {len(masks)} ROIs after cleanup/splitting")
        return masks, sizes
