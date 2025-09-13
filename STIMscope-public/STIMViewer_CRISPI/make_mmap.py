
from __future__ import annotations
import os
import sys
from typing import Tuple

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None 

def _ensure_npy_suffix(path: str) -> str:
    return path if path.endswith(".npy") else f"{path}.npy"


def _as_gray(frame: np.ndarray) -> np.ndarray:
   
    if frame is None:
        raise ValueError("Invalid frame (None)")
    if frame.ndim == 2: 
        return frame.astype(np.float32, copy=False)
    if frame.ndim == 3:
        if cv2 is None:
            return frame[..., 0].astype(np.float32, copy=False)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32, copy=False)
    raise ValueError(f"Unexpected frame shape: {frame.shape}")


def _cap_frame_count(cap) -> int:
    if cv2 is None:
        return -1
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return n if n > 0 and n < 10_000_000 else -1 


def _cap_wh(cap) -> Tuple[int, int]:
    if cv2 is None:
        return (0, 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return (w, h)


def _symlink_or_copy(src: str, dst: str):
   
    try:
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)
        print(f"Memmap path points to existing numpy file via symlink: {dst} -> {src}")
        return
    except Exception:
        pass
    try:
        import shutil
        shutil.copy2(src, dst)
        print(f"Copied existing numpy file to memmap path: {dst}")
    except Exception as e:
        print(f"Failed to duplicate source numpy file: {e}")
        raise


def make_memmap(video_path: str, memmap_path: str) -> Tuple[int, int, int]:
    """
    Create a memory-mapped .npy (C-order float32, shape=(N,H,W)) from a movie.

    Supports:
      - Standard video via OpenCV (mp4/avi/…)
      - Existing .npy / .npz inputs (symlink instead of re-encoding when possible)

    Returns:
      (N, H, W) tuple of the created memmap array (or referenced one).
    """
    if not video_path:
        raise ValueError("video_path must be provided")

    memmap_path = _ensure_npy_suffix(memmap_path)

    ext = os.path.splitext(video_path)[1].lower()
    if ext in (".npy", ".npz"):
        try:
            arr = np.load(video_path, mmap_mode="r")
            if isinstance(arr, np.lib.npyio.NpzFile):
                keys = list(arr.keys())
                if not keys:
                    raise ValueError("Empty .npz file")
                data = arr[keys[0]]
            else:
                data = arr

            if data.ndim < 2:
                raise ValueError(f"Unsupported array shape in {video_path}: {data.shape}")
            if data.ndim == 4 and data.shape[-1] == 1:
                data = data[..., 0]
            if data.ndim != 3:
                raise ValueError(f"Expect 3D array (N,H,W); got {data.shape}")

            _symlink_or_copy(video_path, memmap_path)
            shape = tuple(int(x) for x in data.shape)
            print(f"Using existing numpy movie as memmap: shape={shape}")
            return shape
        except Exception as e:
            print(f"Could not use {video_path} directly as memmap ({e}); will re-encode.")

    from otsu_thresh import load_movie

    movie = load_movie(video_path)

    if isinstance(movie, np.ndarray):
        arr = movie
        if arr.ndim == 4 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.ndim == 4 and arr.shape[-1] == 3:
            if cv2 is None:
                arr = arr[..., 0]
            else:
                arr = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in arr], axis=0)
        if arr.ndim != 3:
            raise ValueError(f"Unsupported array shape from loader: {arr.shape}")

        N, H, W = map(int, arr.shape)
        if os.path.exists(memmap_path):
            try:
                os.remove(memmap_path)
            except Exception:
                pass
        mm = np.lib.format.open_memmap(memmap_path, mode="w+", dtype=np.float32, shape=(N, H, W))
        CHUNK = max(1, 512 // max(H * W / (1024*1024), 1))  # heuristic
        for i in range(0, N, CHUNK):
            mm[i:i+CHUNK] = arr[i:i+CHUNK].astype(np.float32, copy=False)
        del mm  
        print(f"Memmap created: {memmap_path} shape=({N},{H},{W})")
        return (N, H, W)

    if hasattr(movie, "read") and cv2 is not None:
        cap = movie
        try:
            frame_count = _cap_frame_count(cap)
            if frame_count <= 0:
                print("Frame count unknown; doing a counting pass…")
                count = 0
                while True:
                    ok, _ = cap.read()
                    if not ok:
                        break
                    count += 1
                cap.release()
                cap = load_movie(video_path)  
                frame_count = count
            if frame_count <= 0:
                raise ValueError("Could not determine frame count")

            w, h = _cap_wh(cap)
            if w <= 0 or h <= 0:
                ok, frame = cap.read()
                if not ok:
                    raise ValueError("Could not read first frame")
                g = _as_gray(frame)
                h, w = g.shape
                cap.release()
                cap = load_movie(video_path) 

            if os.path.exists(memmap_path):
                try:
                    os.remove(memmap_path)
                except Exception:
                    pass
            mm = np.lib.format.open_memmap(memmap_path, mode="w+", dtype=np.float32, shape=(frame_count, h, w))

            i = 0
            report_next = 100
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                gray = _as_gray(frame)
                if gray.shape != (h, w):
                    if cv2 is not None:
                        gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
                    else:
                        raise ValueError("Frame size changed and OpenCV unavailable for resize")
                mm[i] = gray
                i += 1
                if i >= report_next:
                    print(f"Processed {i}/{frame_count} frames…")
                    report_next += 100

            if i != frame_count:
                print(f"Expected {frame_count} frames but read {i}; rewriting memmap header to shrink.")
                tmp = memmap_path + ".tmp"
                mm.flush(); del mm
                src = np.load(memmap_path, mmap_mode="r")
                mm2 = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(i, h, w))
                mm2[:] = src[:i]
                del mm2, src
                os.replace(tmp, memmap_path)

            print(f"Memmap created: {memmap_path} shape=({i},{h},{w})")
            return (i, h, w)
        finally:
            try:
                cap.release()
            except Exception:
                pass

    raise TypeError(f"Unsupported movie object from loader: {type(movie)}")
