import os
import sys
if sys.version_info < (3, 8):
    try:
        import importlib_metadata as importlib_metadata  # type: ignore
        # Provide backport under stdlib name expected by some packages
        sys.modules['importlib.metadata'] = importlib_metadata
    except Exception:
        # Will likely fail later when importing cellpose; user can install:
        # pip install importlib-metadata
        pass
import argparse
import numpy as np
import cv2


def _read_stack_tiff_max_projection(path: str) -> np.ndarray:
    try:
        import tifffile
    except Exception as e:
        raise RuntimeError(f"tifffile required for TIFF input: {e}")
    arr = tifffile.imread(path)
    if arr.ndim < 2:
        raise ValueError(f"Unexpected TIFF shape: {arr.shape}")
    if arr.ndim == 2:
        img = arr.astype(np.float32, copy=False)
    else:
        # assume (T,H,W[,C])
        img = np.max(arr, axis=0).astype(np.float32, copy=False)
    return img


def _read_video_mean_projection(path: str, calib_frames: int = 900) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {path}")
    acc = None
    n = 0
    try:
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
    finally:
        try:
            cap.release()
        except Exception:
            pass
    if acc is None or n == 0:
        raise RuntimeError("No frames read from video for projection")
    return (acc / float(n)).astype(np.float32, copy=False)


def _read_npy_projection(path: str, use_mean: bool = True, calib_frames: int = 900) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr.astype(np.float32, copy=False)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    elif arr.ndim == 4 and arr.shape[-1] == 3:
        arr = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in arr], axis=0)
    if arr.ndim != 3:
        raise ValueError(f"Unsupported array shape: {arr.shape}")
    if use_mean:
        count = min(int(calib_frames), int(arr.shape[0]))
        acc = arr[:count].astype(np.float32, copy=False).sum(axis=0)
        return (acc / float(count)).astype(np.float32, copy=False)
    else:
        return np.max(arr, axis=0).astype(np.float32, copy=False)


def _build_clahe_image(img: np.ndarray) -> np.ndarray:
    norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(norm).astype(np.float32)


def _remap_labels_contiguous(labels: np.ndarray) -> np.ndarray:
    ids = np.unique(labels)
    ids = ids[ids > 0]
    if ids.size == 0:
        return labels.astype(np.int32, copy=False)
    id_map = {old: new for new, old in enumerate(ids, start=1)}
    out = np.zeros_like(labels, dtype=np.int32)
    for old, new in id_map.items():
        out[labels == old] = new
    return out


def run_cellpose(clahe_img: np.ndarray,
                 model_path: str = None,
                 size_path: str = None,
                 diameter: float = 18.0,
                 flow_threshold: float = 0.5,
                 cellprob_threshold: float = 0.0) -> np.ndarray:
    from cellpose import models
    if model_path and os.path.exists(model_path):
        model = models.CellposeModel(
            gpu=True,
            pretrained_model=model_path,
            model_type=None,
            net_avg=False
        )
        if size_path and os.path.exists(size_path):
            model.sz = models.SizeModel(model, pretrained_size=size_path)
    else:
        # fallback to built-in model type
        model = models.Cellpose(gpu=True, model_type='cyto')

    masks, styles, flows = model.eval(
        [clahe_img],
        channels=[0, 0],
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold
    )
    if isinstance(masks, (list, tuple)):
        lab = masks[0]
    else:
        lab = masks
    return lab.astype(np.int32, copy=False)


def save_rois_npz(labels: np.ndarray, out_path: str) -> None:
    labels = labels.astype(np.int32, copy=False)
    labels = _remap_labels_contiguous(labels)
    max_id = int(labels.max(initial=0))
    masks_list = [(labels == i) for i in range(1, max_id + 1)]
    sizes = [int(m.sum()) for m in masks_list]
    np.savez_compressed(
        out_path,
        masks=np.asarray(masks_list, dtype=np.uint8),
        sizes=np.asarray(sizes, dtype=np.int32),
        labels=labels
    )


def main():
    p = argparse.ArgumentParser(description="Run Cellpose on selected video and emit rois.npz")
    p.add_argument('--video', required=True, help='Input video (tiff stack, npy/npz, or standard video)')
    p.add_argument('--out', required=True, help='Output rois.npz path')
    p.add_argument('--model', default=None, help='Path to custom cellpose model')
    p.add_argument('--size', default=None, help='Path to custom size model .npy')
    p.add_argument('--diameter', type=float, default=18.0)
    p.add_argument('--flow-threshold', type=float, default=0.5)
    p.add_argument('--cellprob-threshold', type=float, default=0.0)
    args = p.parse_args()

    vid_path = args.video
    ext = os.path.splitext(vid_path)[1].lower()

    if ext in ('.tif', '.tiff', '.ome.tif', '.ome.tiff'):
        proj = _read_stack_tiff_max_projection(vid_path)
    elif ext in ('.npy', '.npz'):
        proj = _read_npy_projection(vid_path, use_mean=True)
    else:
        proj = _read_video_mean_projection(vid_path, calib_frames=900)

    clahe_img = _build_clahe_image(proj)

    # Optional resize hook: keep original
    labels = run_cellpose(
        clahe_img,
        model_path=args.model,
        size_path=args.size,
        diameter=args.diameter,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
    )

    save_rois_npz(labels, args.out)
    print(f"✅ Saved ROIs → {args.out}")


if __name__ == '__main__':
    main()


