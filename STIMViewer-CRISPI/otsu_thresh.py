# otsu_thresh.py
import os
import numpy as np
import cv2
import time

from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi

# ===============================================
# Core ROI discovery pipeline with instrumentation & splitting of large ROIs
# ===============================================

def load_movie(movie_path, dataset_name=None):
    """
    Load a movie from various formats.

    Supported formats:
      - HDF5 (.h5): returns (file_handle, dataset) for chunked reads
      - NumPy (.npy, .npz): returns a memmap ndarray
      - Video (.avi, .mp4, .mov, .mkv): returns a cv2.VideoCapture stream

    Parameters
    ----------
    movie_path : str
        Path to the movie file.
    dataset_name : str or None
        Name of the HDF5 dataset inside .h5 (ignored for other formats).

    Returns
    -------
    movie_handle : tuple or ndarray or cv2.VideoCapture
        A handle for subsequent processing:
          - HDF5: (h5py.File, h5py.Dataset)
          - NumPy: numpy.ndarray (mmap)
          - Video: cv2.VideoCapture
    """
    ext = os.path.splitext(movie_path)[1].lower()

    if ext in ('.npy', '.npz'):
        return np.load(movie_path, mmap_mode='r')
    elif ext in ('.avi', '.mp4', '.mov', '.mkv'):
        cap = cv2.VideoCapture(movie_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {movie_path}")
        return cap
    else:
        raise ValueError(f"Unsupported movie format: {ext}")


def compute_mean_projection(
    movie,
    calib_frames=900,
    chunk_size=5 # how many frames to sum at once
):
    """
    Compute the average (mean) image over the first `calib_frames` frames.

    Handles three types of inputs:
      1) HDF5-backed tuple     -> chunked slice summation
      2) VideoCapture stream    -> frame-by-frame accumulation
      3) NumPy array (mmap)     -> chunked slice summation

    Parameters
    ----------
    movie : tuple or cv2.VideoCapture or ndarray
        Movie handle returned by load_movie.
    calib_frames : int
        Number of frames to include in the projection. Reduce if memory-bound.
    chunk_size : int
        Number of frames to read at once when summing (HDF5/ndarray case).

    Returns
    -------
    mean_img : 2D float array
        The mean image (H x W) for thresholding.
    """
    if isinstance(movie, tuple):  # HDF5
        f, dset = movie
        total = dset.shape[0]
        count = min(calib_frames, total)
        acc = np.zeros(dset.shape[1:], dtype=np.float64)
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            acc += dset[start:stop].sum(axis=0)
        f.close()
        return acc / count

    if isinstance(movie, cv2.VideoCapture):  # Video stream
        cap = movie
        acc = None
        frames_read = 0
        while frames_read < calib_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame.astype(np.float64)
            acc = frame if acc is None else acc + frame
            frames_read += 1
        cap.release()
        if acc is None or frames_read == 0:
            raise RuntimeError("No frames read from video for projection")
        return acc / frames_read

    # NumPy memmap
    arr = movie
    total = arr.shape[0]
    count = min(calib_frames, total)
    acc = np.zeros(arr.shape[1:], dtype=np.float64)
    print(arr.shape)
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        acc += arr[start:stop].sum(axis=0)
    return acc / count




def save_rois(masks, sizes, output_npz='rois.npz'):
    """
    Save ROI masks and sizes to disk in compressed format.

    If stacking fails due to memory constraints, save each mask separately.

    Parameters
    ----------
    masks : list of bool arrays
        ROI masks to save.
    sizes : list of int
        Corresponding pixel counts for each mask.
    output_npz : str
        Filename for the .npz archive or directory base.
    """
    try:
        stack = np.stack(masks).astype(np.uint8)
        np.savez_compressed(output_npz, masks=stack, sizes=sizes)
    except MemoryError:
        base, _ = os.path.splitext(output_npz)
        os.makedirs(base, exist_ok=True)
        for i, mask in enumerate(masks):
            fname = os.path.join(base, f"mask_{i:04d}.npz")
            np.savez_compressed(fname,
                                 mask=mask.astype(np.uint8),
                                 size=sizes[i])


def denoise_and_threshold_gpu(
    mean_img,
    gauss_ksize=(3,3),
    gauss_sigma=0.5,
    min_area=5,
    max_area=200
):
    """
    GPU-accelerated ROI segmentation using OpenCV UMat for blur, threshold, morphology, distance transform.
    """
    t0 = time.perf_counter()
    # Upload to GPU (UMat)
    umat = cv2.UMat(mean_img.astype(np.float32))

    # 1) Gaussian blur on GPU
    blur_umat = cv2.GaussianBlur(
        umat,
        gauss_ksize,
        sigmaX=gauss_sigma
    )

    # 2) Normalize to 8-bit on GPU
    blur_norm_umat = cv2.normalize(
        blur_umat, None, 0, 255,
        cv2.NORM_MINMAX, dtype=cv2.CV_8UC1
    )

    # 3) Adaptive threshold on GPU
    bw_umat = cv2.adaptiveThreshold(
        blur_norm_umat,
        maxValue=1,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=21,# width and height of square patch around each pixel
        C=0 # larger means pixels need to be brighter above the local mean so pick up less pixels
    )

    # 4) Morphological cleanup on GPU
    kern3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    kern5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    bw_umat = cv2.morphologyEx(bw_umat, cv2.MORPH_OPEN,  kern3)
    bw_umat = cv2.morphologyEx(bw_umat, cv2.MORPH_CLOSE, kern5)

    # Download to CPU for connected components & watershed
    bw = bw_umat.get().astype(np.uint8)

    # Connected components (CPU)
    num_labels, labels_img = cv2.connectedComponents(bw, connectivity=8)
    masks, sizes = [], []
    for lab in range(1, num_labels):
        mask = (labels_img == lab)
        area = int(mask.sum())
        if area < min_area:
            continue
        # Split oversized via distance transform & watershed
        if max_area and area > max_area:
            # Distance transform on GPU
            dist_umat = cv2.distanceTransform(cv2.UMat(mask.astype(np.uint8)*255), cv2.DIST_L2, 5)
            dist = dist_umat.get()

            # Peak local max (CPU)
            coords = peak_local_max(dist, min_distance=5, labels=mask)
            peaks = np.zeros_like(dist, dtype=bool)
            peaks[coords[:,0], coords[:,1]] = True
            markers = ndi.label(peaks)[0]

            labels_ws = watershed(-dist, markers, mask=mask)
            for lab_ws in np.unique(labels_ws):
                if lab_ws == 0:
                    continue
                submask = (labels_ws == lab_ws)
                if submask.sum() >= min_area:
                    masks.append(submask)
                    sizes.append(int(submask.sum()))
        else:
            masks.append(mask)
            sizes.append(area)
        # Instrumentation: end timer
    t1 = time.perf_counter()
    print(f"Denoise+threshold+split took {(t1-t0)*1000:.1f} ms (CPU)")
    return masks, sizes
    # return labels_img