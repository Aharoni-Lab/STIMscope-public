
import numpy as np
import cv2
import time
import gc
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple, Optional
import logging

from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


try:
    import cupy as cp
    CUDA_AVAILABLE = True
    print("✅ CUDA/GPU acceleration available for roi_thresh")
except ImportError:
    CUDA_AVAILABLE = False
    print("⚠️ CUDA not available for roi_thresh, using CPU")


class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.memory_before = None
        self.memory_after = None
    
    def start(self):
        self.start_time = time.perf_counter()
        try:
            import psutil
            self.memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            self.memory_before = 0
    
    def end(self, operation_name: str):
        if self.start_time is None:
            return
        
        duration = time.perf_counter() - self.start_time
        try:
            import psutil
            self.memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_diff = self.memory_after - self.memory_before
            print(f"⏱️ {operation_name}: {duration:.3f}s, Memory: {memory_diff:+.1f}MB")
        except ImportError:
            print(f"⏱️ {operation_name}: {duration:.3f}s")
        
        self.start_time = None







def threshold_patch(img, gauss_ksize=(3,3), gauss_sigma=0.5, min_area=5, max_area=200, use_gpu=True):
   
    monitor = PerformanceMonitor()
    monitor.start()
    
    try:

        if use_gpu and CUDA_AVAILABLE:
            print("🚀 Using GPU acceleration for ROI thresholding")

            img_gpu = cp.asarray(img.astype(np.float32))
            

            blur_gpu = cv2.GaussianBlur(cp.asnumpy(img_gpu), gauss_ksize, gauss_sigma)
            blur_gpu = cp.asarray(blur_gpu)
            

            norm_gpu = cv2.normalize(cp.asnumpy(blur_gpu), None, 0, 255, cv2.NORM_MINMAX)
            norm_gpu = cp.asarray(norm_gpu)
            

            bw_gpu = cv2.adaptiveThreshold(
                cp.asnumpy(norm_gpu.astype('uint8')), 1,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY, 21, 0
            )
            bw = cp.asarray(bw_gpu)
        else:
            print("💻 Using CPU processing for ROI thresholding")
            blur = cv2.GaussianBlur(img.astype(np.float32), gauss_ksize, gauss_sigma)
            norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)
            bw = cv2.adaptiveThreshold(
                norm.astype('uint8'), 1,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY, 21, 0
            )


        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

        bw_np = cp.asnumpy(bw) if hasattr(bw, 'get') else bw
        bw_np = cv2.erode(bw_np, k3, iterations=1)
        bw_np = cv2.morphologyEx(bw_np, cv2.MORPH_OPEN, k3)

        bw = cp.asarray(bw_np) if use_gpu and CUDA_AVAILABLE else bw_np
        

        bw_np = cp.asnumpy(bw) if hasattr(bw, 'get') else bw
        num, lab = cv2.connectedComponents(bw_np, 8)
        print(f"🔍 Found {num-1} initial ROIs in threshold_patch")
        
        masks, sizes = [], []
        processed_count = 0
        
        for lab_id in range(1, num):
            m = lab == lab_id
            pix = int(m.sum())
            
            if pix < min_area:
                continue
                
            if max_area and pix > max_area:

                if use_gpu and CUDA_AVAILABLE:
                    dist = cv2.distanceTransform(cp.asnumpy(m.astype('uint8')*255), cv2.DIST_L2, 5)
                else:
                    dist = cv2.distanceTransform(m.astype('uint8')*255, cv2.DIST_L2, 5)
                

                coords = peak_local_max(dist, min_distance=4, labels=m)
                peaks = np.zeros_like(dist, bool)
                peaks[tuple(coords.T)] = True
                labels_ws = watershed(-dist, ndi.label(peaks)[0], mask=m)
                
                for sub in np.unique(labels_ws)[1:]:
                    submask = labels_ws == sub
                    if submask.sum() >= min_area:
                        masks.append(submask)
                        sizes.append(int(submask.sum()))
                        processed_count += 1
            else:
                masks.append(m)
                sizes.append(pix)
                processed_count += 1
        
        print(f"✅ Extracted {len(masks)} final ROIs from {num-1} initial candidates")
        monitor.end("ROI thresholding")
        return masks, sizes
        
    except Exception as e:
        print(f"Error in threshold_patch: {e}")
        return [], []
    finally:

        gc.collect()
        print("ROI thresholding cleanup completed.")
