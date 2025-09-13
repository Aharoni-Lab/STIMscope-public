
from __future__ import annotations

import os
import gc
import time
import queue
import threading
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import psutil
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", "pkg_resources is deprecated", DeprecationWarning)
import pygame
import cv2

from PyQt5.QtCore import QObject, pyqtSignal, QThread, pyqtSlot, Qt
from PyQt5.QtGui import QImage
try:
    import pyqtgraph as pg
    PYQTPGRAPH_AVAILABLE = True
except Exception:
    PYQTPGRAPH_AVAILABLE = False
    pg = None

try:
    import cupy as cp
    CUDA_AVAILABLE = True
    print("✅ CUDA/CuPy available for live_trace_extractor")
except Exception:
    CUDA_AVAILABLE = False
    cp = None
    print("ℹ️ CUDA not available for live_trace_extractor; CPU path will be used")

MAX_FRAME_QUEUE_SIZE = 8
THREAD_POOL_SIZE = 1
SYNCHRONIZATION_TIMEOUT = 3.0
MEMORY_MONITORING_INTERVAL = 5
GPU_MEMORY_CLEANUP_INTERVAL = 15
JETSON_GPU_MEMORY_LIMIT = 0.60

def qimage_to_gray_np(qimg: QImage) -> np.ndarray:
    
    if qimg.isNull():
        raise ValueError("Null QImage")
    fmt = qimg.format()
    if fmt not in (QImage.Format_Grayscale8, QImage.Format_RGB888, QImage.Format_ARGB32, QImage.Format_RGBA8888):
        qimg = qimg.convertToFormat(QImage.Format_ARGB32)
        fmt = qimg.format()

    width = qimg.width()
    height = qimg.height()
    ptr = qimg.bits()
    ptr.setsize(qimg.byteCount())
    buf = np.frombuffer(ptr, dtype=np.uint8)

    if fmt == QImage.Format_Grayscale8:
        arr = buf.reshape((height, width))
        return arr.copy()

    if fmt in (QImage.Format_ARGB32, QImage.Format_RGBA8888):
        arr = buf.reshape((height, width, 4))
        gray = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
        return gray

    if fmt == QImage.Format_RGB888:
        arr = buf.reshape((height, width, 3))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        return gray

    qimg = qimg.convertToFormat(QImage.Format_Grayscale8)
    ptr = qimg.bits(); ptr.setsize(qimg.byteCount())
    return np.frombuffer(ptr, dtype=np.uint8).reshape((qimg.height(), qimg.width())).copy()



class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.memory_before = 0.0

    def start(self):
        self.start_time = time.perf_counter()
        try:
            self.memory_before = psutil.Process().memory_info().rss / 1024 / 1024
        except Exception:
            self.memory_before = 0.0

    def end(self, label: str):
        if self.start_time is None:
            return
        dt = time.perf_counter() - self.start_time
        try:
            mem_after = psutil.Process().memory_info().rss / 1024 / 1024
            print(f"⏱️ {label}: {dt:.3f}s, ΔMem {mem_after - self.memory_before:+.1f} MB")
        except Exception:
            print(f"⏱️ {label}: {dt:.3f}s")
        self.start_time = None



class SyncState(Enum):
    IDLE = "idle"
    INITIALIZING = "initializing"
    RECORDING = "recording"
    PROCESSING = "processing"
    PROJECTING = "projecting"
    STOPPING = "stopping"
    ERROR = "error"

@dataclass
class SyncInfo:
    state: SyncState
    timestamp: float
    frame_count: int
    memory_usage: float
    gpu_memory_usage: float
    error_message: Optional[str] = None



from concurrent.futures import ThreadPoolExecutor

class FrameProcessor(QThread):
    frame_processed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, max_workers: int = 1):
        super().__init__()
        self.frame_queue: "queue.Queue[Any]" = queue.Queue(maxsize=MAX_FRAME_QUEUE_SIZE)
        self.running = True
        self.pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="FrameProc")
        self.perf = PerformanceMonitor()
        self._frames = 0

    def add_frame(self, frame: Any):
        try:
            if self.frame_queue.qsize() > int(MAX_FRAME_QUEUE_SIZE * 0.8):
                drop = max(1, self.frame_queue.qsize() // 4)
                for _ in range(drop):
                    try: self.frame_queue.get_nowait()
                    except queue.Empty: break
                print(f"Frame queue high-watermark; dropped {drop} frames")
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            print("Frame queue full; skipping frame")
        except Exception as e:
            self.error_occurred.emit(f"Queue add error: {e}")

    def run(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                fut = self.pool.submit(self._process_one, frame)
                fut.add_done_callback(self._on_done)
            except queue.Empty:
                continue
            except Exception as e:
                self.error_occurred.emit(f"FrameProcessor error: {e}")

    def _process_one(self, frame: Any) -> dict:
        self.perf.start()
        try:
            if hasattr(frame, "get_numpy_1D"): 
                h, w = frame.Height(), frame.Width()
                arr4 = np.array(frame.get_numpy_1D(), dtype=np.uint8).reshape((h, w, 4))
                gray = arr4[..., 0]
            elif isinstance(frame, np.ndarray):
                if frame.ndim == 2:
                    gray = frame
                elif frame.ndim == 3 and frame.shape[2] >= 3:
                    gray = frame[..., 0]
                else:
                    raise ValueError("Unsupported ndarray shape")
            elif isinstance(frame, QImage):
                gray = qimage_to_gray_np(frame)
            else:
                raise ValueError("Unsupported frame type")

            self._frames += 1
            return {"frame": gray, "timestamp": time.time(), "frame_id": self._frames}
        finally:
            pass

    def _on_done(self, fut):
        try:
            res = fut.result()
            self.frame_processed.emit(res)
        except Exception as e:
            self.error_occurred.emit(f"Processing failure: {e}")

    def stop(self):
        self.running = False
        try:
            self.pool.shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass



class LiveTraceExtractor(QObject):
    update_plot_signal = pyqtSignal()
    gpu_memory_infoing = pyqtSignal(str)
    sync_state_changed = pyqtSignal(SyncInfo)
    performance_update = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(
         self,
        camera,
        label_path,
        plot_widget=None,
        max_points: int = 150,
        max_rois: int = 6,
        use_pygame_plot: bool = False,
        enable_sync: bool = False,
    ):
        super().__init__()
        
        self.camera = camera
        self.use_pygame_plot = bool(use_pygame_plot)
        self.enable_sync = bool(enable_sync)

        self._camera_signal_refs: List[Tuple[object, callable]] = []
        self._cleanup_event = threading.Event()
        self.plot_widget = None
        self._plot_curves = {}
        self._plot_timer = None
        self._labels_gpu = None

        self._frame_count = 0
        
        self._max_rois_cfg = max_rois
        self._update_every_n = self._calculate_update_throttle(max_rois)
        
        if max_rois <= 10:
            self._process_every_n = 1   
        elif max_rois <= 25:
            self._process_every_n = 2   
        elif max_rois <= 50:
            self._process_every_n = 3   
        else:
            self._process_every_n = 5  
        
        print(f"🚀 Performance optimized: update_throttle={self._update_every_n}, process_throttle={self._process_every_n} for {max_rois} ROIs")

        self.start_time = time.time()
        self.stats = {
            "frames_processed": 0,
            "frames_failed": 0,
            "memory_usage_peak": 0.0,
            "uptime_seconds": 0.0,
            "last_frame_time": 0.0,
            "gpu_memory_peak": 0.0,
            "sync_operations": 0,
            "sync_failures": 0,
        }

        self._sync_lock = threading.RLock()
        self._frame_lock = threading.Lock()
        self._gpu_lock = threading.Lock()

        self._sync_state = SyncState.IDLE
        self._syncprint = SyncInfo(self._sync_state, time.time(), 0, 0.0, 0.0, None)


        self.ids: np.ndarray = np.array([], dtype=np.int32)
        self.buffers: Dict[int, deque] = {}
        self._cpu_masks: Optional[List[np.ndarray]] = None  # list of boolean 1D masks
        self.mask_mat = None       
        self.roi_sizes = None        
        self._f_gpu = None           
        self._H = 0
        self._W = 0

        self.export_counter = 0



        self.update_plot_signal.connect(self._update_plot, Qt.QueuedConnection)
        if self.ids.size == 0:
            print("⚠️ No positive ROI labels found in labels array; running in empty-safe mode")

            self.ids = np.array([], dtype=np.int32)
            self.buffers = {}


        self._init_roi_processing(label_path, max_rois=max_rois, max_points=max_points)


        self._init_plotting(plot_widget)
        self.update_plot_signal.connect(self._update_plot)



        self.frame_processor = FrameProcessor(max_workers=THREAD_POOL_SIZE)
        self.frame_processor.frame_processed.connect(self._on_frame_processed, Qt.QueuedConnection)
        self.frame_processor.error_occurred.connect(self._on_processing_error, Qt.QueuedConnection)
        self.frame_processor.start()

        self._start_monitors()



        self._connect_camera_signals()

        self._update_sync_state(SyncState.INITIALIZING)
        print("🚀 LiveTraceExtractor initialized")



    def _init_roi_processing(self, label_path: str, max_rois: int, max_points: int):
        labels = np.load(label_path)["labels"].astype(np.int32)
        if labels.ndim != 2:
            raise ValueError("labels must be 2D")
        self._labels_orig = labels            
        self._roi_max = int(labels.max(initial=0))
        self._max_rois_cfg = max_rois
        self._max_points_cfg = max_points

        self._roi_ready = False

        self._ids_gpu = None
        self._roi_sizes_gpu = None
        self._f_gpu = None
        self._roi_sizes_cpu = None
        self._flat_labels_cpu = None
        self._max_label = 0
        self.ids = []

    def _limit_cuda_pools(self):
        try:
            mempool = cp.get_default_memory_pool()
            if hasattr(mempool, "set_limit"):
                mempool.set_limit(size=2**28)  # 256MB
                print("✅ CUDA memory pool limit set to 256MB")
            pmp = cp.get_default_pinned_memory_pool()
            if hasattr(pmp, "set_limit"):
                pmp.set_limit(size=2**28)
                print("✅ CUDA pinned memory pool limit set to 256MB")
        except Exception as e:
            print(f"Could not set CUDA pool limits: {e}")


    def _init_plotting(self, plot_widget=None):
        self._legend = None
        if self.use_pygame_plot:
            return
        if plot_widget is not None and PYQTPGRAPH_AVAILABLE:
            roi_count = len(self.ids)
            print(f"🎨 Setting up optimized plotting for {roi_count} ROIs...")
            

            if roi_count <= 20:
                self._setup_single_plot_layout(plot_widget, roi_count)
            else:
                self._setup_multi_plot_layout(plot_widget, roi_count)

        from PyQt5.QtCore import QTimer
        self._plot_timer = QTimer(self)
        

        camera_fps = self._detect_camera_fps()
        plot_interval_ms = int(1000 / camera_fps)
        
        self._plot_timer.setInterval(plot_interval_ms)
        self._plot_timer.timeout.connect(lambda: self.update_plot_signal.emit(), Qt.QueuedConnection)
        self._plot_timer.start()
        print(f"✅ Plot timer synchronized: {plot_interval_ms}ms for {camera_fps:.1f} fps (camera-matched)")

    def _detect_camera_fps(self):
        
        try:

            if hasattr(self.camera, 'get_actual_fps'):
                fps = self.camera.get_actual_fps()
                if fps and fps > 0:
                    print(f"🎥 Camera FPS detected via get_actual_fps(): {fps:.1f}")
                    return float(fps)
            

            if hasattr(self.camera, 'node_map') and self.camera.node_map:
                try:
                    fps_node = self.camera.node_map.FindNode("AcquisitionFrameRate")
                    if fps_node and fps_node.IsReadable():
                        fps = float(fps_node.Value())
                        if fps > 0:
                            print(f"🎥 Camera FPS detected via node map: {fps:.1f}")
                            return fps
                except Exception as e:
                    print(f"⚠️ Node map FPS detection failed: {e}")
            

            fps_attrs = ['fps', 'framerate', 'frame_rate', 'acquisition_fps']
            for attr in fps_attrs:
                if hasattr(self.camera, attr):
                    try:
                        fps = getattr(self.camera, attr)
                        if fps and fps > 0:
                            print(f"🎥 Camera FPS detected via {attr}: {fps:.1f}")
                            return float(fps)
                    except:
                        pass
            

            if hasattr(self.camera, 'get_fps'):
                try:
                    fps = self.camera.get_fps()
                    if fps and fps > 0:
                        print(f"🎥 Camera FPS detected via get_fps(): {fps:.1f}")
                        return float(fps)
                except:
                    pass
            

            print("⚠️ Could not detect camera FPS, using 30 fps default")
            return 30.0
            
        except Exception as e:
            print(f"❌ Camera FPS detection error: {e}, using 30 fps default")
            return 30.0

    def _calculate_update_throttle(self, max_rois):
       
        if max_rois <= 10:
            return 2  
        elif max_rois <= 25:
            return 3  
        elif max_rois <= 50:
            return 5 
        else:
            return 8 

    def _setup_single_plot_layout(self, plot_widget, roi_count):
       
        try:
            self.plot_widget = plot_widget
            self.plot_widget.setBackground('k')
            self.plot_widget.setDownsampling(auto=True, mode='peak')
            self.plot_widget.setClipToView(True)
            self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
            self.plot_widget.setMouseEnabled(x=True, y=True)


            self.plot_widget.setLabel('left', 'Intensity', units='AU')
            self.plot_widget.setLabel('bottom', 'Time Points', units='frames')


            self._legend = self.plot_widget.addLegend(offset=(10, 10))


            for idx, rid in enumerate(self.ids):

                unified_color = self._get_unified_roi_color(int(rid))
                pen = pg.mkPen(unified_color, width=2)

                curve = self.plot_widget.plot(pen=pen)
                self._plot_curves[int(rid)] = curve

            print(f"✅ Single plot layout complete for {roi_count} ROIs")

        except Exception as e:
            print(f"❌ Single plot setup failed: {e}")

    def _setup_multi_plot_layout(self, plot_widget, roi_count):
       
        try:

            parent_widget = plot_widget.parent() if plot_widget.parent() else plot_widget
            

            if hasattr(parent_widget, 'layout') or hasattr(parent_widget, 'setLayout'):
                self._setup_plot_with_external_legend(plot_widget, parent_widget, roi_count)
            else:

                self._setup_optimized_single_plot(plot_widget, roi_count)
                
        except Exception as e:
            print(f"❌ Multi-plot setup failed: {e}")

            self._setup_optimized_single_plot(plot_widget, roi_count)

    def _setup_plot_with_external_legend(self, plot_widget, parent_widget, roi_count):
       
        try:
            from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget, QLabel, QScrollArea
            from PyQt5.QtCore import Qt
            

            main_layout = QHBoxLayout()
            

            self.plot_widget = plot_widget
            self.plot_widget.setBackground('k')
            self.plot_widget.setDownsampling(auto=True, mode='peak')
            self.plot_widget.setClipToView(True)
            self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
            self.plot_widget.setMouseEnabled(x=True, y=True)
            

            self.plot_widget.setLabel('left', 'Intensity', units='AU')
            self.plot_widget.setLabel('bottom', 'Time Points', units='frames')
            

            legend_widget = QWidget()
            legend_widget.setMaximumWidth(200)
            legend_widget.setMinimumWidth(150)
            legend_layout = QVBoxLayout(legend_widget)
            

            header_label = QLabel(f"ROI Legend ({roi_count} ROIs)")
            header_label.setStyleSheet("font-weight: bold; color: white; background: #333; padding: 5px;")
            legend_layout.addWidget(header_label)
            

            scroll_area = QScrollArea()
            scroll_content = QWidget()
            scroll_layout = QVBoxLayout(scroll_content)
            

            for idx, rid in enumerate(self.ids):

                unified_color = self._get_unified_roi_color(int(rid))
                pen = pg.mkPen(unified_color, width=1)
                

                curve = self.plot_widget.plot(pen=pen)
                

                if roi_count > 30:
                    curve.setDownsampling(factor=2, auto=True, method='peak')
                
                self._plot_curves[int(rid)] = curve
                

                color_hex = unified_color
                legend_entry = QLabel(f"<span style='color: {color_hex}'>●</span> ROI {int(rid)}")
                legend_entry.setStyleSheet("color: white; padding: 2px; font-size: 10px;")
                scroll_layout.addWidget(legend_entry)
            
            scroll_area.setWidget(scroll_content)
            scroll_area.setWidgetResizable(True)
            legend_layout.addWidget(scroll_area)
            

            if hasattr(parent_widget, 'layout') and parent_widget.layout():

                parent_layout = parent_widget.layout()
                main_layout.addWidget(self.plot_widget, stretch=3)
                main_layout.addWidget(legend_widget, stretch=1)
                parent_layout.addLayout(main_layout)
            else:
                print("⚠️ Could not create external legend, using optimized single plot")
                self._setup_optimized_single_plot(plot_widget, roi_count)
                return
            
            print(f"✅ Multi-plot layout with external legend complete for {roi_count} ROIs")
            
        except Exception as e:
            print(f"❌ External legend setup failed: {e}")
            self._setup_optimized_single_plot(plot_widget, roi_count)

    def _setup_optimized_single_plot(self, plot_widget, roi_count):
       
        try:
            self.plot_widget = plot_widget
            self.plot_widget.setBackground('k')
            self.plot_widget.setDownsampling(auto=True, mode='peak')
            self.plot_widget.setClipToView(True)
            self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
            self.plot_widget.setMouseEnabled(x=True, y=True)
            

            self.plot_widget.setLabel('left', 'Intensity', units='AU')
            self.plot_widget.setLabel('bottom', 'Time Points', units='frames')
            

            print(f"📊 {roi_count} ROIs - using optimized mode without legend")
            

            for idx, rid in enumerate(self.ids):
                hue_count = min(15, max(8, roi_count))  
                color = pg.intColor(idx, hues=hue_count)
                pen = pg.mkPen(color, width=1)
                
                curve = self.plot_widget.plot(pen=pen)
                

                if roi_count > 25:
                    curve.setDownsampling(factor=3, auto=True, method='peak')
                
                self._plot_curves[int(rid)] = curve
            
            print(f"✅ Optimized single plot complete for {roi_count} ROIs")
            
        except Exception as e:
            print(f"❌ Optimized plot setup failed: {e}")


    def _start_monitors(self):

        if not hasattr(self, '_monitor_threads'):
            self._monitor_threads = []
        
        def perf_loop():
            thread_name = threading.current_thread().name
            print(f"🔄 Performance monitor thread started: {thread_name}")
            while not self._cleanup_event.is_set():
                try:
                    self._update_performance_stats()
                except Exception as e:
                    print(f"Performance monitor error: {e}")
                time.sleep(MEMORY_MONITORING_INTERVAL)
            print(f"🛑 Performance monitor thread stopping: {thread_name}")
        
        perf_thread = threading.Thread(target=perf_loop, daemon=True, name="PerfMonitor")
        perf_thread.start()
        self._monitor_threads.append(perf_thread)

        if CUDA_AVAILABLE:
            def gpu_loop():
                thread_name = threading.current_thread().name
                print(f"🔄 GPU monitor thread started: {thread_name}")
                while not self._cleanup_event.is_set():
                    try:
                        self._monitor_gpu_memory()
                    except Exception as e:
                        print(f"GPU monitor error: {e}")
                    time.sleep(MEMORY_MONITORING_INTERVAL)
                print(f"🛑 GPU monitor thread stopping: {thread_name}")
                
            gpu_thread = threading.Thread(target=gpu_loop, daemon=True, name="GPUMonitor")
            gpu_thread.start()
            self._monitor_threads.append(gpu_thread)


    def _connect_camera_signals(self):
        """
        Try several common signal names; prefer connecting to the generic on_frame(Object)
        to avoid Qt signature mismatches. Fall back to QImage-typed slot if needed.
        """
        connected = False

        candidates = (
            "image_update_signal", "frame_numpy", "frame_np",
            "frame_ready", "newFrame", "frame_signal", "new_qimage", "frame_qimage"
        )

        for name in candidates:
            try:
                sig = getattr(self.camera, name, None)
            except Exception:
                sig = None
            if sig is None:
                continue


            try:
                sig.connect(self.on_frame, Qt.QueuedConnection)
                self._camera_signal_refs.append((sig, self.on_frame))
                print(f"LiveTraceExtractor: connected to camera signal '{name}' → on_frame(object)")
                connected = True
                break
            except Exception:
                pass


            try:
                sig.connect(self._on_camera_qimage, Qt.QueuedConnection)
                self._camera_signal_refs.append((sig, self._on_camera_qimage))
                print(f"LiveTraceExtractor: connected to camera signal '{name}' → _on_camera_qimage(QImage)")
                connected = True
                break
            except Exception:
                pass


        if not connected:
            cb = getattr(self.camera, "register_consumer", None)
            if callable(cb):
                try:
                    cb(self.on_frame)
                    print("LiveTraceExtractor: registered camera consumer callback")
                    connected = True
                except Exception as e:
                    print(f"register_consumer failed: {e}")

        if not connected:
            print("LiveTraceExtractor: could not connect to camera; waiting for manual feed (on_frame)")


    def _disconnect_camera_signals(self):
        for sig, slot in list(getattr(self, "_camera_signal_refs", [])):
            try:
                sig.disconnect(slot)
            except Exception:
                pass
        if hasattr(self, "_camera_signal_refs"):
            self._camera_signal_refs.clear()



    @pyqtSlot(object)
    def _on_camera_frame(self, frame_obj: object):
        self.on_frame(frame_obj)

    @pyqtSlot(QImage)
    def _on_camera_qimage(self, qimg: QImage):
        try:
            arr = qimage_to_gray_np(qimg)
            self.on_frame(arr)
        except Exception as e:
            print(f"QImage→np conversion failed: {e}")

    def on_frame(self, frame):
       
        try:
            self.frame_processor.add_frame(frame)
        except Exception as e:
            print(f"Error queueing frame: {e}")
            self.error_occurred.emit(str(e))


    def _monitor_gpu_memory(self):
        if not CUDA_AVAILABLE:
            return
        mempool = cp.get_default_memory_pool()
        used = mempool.used_bytes()
        total = mempool.total_bytes()
        ratio = (used / total) if total else 0.0
        self.stats["gpu_memory_peak"] = max(self.stats["gpu_memory_peak"], ratio)
        if ratio > JETSON_GPU_MEMORY_LIMIT:
            msg = f"High GPU memory usage: {ratio:.1%} ({used/1024**2:.1f} MB)"
            print(msg)
            self.gpu_memory_infoing.emit(msg)
            self._cleanup_gpu_memory()

    def _cleanup_gpu_memory(self):
        if not CUDA_AVAILABLE:
            return
        with self._gpu_lock:
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception as e:
                print(f"GPU mempool free failed: {e}")

    def _update_performance_stats(self):
        self.stats["uptime_seconds"] = time.time() - self.start_time
        try:
            mem_mb = psutil.Process().memory_info().rss / 1024 / 1024
            self.stats["memory_usage_peak"] = max(self.stats["memory_usage_peak"], mem_mb)
        except Exception:
            pass
        self.performance_update.emit(self.stats.copy())

    def _on_frame_processed(self, processed_data: dict):
        try:

            if not isinstance(processed_data, dict) or 'frame' not in processed_data:
                print("⚠️ Invalid frame data received, skipping")
                return
                
            gray = processed_data['frame']
            

            if gray is None:
                print("⚠️ Received None frame, skipping")
                return
                
            if not hasattr(gray, 'shape') or len(gray.shape) < 2:
                print(f"⚠️ Invalid frame shape: {getattr(gray, 'shape', 'no shape')}, skipping")
                return
                
            H, W = gray.shape[:2]
            

            if H <= 0 or W <= 0 or H > 10000 or W > 10000:
                print(f"⚠️ Unreasonable frame dimensions {W}x{H}, skipping")
                return


            if not getattr(self, "_roi_ready", False):
                if not hasattr(self, '_labels_orig') or self._labels_orig is None:
                    print("⚠️ No ROI labels loaded, cannot process frame")
                    return
                    
                self._build_rois_for_shape(H, W)
                if not self._roi_ready or self.ids.size == 0:
                    return 


            self._proc_gate = (getattr(self, "_proc_gate", -1) + 1) % self._process_every_n
            if self._proc_gate: 

                self.stats['last_frame_time'] = time.time()
                return


            flat = gray.ravel().astype(np.float32, copy=False)


            if CUDA_AVAILABLE and hasattr(self, '_labels_gpu') and self._labels_gpu is not None:

                if not hasattr(self, '_roi_sizes_gpu') or self._roi_sizes_gpu is None:
                    print("⚠️ GPU ROI sizes not initialized, falling back to CPU")
                else:
                    with self._gpu_lock:
                        self._f_gpu.set(flat)
                        if not hasattr(self, '_max_label') or self._max_label is None:
                            self._max_label = int(self._labels_gpu.max().get())
                        sums = cp.bincount(
                            self._labels_gpu,
                            weights=self._f_gpu,
                            minlength=self._max_label + 1
                        )
                        den = cp.maximum(self._roi_sizes_gpu, 1e-6)
                        means = (sums[self._ids_gpu] / den).get()
                        
                        for val, rid in zip(means, self.ids):
                            rid_key = int(rid)
                            if rid_key not in self.buffers:
                                print(f"⚠️ GPU path: ROI {rid_key} not in buffers, creating...")
                                from collections import deque
                                self.buffers[rid_key] = deque(maxlen=self._max_points_cfg)
                            
                            try:
                                self.buffers[rid_key].append(float(val))
                            except Exception as e:
                                print(f"❌ GPU buffer error for ROI {rid_key}: {e}")
                        
                        self.stats['frames_processed'] += 1
                        self.stats['last_frame_time'] = time.time()
                        return
            else:

                if not hasattr(self, '_flat_labels_cpu') or self._flat_labels_cpu is None:
                    print("⚠️ CPU labels not initialized, skipping frame")
                    return
                if not hasattr(self, '_roi_sizes_cpu') or self._roi_sizes_cpu is None:
                    print("⚠️ CPU ROI sizes not initialized, attempting to initialize...")
                    try:
                        if hasattr(self, '_flat_labels_cpu') and self._flat_labels_cpu is not None:
                            if not hasattr(self, '_max_label') or self._max_label is None:
                                self._max_label = int(self._flat_labels_cpu.max(initial=0))
                            counts = np.bincount(self._flat_labels_cpu, minlength=self._max_label + 1)
                            self._roi_sizes_cpu = counts[self.ids].astype(np.float32)
                            print("✅ CPU ROI sizes initialized")
                        else:
                            print("⚠️ Cannot initialize ROI sizes, skipping frame")
                            return
                    except Exception as e:
                        print(f"⚠️ Failed to initialize ROI sizes: {e}, skipping frame")
                        return
                    
                sums = np.bincount(
                    self._flat_labels_cpu,
                    weights=flat,
                    minlength=self._max_label + 1
                )
                if self._roi_sizes_cpu is None:
                    print("⚠️ CPU ROI sizes still None after initialization attempt, skipping frame")
                    return
                den = np.maximum(self._roi_sizes_cpu, 1e-6)
                means = (sums[self.ids] / den)


            for val, rid in zip(means, self.ids):
                rid_key = int(rid)
                if rid_key not in self.buffers:
                    print(f"⚠️ ROI {rid_key} not in buffers, reinitializing buffers...")

                    from collections import deque
                    for missing_rid in self.ids:
                        missing_key = int(missing_rid)
                        if missing_key not in self.buffers:
                            self.buffers[missing_key] = deque(maxlen=self._max_points_cfg)
                            print(f"   ✅ Created buffer for ROI {missing_key}")
                
                try:
                    self.buffers[rid_key].append(float(val))
                except KeyError as e:
                    print(f"❌ Still missing buffer for ROI {rid_key}: {e}")

                    from collections import deque
                    self.buffers[rid_key] = deque(maxlen=self._max_points_cfg)
                    self.buffers[rid_key].append(float(val))
                    print(f"   🔧 Emergency buffer created for ROI {rid_key}")
                except Exception as e:
                    print(f"❌ Unexpected buffer error for ROI {rid_key}: {e}")


            self.stats['frames_processed'] += 1
            self.stats['last_frame_time'] = time.time()

        except Exception as e:
            self.stats['frames_failed'] += 1
            error_type = type(e).__name__
            error_msg = str(e)
            print(f"❌ Frame processing error [{error_type}]: {error_msg}")
            

            if hasattr(self, '_labels_orig') and self._labels_orig is not None:
                print(f"   Labels shape: {self._labels_orig.shape}")
            if hasattr(self, 'ids') and self.ids is not None:
                print(f"   Active ROIs: {len(self.ids)}")
            if hasattr(gray, 'shape'):
                print(f"   Frame shape: {gray.shape}")
            

            if "index" in error_msg.lower() or "shape" in error_msg.lower():
                print("🔧 Attempting ROI reinitialization due to indexing/shape error...")
                try:
                    if hasattr(gray, 'shape') and len(gray.shape) >= 2:
                        self._build_rois_for_shape(gray.shape[0], gray.shape[1])
                        print("✅ ROI reinitialization successful")
                        return  
                except Exception as recovery_error:
                    print(f"❌ ROI recovery failed: {recovery_error}")
            

            if self.stats['frames_failed'] % 10 == 0:  
                self.error_occurred.emit(f"Frame processing error [{error_type}]: {error_msg}")

    @pyqtSlot(str)
    def _on_processing_error(self, msg: str):
        print(f"Processing error: {msg}")
        self.error_occurred.emit(msg)


    @pyqtSlot()
    def _update_plot(self):
        try:
            if self.use_pygame_plot:
                self._update_pygame_plot()
            elif self.plot_widget is not None:
                self._update_pyqtgraph_plot()
        except Exception as e:
            print(f"Plot update error: {e}")

    def _update_pygame_plot(self):
        try:
            any_data = any(len(buf) > 1 for buf in self.buffers.values())
            if not any_data:
                return


            y_min = min(min(buf) for buf in self.buffers.values() if len(buf) > 0)
            y_max = max(max(buf) for buf in self.buffers.values() if len(buf) > 0)
            if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
                y_min, y_max = 0.0, 1.0 

            yr = y_max - y_min
            y_min -= 0.05 * yr
            y_max += 0.05 * yr

            self.screen.fill((0, 0, 0))
            margin = 50
            w = self.screen_width
            h = self.screen_height
            plot_w = w - 2 * margin
            plot_h = h - 2 * margin

            axis_color = (160, 160, 160)
            pygame.draw.rect(self.screen, axis_color, (margin-1, margin-1, plot_w+2, plot_h+2), 1)


            def to_xy(j, val, npoints):
                x = margin + int(j * (plot_w / max(1, npoints-1)))

                t = (val - y_min) / max(1e-6, (y_max - y_min))
                y = margin + (plot_h - int(t * plot_h))
                return x, y

            colors = [(255, 64, 64), (64, 255, 64), (64, 64, 255),
                    (255, 255, 64), (255, 64, 255), (64, 255, 255),
                    (200, 200, 200), (255, 128, 0)]

            for i, (rid, buf) in enumerate(self.buffers.items()):
                n = len(buf)
                if n < 2:
                    continue
                color = colors[i % len(colors)]

                pts = [to_xy(j, buf[j], n) for j in range(n)]
                pygame.draw.lines(self.screen, color, False, pts, 1)

            pygame.display.flip()
        except Exception as e:
            print(f"Error in pygame plotting: {e}")


    def _update_pyqtgraph_plot(self):
       
        if self.plot_widget is None:
            return
        try:
            roi_count = len(self.buffers)
            

            skip_factor = self._calculate_skip_factor(roi_count)
            if skip_factor > 1 and self._frame_count % skip_factor != 0:
                return
            
            self._update_paged_trace_mode()
                
        except Exception as e:
            print(f"❌ PyQtGraph plot update error: {e}")

    def _calculate_skip_factor(self, roi_count):
       
        if roi_count <= 10:
            return 1  
        elif roi_count <= 25:
            return 2 
        elif roi_count <= 50:
            return 3  
        else:
            return 5  

    def _update_paged_trace_mode(self):
       
        try:

            if getattr(self, '_is_shutting_down', False):
                return
            if hasattr(self, '_cleanup_event') and self._cleanup_event and self._cleanup_event.is_set():
                return

            if not self.plot_widget or not hasattr(self.plot_widget, 'plot'):
                return
            

            try:
                viewbox = self.plot_widget.getViewBox()
                if not viewbox:
                    self._plot_curves.clear()
                    return

                _ = viewbox.viewRange()
            except Exception as viewbox_error:
                print(f"⚠️ Plot widget invalid, clearing curves: {viewbox_error}")
                self._plot_curves.clear()
                return
            

            if not hasattr(self, '_trace_page_index'):
                self._trace_page_index = 0
                self._traces_per_page = 5
                self._setup_pagination_controls()
            

            active_rois = sorted([rid for rid, buf in self.buffers.items() if len(buf) >= 2])
            
            if not active_rois:
                return
            

            total_pages = max(1, (len(active_rois) + self._traces_per_page - 1) // self._traces_per_page)
            self._trace_page_index = min(self._trace_page_index, total_pages - 1)
            

            start_idx = self._trace_page_index * self._traces_per_page
            end_idx = min(start_idx + self._traces_per_page, len(active_rois))
            page_rois = active_rois[start_idx:end_idx]
            

            valid_curves = {}
            for roi_id, curve in list(self._plot_curves.items()):
                try:

                    if (hasattr(curve, 'setData') and 
                        hasattr(curve, 'clear') and 
                        not curve.__class__.__name__.endswith('_deleted')):
                        

                        try:
                            scene = curve.scene()
                            if scene is not None:
                                curve.clear()
                                valid_curves[roi_id] = curve
                            else:

                                pass
                        except Exception as scene_error:
                            if "deleted" not in str(scene_error).lower():
                                print(f"⚠️ Curve for ROI {roi_id}: scene access error: {scene_error}")
                    else:

                        pass
                except Exception as curve_error:
                    if "deleted" not in str(curve_error).lower():
                        print(f"⚠️ Curve error for ROI {roi_id}: {curve_error}")
            
            self._plot_curves = valid_curves
            if len(valid_curves) != len(self._plot_curves):
                print(f"🔄 Curve validation: {len(valid_curves)} valid curves retained")
            

            for i, roi_id in enumerate(page_rois):
                buffer = self.buffers.get(roi_id, [])
                if len(buffer) < 2:
                    continue
                
                try:
                    if roi_id not in self._plot_curves or not hasattr(self._plot_curves[roi_id], 'setData'):
                        if self.plot_widget and hasattr(self.plot_widget, 'plot'):
                            unified_color = self._get_unified_roi_color(roi_id)
                            pen = pg.mkPen(color=unified_color, width=2)
                            self._plot_curves[roi_id] = self.plot_widget.plot(pen=pen)
                        else:
                            continue
                    
                    x_data = np.arange(len(buffer), dtype=np.float32)
                    y_data = np.array(list(buffer), dtype=np.float32)
                    self._plot_curves[roi_id].setData(x=x_data, y=y_data)
                    
                except Exception as curve_error:
                    if roi_id in self._plot_curves:
                        del self._plot_curves[roi_id]
                    print(f"⚠️ Curve error for ROI {roi_id}: {curve_error}")
            

            for roi_id, curve in list(self._plot_curves.items()):
                if roi_id not in page_rois:
                    try:
                        if hasattr(curve, 'clear'):
                            curve.clear()
                    except Exception:

                        del self._plot_curves[roi_id]
            

            self._update_page_label_safe()

            self._update_legend_for_page(page_rois)
            

            self.plot_widget.autoRange()
            

            self._update_expanded_plot()
            
        except Exception as e:

            if "deleted" not in str(e).lower() and "viewbox" not in str(e).lower():
                print(f"❌ Paged trace mode error: {e}")

    def _update_legend_for_page(self, page_rois):
       
        try:

            if not hasattr(self, '_legend_layout') or not self._legend_layout:
                return
            

            if not hasattr(self, '_combined_legend_label') or self._combined_legend_label is None:
                from PyQt5.QtWidgets import QLabel
                from PyQt5.QtCore import Qt
                self._combined_legend_label = QLabel("ROI Legend")
                self._combined_legend_label.setStyleSheet("""
                    QLabel {
                        font-size: 10px; 
                        padding: 5px; 
                        color: #333;
                        background-color: #f8f8f8;
                        border: 1px solid #ddd;
                        border-radius: 3px;
                    }
                """)

                self._combined_legend_label.setTextFormat(Qt.RichText)
                self._legend_layout.addWidget(self._combined_legend_label)
            

            if page_rois:
                legend_text_parts = []
                for roi_id in page_rois:

                    if roi_id in self._plot_curves and hasattr(self._plot_curves[roi_id], 'opts'):
                        try:
                            curve_pen = self._plot_curves[roi_id].opts.get('pen', None)
                            if curve_pen and hasattr(curve_pen, 'color'):

                                curve_color = curve_pen.color()
                                color_hex = f"#{curve_color.red():02x}{curve_color.green():02x}{curve_color.blue():02x}"
                            else:

                                color_hex = self._get_unified_roi_color(roi_id)
                        except Exception:
                            color_hex = self._get_unified_roi_color(roi_id)
                    else:
                        color_hex = self._get_unified_roi_color(roi_id)
                    
                    legend_text_parts.append(f'<span style="color: {color_hex}; font-weight: bold;">● ROI {roi_id}</span>')
                
                legend_text = " | ".join(legend_text_parts)
            else:
                legend_text = "<span style='color: #666;'>No active traces</span>"
            

            self._combined_legend_label.setText(legend_text)
            
        except Exception as e:
            print(f"⚠️ Legend update error (suppressed): {e}")
            pass

    def _expand_all_rois(self):
       
        try:
            if not self.plot_widget:
                print("⚠️ No plot widget available for expansion")
                return
            

            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QScrollArea, QWidget
            import pyqtgraph as pg
            
            self._expanded_dialog = QDialog()
            self._expanded_dialog.setWindowTitle(f"All ROIs - Live Trace View ({len(self.buffers)} ROIs)")
            self._expanded_dialog.resize(1400, 900)
            
            layout = QVBoxLayout(self._expanded_dialog)
            

            header_layout = QHBoxLayout()
            header_label = QLabel(f"📊 Displaying all {len(self.buffers)} ROIs in real-time")
            header_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 10px;")
            
            close_btn = QPushButton("✖ Close Expanded View")
            close_btn.setMaximumWidth(200)
            close_btn.clicked.connect(self._expanded_dialog.close)
            
            header_layout.addWidget(header_label)
            header_layout.addStretch()
            header_layout.addWidget(close_btn)
            layout.addLayout(header_layout)
            

            scroll_area = QScrollArea()
            scroll_widget = QWidget()
            scroll_layout = QVBoxLayout(scroll_widget)
            

            self._expanded_plot = pg.PlotWidget()
            self._expanded_plot.setMinimumHeight(800)  
            self._expanded_plot.setLabel('left', 'Intensity') 
            self._expanded_plot.setLabel('bottom', 'Time (frames)')
            self._expanded_plot.showGrid(x=True, y=True, alpha=0.3)
            self._expanded_plot.setTitle(f"All {len(self.buffers)} ROIs - Live Traces (Optimized View)")
            

            viewbox = self._expanded_plot.getViewBox()
            viewbox.setAspectLocked(False)

            import pyqtgraph as pg
            viewbox.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
            

            self._expanded_curves = {}
            active_rois = sorted([rid for rid, buf in self.buffers.items() if len(buf) >= 2])
            

            if len(active_rois) > 10:

                all_traces = []
                for roi_id in active_rois:
                    buffer = list(self.buffers[roi_id])
                    if len(buffer) >= 2:
                        all_traces.append(np.array(buffer, dtype=np.float32))
                
                if all_traces:

                    global_min = min(np.min(trace) for trace in all_traces)
                    global_max = max(np.max(trace) for trace in all_traces)
                    trace_range = global_max - global_min if global_max > global_min else 1.0
                    

                    spacing = trace_range * 0.3  
                    
                    for i, roi_id in enumerate(active_rois):
                        buffer = list(self.buffers[roi_id])
                        if len(buffer) >= 2:
                            unified_color = self._get_unified_roi_color(roi_id)
                            pen = pg.mkPen(color=unified_color, width=1.0, alpha=0.7) 
                            
                            x_data = np.arange(len(buffer), dtype=np.float32)
                            y_data = np.array(buffer, dtype=np.float32)
                            

                            normalized_y = ((y_data - global_min) / trace_range) + (i * spacing)
                            
                            curve = self._expanded_plot.plot(x_data, normalized_y, pen=pen)
                            self._expanded_curves[roi_id] = curve
            else:

                for roi_id in active_rois:
                    buffer = list(self.buffers[roi_id])
                    if len(buffer) >= 2:
                        unified_color = self._get_unified_roi_color(roi_id)
                        pen = pg.mkPen(color=unified_color, width=1.5, alpha=0.8)
                        
                        x_data = np.arange(len(buffer), dtype=np.float32)
                        y_data = np.array(buffer, dtype=np.float32)
                        
                        curve = self._expanded_plot.plot(x_data, y_data, pen=pen)
                        self._expanded_curves[roi_id] = curve
            
            scroll_layout.addWidget(self._expanded_plot)
            

            legend_label = QLabel("ROI Legend (Colors match unified system):")
            legend_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
            scroll_layout.addWidget(legend_label)
            

            legend_layout = QHBoxLayout()
            legend_layout.setContentsMargins(10, 5, 10, 5)
            

            for i, roi_id in enumerate(active_rois):
                color = self._get_unified_roi_color(roi_id)
                legend_item = QLabel(f"● ROI {roi_id}")
                legend_item.setStyleSheet(f"color: {color}; font-weight: bold; margin: 2px; font-size: 10px;")
                legend_layout.addWidget(legend_item)
                
                if (i + 1) % 10 == 0:
                    scroll_layout.addLayout(legend_layout)
                    legend_layout = QHBoxLayout()
                    legend_layout.setContentsMargins(10, 5, 10, 5)
            
            if legend_layout.count() > 0:
                scroll_layout.addLayout(legend_layout)
            

            total_label = QLabel(f"Total: {len(active_rois)} ROIs displayed")
            total_label.setStyleSheet("font-weight: bold; color: #333; margin: 5px; font-size: 12px;")
            scroll_layout.addWidget(total_label)
            
            scroll_area.setWidget(scroll_widget)
            scroll_area.setWidgetResizable(True)
            layout.addWidget(scroll_area)
            

            self._expanded_dialog.show()
            

            self._update_expanded_plot()
            
            print(f"✅ Expanded view opened with {len(active_rois)} ROIs")
            
        except Exception as e:
            print(f"❌ Error creating expanded view: {e}")
            import traceback
            traceback.print_exc()

    def _update_expanded_plot(self):
       
        try:
            if not hasattr(self, '_expanded_dialog') or not hasattr(self, '_expanded_curves'):
                return
            
            if not self._expanded_dialog.isVisible():
                return
            

            for roi_id, curve in self._expanded_curves.items():
                if roi_id in self.buffers:
                    buffer = list(self.buffers[roi_id])
                    if len(buffer) >= 2:
                        try:
                            x_data = np.arange(len(buffer), dtype=np.float32)
                            y_data = np.array(buffer, dtype=np.float32)
                            curve.setData(x=x_data, y=y_data, skipFiniteCheck=True)
                        except Exception:
                            pass  
            

            if hasattr(self, '_expand_update_count'):
                self._expand_update_count += 1
            else:
                self._expand_update_count = 0
            
            if self._expand_update_count % 30 == 0:  
                self._expanded_plot.autoRange()
                
        except Exception as e:

            pass

    def _get_unified_roi_color(self, roi_id):
       

        colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
            '#DDA0DD', '#98D8C8', '#FFA07A', '#87CEEB', '#DEB887',  
            '#FF9F43', '#10AC84', '#EE5A24', '#0084FF', '#341F97', 
            '#F8B500', '#6C5CE7', '#A29BFE', '#FD79A8', '#FDCB6E', 
            '#E17055', '#00B894', '#00CECE', '#2D3436', '#636E72',  
            '#FAB1A0', '#74B9FF', '#55A3FF', '#FF7675', '#6C5CE7',  
        ]
        

        color_index = (roi_id - 1) % len(colors) 
        return colors[color_index]

    def _update_direct_overlay_mode(self):
       
        try:

            active_buffers = {}
            all_vals = []

            for rid, buf in self.buffers.items():
                if len(buf) == 0:
                    continue
                    

                if len(buf) > 1000:
                    step = max(1, len(buf) // 500)
                    sampled_buf = buf[::step]
                else:
                    sampled_buf = buf
                
                active_buffers[rid] = sampled_buf
                all_vals.extend(sampled_buf)
            

            if len(all_vals) >= 4:
                vals_array = np.array(all_vals, dtype=np.float32)
                global_min, global_max = float(np.min(vals_array)), float(np.max(vals_array))
                
                if np.isfinite(global_min) and np.isfinite(global_max) and global_max > global_min:
                    range_pad = 0.1 * (global_max - global_min)
                    self.plot_widget.setYRange(global_min - range_pad, global_max + range_pad, padding=0.0)
            

            for rid, sampled_buf in active_buffers.items():
                curve = self._plot_curves.get(int(rid))
                if curve is None:
                    continue
                
                y_data = np.asarray(sampled_buf, dtype=np.float32)
                x_data = np.arange(len(y_data), dtype=np.float32)
                

                curve.setData(x=x_data, y=y_data, skipFiniteCheck=True)
                

                alpha = 0.8 if len(self.buffers) <= 10 else 0.6
                pen = curve.opts['pen']
                if hasattr(pen, 'color'):
                    color = pen.color()
                    color.setAlphaF(alpha)
                    pen.setColor(color)
                    curve.setPen(pen)
            
        except Exception as e:
            print(f"❌ Direct overlay mode error: {e}")

    def _update_statistical_aggregation_mode(self):
       
        try:
            if not hasattr(self, '_stat_curves'):
                self._stat_curves = {}
                self._setup_statistical_plot()
            

            max_len = max(len(buf) for buf in self.buffers.values() if len(buf) > 0)
            if max_len == 0:
                return
                

            target_points = min(300, max_len)
            
            trace_matrix = []
            active_rois = []
            
            for rid, buf in self.buffers.items():
                if len(buf) < 2:
                    continue
                    

                if len(buf) > target_points:
                    indices = np.linspace(0, len(buf) - 1, target_points, dtype=int)
                    resampled = [buf[i] for i in indices]
                else:
                    resampled = list(buf)

                    while len(resampled) < target_points:
                        resampled.append(resampled[-1])
                
                trace_matrix.append(resampled)
                active_rois.append(rid)
            
            if not trace_matrix:
                return
                

            trace_array = np.array(trace_matrix, dtype=np.float32)
            x_data = np.arange(target_points, dtype=np.float32)
            

            mean_trace = np.mean(trace_array, axis=0)
            std_trace = np.std(trace_array, axis=0)
            percentile_25 = np.percentile(trace_array, 25, axis=0)
            percentile_75 = np.percentile(trace_array, 75, axis=0)
            percentile_10 = np.percentile(trace_array, 10, axis=0)
            percentile_90 = np.percentile(trace_array, 90, axis=0)
            

            if 'mean' in self._stat_curves:
                self._stat_curves['mean'].setData(x=x_data, y=mean_trace, skipFiniteCheck=True)
            
            if 'upper_std' in self._stat_curves and 'lower_std' in self._stat_curves:
                upper_std = mean_trace + std_trace
                lower_std = mean_trace - std_trace
                self._stat_curves['upper_std'].setData(x=x_data, y=upper_std, skipFiniteCheck=True)
                self._stat_curves['lower_std'].setData(x=x_data, y=lower_std, skipFiniteCheck=True)
            
            if 'p75' in self._stat_curves and 'p25' in self._stat_curves:
                self._stat_curves['p75'].setData(x=x_data, y=percentile_75, skipFiniteCheck=True)
                self._stat_curves['p25'].setData(x=x_data, y=percentile_25, skipFiniteCheck=True)
            

            if len(active_rois) >= 3:

                if not hasattr(self, '_roi_page_index'):
                    self._roi_page_index = 0
                    self._roi_page_size = 3  # Show 3 traces per page
                    self._roi_total_pages = max(1, len(active_rois))  # One page per ROI for full coverage
                    self._setup_pagination_controls()
                    print(f"📄 ROI Pagination initialized: {self._roi_total_pages} ROIs with manual controls")
                

                if self._roi_total_pages != len(active_rois):
                    self._roi_total_pages = len(active_rois)
                    self._roi_page_index = min(self._roi_page_index, self._roi_total_pages - 1)
                

                start_idx = self._roi_page_index
                selected_indices = []
                

                for i in range(3):
                    roi_idx = (start_idx + i) % len(active_rois)
                    selected_indices.append(roi_idx)
                

                for i in range(3):  
                    curve_key = f'highlight_{i}'
                    if curve_key in self._stat_curves:
                        if i < len(selected_indices):
                            idx = selected_indices[i]
                            if idx < len(trace_array):
                                roi_id = active_rois[idx]
                                self._stat_curves[curve_key].setData(x=x_data, y=trace_array[idx], skipFiniteCheck=True)

                                if hasattr(self._stat_curves[curve_key], 'opts') and 'name' in self._stat_curves[curve_key].opts:
                                    self._stat_curves[curve_key].opts['name'] = f'ROI {roi_id} ({idx+1}/{len(active_rois)})'
                        else:

                            self._stat_curves[curve_key].setData(x=[], y=[])
            

            all_stats = np.concatenate([mean_trace, percentile_10, percentile_90])
            if len(all_stats) > 0:
                stat_min, stat_max = float(np.min(all_stats)), float(np.max(all_stats))
                if np.isfinite(stat_min) and np.isfinite(stat_max) and stat_max > stat_min:
                    range_pad = 0.15 * (stat_max - stat_min)
                    self.plot_widget.setYRange(stat_min - range_pad, stat_max + range_pad, padding=0.0)
            
        except Exception as e:
            print(f"❌ Statistical aggregation mode error: {e}")

    def _setup_pagination_controls(self):
       
        try:
            from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel
            from PyQt5.QtCore import Qt, QTimer
            from PyQt5.QtGui import QColor
            import pyqtgraph as pg
            
            if hasattr(self, '_pagination_widget') and self._pagination_widget is not None:
                try:
                    if self._pagination_widget.isVisible():

                        self._update_page_label_safe()
                        return
                    else:

                        self._cleanup_pagination_widget()
                except Exception:

                    self._cleanup_pagination_widget()
                    

            if not hasattr(self, '_current_page'):
                self._current_page = 0
            if not hasattr(self, '_traces_per_page'):
                self._traces_per_page = 5
                

            if not hasattr(self, '_pagination_widget') or self._pagination_widget is None:

                self._pagination_widget = QWidget()
                main_layout = QVBoxLayout(self._pagination_widget)
                main_layout.setSpacing(5)
                

                nav_widget = QWidget()
                pagination_layout = QHBoxLayout(nav_widget)
                pagination_layout.setContentsMargins(0, 0, 0, 0)
                

                self._prev_button = QPushButton("◀ Prev Traces")
                self._prev_button.setMaximumWidth(120)
                self._prev_button.clicked.connect(self._prev_roi_page)
                pagination_layout.addWidget(self._prev_button)
                

                self._page_label = QLabel("Traces 1-5 (Page 1/1)")
                self._page_label.setAlignment(Qt.AlignCenter)
                self._page_label.setStyleSheet("font-weight: bold; padding: 5px; min-width: 150px;")
                pagination_layout.addWidget(self._page_label)
                

                self._next_button = QPushButton("Next Traces ▶")
                self._next_button.setMaximumWidth(120)
                self._next_button.clicked.connect(self._next_roi_page)
                pagination_layout.addWidget(self._next_button)
                

                self._expand_button = QPushButton("🔍 Expand All ROIs")
                self._expand_button.setMaximumWidth(140)
                self._expand_button.setStyleSheet("""
                    QPushButton {
                        background-color: #4CAF50;
                        color: white;
                        font-weight: bold;
                        border-radius: 5px;
                        padding: 6px;
                    }
                    QPushButton:hover {
                        background-color: #45a049;
                    }
                """)
                self._expand_button.clicked.connect(self._expand_all_rois)
                pagination_layout.addWidget(self._expand_button)
                
                main_layout.addWidget(nav_widget)
                

                self._legend_widget = QWidget()
                self._legend_layout = QHBoxLayout(self._legend_widget)
                self._legend_layout.setContentsMargins(5, 5, 5, 5)
                self._legend_layout.setSpacing(10)
                

                legend_title = QLabel("Current ROIs:")
                legend_title.setStyleSheet("font-weight: bold; font-size: 10px;")
                self._legend_layout.addWidget(legend_title)
                

                self._legend_labels = []
                
                main_layout.addWidget(self._legend_widget)
                

                self._pagination_widget.setStyleSheet("""
                    QWidget {
                        background-color: #f8f8f8;
                        border: 1px solid #ddd;
                        border-radius: 5px;
                        margin: 2px;
                    }
                    QPushButton {
                        background-color: #e8e8e8;
                        border: 1px solid #ccc;
                        border-radius: 3px;
                        padding: 5px;
                    }
                    QPushButton:hover {
                        background-color: #d8d8d8;
                    }
                """)
                
                try:

                    self._pagination_widget.setWindowTitle("ROI Pagination Controls")
                    self._pagination_widget.setWindowFlags(Qt.Tool | Qt.WindowStaysOnTopHint)
                    self._pagination_widget.resize(600, 100)
                    

                    if self.plot_widget:
                        try:
                            plot_geometry = self.plot_widget.geometry()
                            self._pagination_widget.move(plot_geometry.x(), plot_geometry.y() + plot_geometry.height() + 10)
                        except Exception:
                            pass  # Use default position
                    
                    try:
                        from PyQt5.QtCore import Qt
                        self._pagination_widget.setWindowModality(Qt.NonModal)
                        self._pagination_widget.setWindowFlags(
                            Qt.Tool | Qt.WindowStaysOnTopHint | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint
                        )

                        if self.plot_widget and hasattr(self.plot_widget, 'window') and self.plot_widget.window():
                            main_window = self.plot_widget.window()
                            try:

                                if not hasattr(self, '_pagination_close_connected'):
                                    main_window.destroyed.connect(self._cleanup_pagination_widget)
                                    self._pagination_close_connected = True
                            except Exception:
                                pass
                    except Exception:
                        pass
                    self._pagination_widget.show()
                    print("✅ ROI pagination controls created as standalone widget")
                    
                except Exception as pagination_error:
                    print(f"❌ Pagination creation failed: {pagination_error}")

                    if hasattr(self, '_pagination_widget'):
                        try:
                            self._pagination_widget.setParent(None)
                            self._pagination_widget.deleteLater()
                        except Exception:
                            pass
                        self._pagination_widget = None

        except Exception as e:
            print(f"⚠️ Could not create pagination controls: {e}")
            import traceback
            print(f"   Stack trace: {traceback.format_exc()}")

            try:
                if hasattr(self, '_pagination_widget') and self._pagination_widget is not None:
                    self._pagination_widget.close()
                    self._pagination_widget.deleteLater()
                    self._pagination_widget = None
            except Exception:
                pass

    def _update_page_label_safe(self):
       
        try:
            if (hasattr(self, '_pagination_widget') and 
                hasattr(self, '_page_label') and 
                hasattr(self, '_trace_page_index') and
                hasattr(self, '_traces_per_page')):
                
                active_rois = sorted([rid for rid, buf in self.buffers.items() if len(buf) >= 2])
                if active_rois:
                    total_pages = max(1, (len(active_rois) + self._traces_per_page - 1) // self._traces_per_page)
                    self._page_label.setText(f"Page {self._trace_page_index + 1} of {total_pages}")
                

                    start_idx = self._trace_page_index * self._traces_per_page
                    end_idx = min(start_idx + self._traces_per_page, len(active_rois))
                    page_rois = active_rois[start_idx:end_idx]
                    self._update_legend_for_page(page_rois)
        except Exception as e:
            pass

    def _prev_roi_page(self):
       
        try:

            if hasattr(self, '_navigation_in_progress') and self._navigation_in_progress:
                return
            self._navigation_in_progress = True
            
            active_rois = sorted([rid for rid, buf in self.buffers.items() if len(buf) >= 2])
            if not active_rois:
                self._navigation_in_progress = False
                return
            
            if not hasattr(self, '_trace_page_index'):
                self._trace_page_index = 0
                
            if self._trace_page_index > 0:
                self._trace_page_index -= 1
            else:

                total_pages = max(1, (len(active_rois) + self._traces_per_page - 1) // self._traces_per_page)
                self._trace_page_index = total_pages - 1
            self._update_paged_trace_mode()
            self._update_page_label_safe()
            print(f"📄 Trace page: {self._trace_page_index + 1}")
            
            self._navigation_in_progress = False
        except Exception as e:
            print(f"⚠️ Previous page error: {e}")
            self._navigation_in_progress = False
    
    def _next_roi_page(self):
       
        try:

            if hasattr(self, '_navigation_in_progress') and self._navigation_in_progress:
                return
            self._navigation_in_progress = True
            
            active_rois = sorted([rid for rid, buf in self.buffers.items() if len(buf) >= 2])
            if not active_rois:
                self._navigation_in_progress = False
                return
            

            if not hasattr(self, '_trace_page_index'):
                self._trace_page_index = 0
            if not hasattr(self, '_traces_per_page'):
                self._traces_per_page = 5
                
            total_pages = max(1, (len(active_rois) + self._traces_per_page - 1) // self._traces_per_page)
            
            if self._trace_page_index < total_pages - 1:
                self._trace_page_index += 1
            else:

                self._trace_page_index = 0
            self._update_paged_trace_mode()
            self._update_page_label_safe()
            print(f"📄 Trace page: {self._trace_page_index + 1}")
            
            self._navigation_in_progress = False
        except Exception as e:
            print(f"⚠️ Next page error: {e}")
            self._navigation_in_progress = False

    def restart_after_napari(self, new_plot_widget=None):
       
        try:
            print("🔄 Restarting LiveTraceExtractor after Napari...")
            

            if new_plot_widget:
                self.plot_widget = new_plot_widget
                print("✅ Plot widget updated")
            

            if self.plot_widget:

                if hasattr(self, '_pagination_widget'):
                    self._cleanup_pagination_widget()
                

                self._setup_pagination_controls()
                print("✅ Pagination controls reinitialized")
            

            if hasattr(self, 'buffers') and self.buffers:
                self._update_paged_trace_mode()
                print("✅ Live traces resumed")
            
            return True
            
        except Exception as e:
            print(f"❌ Restart after Napari failed: {e}")
            return False

    def _cleanup_pagination_widget(self):
       
        try:
            if hasattr(self, '_pagination_widget') and self._pagination_widget is not None:
                try:
                    self._pagination_widget.close()
                except Exception:
                    pass
                self._pagination_widget.setParent(None)
                self._pagination_widget.deleteLater()
                self._pagination_widget = None
                

            if hasattr(self, '_legend_labels'):
                for label in self._legend_labels:
                    if label:
                        label.setParent(None)
                        label.deleteLater()
                self._legend_labels.clear()
                
        except Exception as e:
            print(f"⚠️ Pagination cleanup warning: {e}")

    def _update_page_label_safe(self):
       
        try:
            if not hasattr(self, '_page_label') or not self._page_label:
                return
                
            active_rois = sorted([rid for rid, buf in self.buffers.items() if len(buf) >= 2])
            if not active_rois:
                self._page_label.setText("No active traces")
                if hasattr(self, '_prev_button'):
                    self._prev_button.setEnabled(False)
                if hasattr(self, '_next_button'):
                    self._next_button.setEnabled(False)
                return
                
            total_pages = max(1, (len(active_rois) + self._traces_per_page - 1) // self._traces_per_page)
            current_page = getattr(self, '_trace_page_index', 0) + 1
            
            start_roi = (getattr(self, '_trace_page_index', 0) * self._traces_per_page) + 1
            end_roi = min(start_roi + self._traces_per_page - 1, len(active_rois))
            
            self._page_label.setText(f"Traces {start_roi}-{end_roi} (Page {current_page}/{total_pages})")
            

            if hasattr(self, '_prev_button'):
                self._prev_button.setEnabled(True)  
            if hasattr(self, '_next_button'):
                self._next_button.setEnabled(True)  
                
        except Exception as e:
            print(f"⚠️ Page label update error: {e}")

    def _update_page_label(self):
       
        try:
            if hasattr(self, '_page_label') and hasattr(self, '_trace_page_index'):

                active_rois = [rid for rid, buf in self.buffers.items() if len(buf) >= 2]
                total_pages = max(1, (len(active_rois) + self._traces_per_page - 1) // self._traces_per_page)
                

                start_idx = self._trace_page_index * self._traces_per_page
                end_idx = min(start_idx + self._traces_per_page, len(active_rois))
                
                self._page_label.setText(f"Traces {start_idx + 1}-{end_idx} (Page {self._trace_page_index + 1}/{total_pages})")
        except Exception as e:
            print(f"⚠️ Page label update error: {e}")

    def _setup_statistical_plot(self):
       
        try:
            self._stat_curves = {}
            

            if hasattr(self, '_plot_curves'):
                for curve in self._plot_curves.values():
                    self.plot_widget.removeItem(curve)
                self._plot_curves.clear()
            

            mean_pen = pg.mkPen(color='#3498db', width=3, style=pg.QtCore.Qt.SolidLine)
            self._stat_curves['mean'] = self.plot_widget.plot(pen=mean_pen, name='Mean')
            

            std_pen = pg.mkPen(color='#85c1e8', width=2, style=pg.QtCore.Qt.DashLine)
            self._stat_curves['upper_std'] = self.plot_widget.plot(pen=std_pen, name='Mean + 1σ')
            self._stat_curves['lower_std'] = self.plot_widget.plot(pen=std_pen, name='Mean - 1σ')
            

            perc_pen = pg.mkPen(color='#2ecc71', width=2, style=pg.QtCore.Qt.DotLine)
            self._stat_curves['p75'] = self.plot_widget.plot(pen=perc_pen, name='75th percentile')
            self._stat_curves['p25'] = self.plot_widget.plot(pen=perc_pen, name='25th percentile')
            

            highlight_colors = ['#e74c3c', '#f39c12', '#9b59b6']
            for i in range(3):
                highlight_pen = pg.mkPen(color=highlight_colors[i], width=1, alpha=0.7)
                self._stat_curves[f'highlight_{i}'] = self.plot_widget.plot(pen=highlight_pen)
            
            print("✅ Statistical aggregation plot setup complete")
            
        except Exception as e:
            print(f"❌ Statistical plot setup error: {e}")

    def _update_density_heatmap_mode(self):
       
        try:
            if not hasattr(self, '_density_plot'):
                self._setup_density_plot()
            

            max_len = max(len(buf) for buf in self.buffers.values() if len(buf) > 0)
            if max_len == 0:
                return
            

            target_points = min(200, max_len)
            roi_count = len([buf for buf in self.buffers.values() if len(buf) > 0])
            

            density_matrix = np.zeros((roi_count, target_points), dtype=np.float32)
            
            for i, (rid, buf) in enumerate(self.buffers.items()):
                if len(buf) < 2 or i >= roi_count:
                    continue
                    

                if len(buf) > target_points:
                    indices = np.linspace(0, len(buf) - 1, target_points, dtype=int)
                    resampled = np.array([buf[idx] for idx in indices], dtype=np.float32)
                else:
                    resampled = np.array(list(buf), dtype=np.float32)

                    if len(resampled) < target_points:
                        padding = np.full(target_points - len(resampled), resampled[-1])
                        resampled = np.concatenate([resampled, padding])
                
                density_matrix[i, :] = resampled
            

            if hasattr(self, '_density_image'):
                self._density_image.setImage(density_matrix, autoLevels=True, autoDownsample=True)
            

            if hasattr(self, '_summary_curves'):

                overall_mean = np.mean(density_matrix, axis=0)
                overall_std = np.std(density_matrix, axis=0)
                
                x_data = np.arange(target_points, dtype=np.float32)
                
                self._summary_curves['mean'].setData(x=x_data, y=overall_mean, skipFiniteCheck=True)
                self._summary_curves['upper'].setData(x=x_data, y=overall_mean + overall_std, skipFiniteCheck=True)
                self._summary_curves['lower'].setData(x=x_data, y=overall_mean - overall_std, skipFiniteCheck=True)
            
        except Exception as e:
            print(f"❌ Density heatmap mode error: {e}")

    def _setup_density_plot(self):
       
        try:

            self.plot_widget.clear()
            

            self._density_image = pg.ImageItem()
            self.plot_widget.addItem(self._density_image)
            
            self._summary_curves = {}
            
            mean_pen = pg.mkPen(color='white', width=2)
            self._summary_curves['mean'] = self.plot_widget.plot(pen=mean_pen, name='Population Mean')
            
            bound_pen = pg.mkPen(color='yellow', width=1, alpha=0.7)
            self._summary_curves['upper'] = self.plot_widget.plot(pen=bound_pen, name='Mean + 1σ')
            self._summary_curves['lower'] = self.plot_widget.plot(pen=bound_pen, name='Mean - 1σ')
            
            print("✅ Density heatmap plot setup complete")
            
        except Exception as e:
            print(f"❌ Density plot setup error: {e}")


    def _build_rois_for_shape(self, H: int, W: int):
       
        try:
            print(f"🔄 Building ROIs for frame shape {W}x{H}...")
            
            self._cleanup_existing_rois()
            

            if (self._labels_orig.shape[0], self._labels_orig.shape[1]) != (H, W):
                resized = cv2.resize(self._labels_orig, (W, H), interpolation=cv2.INTER_NEAREST)
                print(f"📐 Resized labels from {self._labels_orig.shape} to {resized.shape}")
            else:
                resized = self._labels_orig

            ids = np.unique(resized)
            ids = ids[ids > 0]
            if ids.size == 0:
                print("⚠️ No positive ROI labels found after resize; running in empty-safe mode")
                self._initialize_empty_state()

                return

            self.ids = ids[: self._max_rois_cfg].astype(np.int32)
            self._H, self._W = H, W
            

            self._initialize_buffers_safely()
            

            self._initialize_processing_structures(resized)
            
            self._roi_ready = True
            print(f"✅ ROIs ready for frame shape {W}x{H} with {len(self.ids)} labels")
            
        except Exception as e:
            print(f"❌ Error building ROIs: {e}")
            import traceback
            print(f"   Stack trace: {traceback.format_exc()}")
            self._initialize_empty_state()
    
    def _cleanup_existing_rois(self):
       
        try:

            if hasattr(self, 'buffers'):
                self.buffers.clear()
            

            if CUDA_AVAILABLE:
                if hasattr(self, '_labels_gpu') and self._labels_gpu is not None:
                    del self._labels_gpu
                if hasattr(self, '_ids_gpu') and self._ids_gpu is not None:
                    del self._ids_gpu
                if hasattr(self, '_roi_sizes_gpu') and self._roi_sizes_gpu is not None:
                    del self._roi_sizes_gpu
                if hasattr(self, '_f_gpu') and self._f_gpu is not None:
                    del self._f_gpu
            

            self._flat_labels_cpu = None
            self._roi_sizes_cpu = None
            

            if hasattr(self, '_plot_curves'):
                self._plot_curves.clear()
                
            print("🧹 Existing ROI structures cleaned up")
            
        except Exception as e:
            print(f"⚠️ Error during ROI cleanup: {e}")
    
    def _initialize_empty_state(self):
       
        self.ids = np.array([], dtype=np.int32)
        self.buffers = {}
        self._roi_ready = False
        self._labels_gpu = None
        self._ids_gpu = None
        self._roi_sizes_gpu = None
        self._f_gpu = None
        self._flat_labels_cpu = None
        self._roi_sizes_cpu = None
    
    def _initialize_buffers_safely(self):
       
        from collections import deque
        
        self.buffers = {}
        for r in self.ids:
            rid_key = int(r)
            self.buffers[rid_key] = deque(maxlen=self._max_points_cfg)
        

        print(f"📊 Initialized buffers for ROI IDs: {sorted(self.buffers.keys())}")
        if len(self.buffers) != len(self.ids):
            print(f"⚠️ Buffer count mismatch: {len(self.buffers)} buffers vs {len(self.ids)} ROIs")

            for r in self.ids:
                rid_key = int(r)
                if rid_key not in self.buffers:
                    self.buffers[rid_key] = deque(maxlen=self._max_points_cfg)
                    print(f"   🔧 Added missing buffer for ROI {rid_key}")
        
        print(f"✅ Buffer verification complete: {len(self.buffers)} buffers for {len(self.ids)} ROIs")
    
    def _initialize_processing_structures(self, resized):
       
        flat = resized.ravel().astype(np.int32)
        self._flat_labels_cpu = flat
        self._max_label = int(flat.max(initial=0))

        if CUDA_AVAILABLE:
            try:
                self._labels_gpu = cp.asarray(flat)
                self._ids_gpu = cp.asarray(self.ids)
                counts = cp.bincount(self._labels_gpu, minlength=self._max_label + 1)
                self._roi_sizes_gpu = counts[self._ids_gpu].astype(cp.float32)
                self._f_gpu = cp.empty(len(flat), dtype=cp.float32)
                self._roi_sizes_cpu = None
                print(f"✅ GPU processing structures initialized for {len(self.ids)} ROIs")
            except Exception as e:
                print(f"⚠️ GPU initialization failed, falling back to CPU: {e}")
                self._initialize_cpu_fallback(flat)
        else:
            self._initialize_cpu_fallback(flat)


        if self.plot_widget is not None and PYQTPGRAPH_AVAILABLE:
            for rid in self.ids:
                if rid not in self._plot_curves:
                    pen = pg.mkPen(pg.intColor(len(self._plot_curves), hues=max(8, len(self.ids))), width=1)
                    self._plot_curves[int(rid)] = self.plot_widget.plot(pen=pen)
    
    def _initialize_cpu_fallback(self, flat):
       
        try:
            counts = np.bincount(flat, minlength=self._max_label + 1)
            self._roi_sizes_cpu = counts[self.ids].astype(np.float32)
            self._labels_gpu = None
            self._ids_gpu = None
            self._roi_sizes_gpu = None
            self._f_gpu = None
            print(f"✅ CPU processing structures initialized for {len(self.ids)} ROIs")
        except Exception as e:
            print(f"❌ CPU initialization also failed: {e}")
            self._initialize_empty_state()


    def get_performance_stats(self) -> Dict[str, Any]:
        try:
            mem_mb = psutil.Process().memory_info().rss / 1024 / 1024
        except Exception:
            mem_mb = 0.0
        uptime = time.time() - self.start_time
        fps = self.stats["frames_processed"] / uptime if uptime > 0 else 0.0
        out = {
            "frames_processed": self.stats["frames_processed"],
            "frames_failed": self.stats["frames_failed"],
            "memory_usage_peak": self.stats["memory_usage_peak"],
            "current_memory_mb": mem_mb,
            "uptime_seconds": uptime,
            "frames_per_second": fps,
            "gpu_memory_peak": self.stats["gpu_memory_peak"],
            "sync_operations": self.stats["sync_operations"],
            "sync_failures": self.stats["sync_failures"],
            "sync_state": self._sync_state.value,
        }
        return out

    def export_traces(self, base_name="live_traces", last_n=100):
        try:
            self.export_counter += 1
            output_path = f"{base_name}_{self.export_counter}.npy"
            roiprint_out = f"roiprint_export_{self.export_counter}.npz"


            traces = {}
            for rid, buf in self.buffers.items():
                if buf:
                    traces[f"roi_{int(rid)}"] = list(buf)[-last_n:]
            np.save(output_path, traces)


            sizes = (self._roi_sizes_gpu.get() if (CUDA_AVAILABLE and self._roi_sizes_gpu is not None)
                     else np.asarray(self._roi_sizes_cpu))
            np.savez_compressed(roiprint_out,
                                ids=np.asarray(self.ids, dtype=np.int32),
                                roi_sizes=np.asarray(sizes, dtype=np.float32),
                                shape=(self._H, self._W))

            print(f"Traces saved → {output_path}, ROI info → {roiprint_out}")

        except Exception as e:
            print(f"Trace export error: {e}")
            self.error_occurred.emit(str(e))

    def _update_sync_state(self, state: SyncState, err: Optional[str] = None):
        with self._sync_lock:
            self._sync_state = state
            self._syncprint = SyncInfo(
                state=state,
                timestamp=time.time(),
                frame_count=self.stats["frames_processed"],
                memory_usage=self.stats["memory_usage_peak"],
                gpu_memory_usage=self.stats["gpu_memory_peak"],
                error_message=err,
            )
            self.sync_state_changed.emit(self._syncprint)


    def cleanup(self):
       
        try:
            print("🧹 Starting LiveTraceExtractor cleanup...")
            self._is_shutting_down = True
            self._update_sync_state(SyncState.STOPPING)
            
            if hasattr(self, "_cleanup_event"):
                self._cleanup_event.set()
                print("✅ Cleanup event set - signaling all threads to stop")
            
            if hasattr(self, '_pagination_widget'):
                try:
                    self._cleanup_pagination_widget()
                    print("✅ Pagination controls cleaned up")
                except Exception as e:
                    print(f"⚠️ Pagination cleanup warning: {e}")
                    
            if hasattr(self, '_expanded_dialog'):
                try:
                    if self._expanded_dialog and self._expanded_dialog.isVisible():
                        self._expanded_dialog.close()
                    self._expanded_dialog = None
                    self._expanded_curves = {}
                    print("✅ Expanded view cleaned up")
                except Exception as e:
                    print(f"⚠️ Expanded view cleanup warning: {e}")

            try:
                self._disconnect_camera_signals()
                print("✅ Camera signals disconnected")
            except Exception as e:
                print(f"⚠️ Error disconnecting camera signals: {e}")

            if hasattr(self, "frame_processor") and self.frame_processor is not None:
                try:
                    if self.frame_processor.isRunning():
                        self.frame_processor.stop()
                        if not self.frame_processor.wait(2000):  
                            print("⚠️ Frame processor did not stop gracefully, forcing termination")
                            self.frame_processor.terminate()
                    self.frame_processor.wait(1000)
                    print("✅ Frame processor stopped")
                except Exception as e:
                    print(f"⚠️ Error stopping frame processor: {e}")

            if getattr(self, "_plot_timer", None):
                try:
                    self._plot_timer.stop()
                    self._plot_timer.deleteLater()
                    self._plot_timer = None
                    print("✅ Plot timer stopped")
                except Exception as e:
                    print(f"⚠️ Error stopping plot timer: {e}")

            if hasattr(self, '_monitor_threads'):
                try:
                    print(f"⏳ Waiting for {len(self._monitor_threads)} monitor threads to stop...")
                    for thread in self._monitor_threads:
                        if thread.is_alive():
                            thread.join(timeout=3.0) 
                            if thread.is_alive():
                                print(f"⚠️ Monitor thread {thread.name} did not stop gracefully")
                            else:
                                print(f"✅ Monitor thread {thread.name} stopped")
                    self._monitor_threads.clear()
                except Exception as e:
                    print(f"⚠️ Error waiting for monitor threads: {e}")

            try:
                if hasattr(self, '_plot_curves'):
                    self._plot_curves.clear()
                if hasattr(self, '_stat_curves'):
                    self._stat_curves.clear()
                if hasattr(self, '_pagination_widget'):
                    try:
                        self._pagination_widget.close()
                        self._pagination_widget.deleteLater()
                        self._pagination_widget = None
                    except Exception:
                        pass
                print("✅ Plot resources cleared")
            except Exception as e:
                print(f"⚠️ Error clearing plot resources: {e}")

            if CUDA_AVAILABLE:
                try:
                    gpu_resources = ['_f_gpu', '_labels_gpu', '_ids_gpu', '_roi_sizes_gpu']
                    for resource in gpu_resources:
                        if hasattr(self, resource) and getattr(self, resource) is not None:
                            try:
                                delattr(self, resource)
                            except Exception:
                                setattr(self, resource, None)

                    cp.get_default_memory_pool().free_all_blocks()
                    print("✅ GPU resources cleaned")
                except Exception as e:
                    print(f"⚠️ GPU cleanup error: {e}")

            if self.use_pygame_plot:
                try:
                    pygame.display.quit()
                    pygame.quit()
                    print("✅ Pygame cleaned up")
                except Exception as e:
                    print(f"⚠️ Pygame cleanup error: {e}")

            try:
                self.buffers.clear()
                self._cpu_masks = None
                self._flat_labels_cpu = None
                self._roi_sizes_cpu = None
                print("✅ Data structures cleared")
            except Exception as e:
                print(f"⚠️ Error clearing data structures: {e}")

            try:
                collected = gc.collect()
                if collected > 0:
                    print(f"✅ Garbage collection freed {collected} objects")
            except Exception as e:
                print(f"⚠️ Garbage collection error: {e}")

            print("✅ LiveTraceExtractor cleanup completed successfully")

        except Exception as e:
            print(f"❌ Critical cleanup error: {e}")
            import traceback
            print(f"   Stack trace: {traceback.format_exc()}")
            try:
                if hasattr(self, 'buffers'):
                    self.buffers.clear()
                gc.collect()
            except Exception:
                pass
            self._update_sync_state(SyncState.IDLE)

        uptime = time.time() - self.start_time
        print("✅ LiveTraceExtractor cleanup complete")
        print(f"📊 Runtime: {uptime:.1f}s, frames: {self.stats['frames_processed']}, "
              f"peak RSS: {self.stats['memory_usage_peak']:.1f} MB")

    def stop(self):
        self.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
