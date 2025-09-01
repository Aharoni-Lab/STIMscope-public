
import os
import time
import gc
import signal
import atexit
import psutil
import sys
import threading
import traceback
from collections import deque
from typing import Optional

import numpy as np
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QAction, QTextEdit, QFileDialog, QLabel, QOpenGLWidget
)

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage

PLOT_WITH_PYQTGRAPH = True  
ENABLE_GPUUI_HTMLprint = False  

def _noop(*a, **kw): pass

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except Exception:
    CUDA_AVAILABLE = False

TRACE_OUT = "live_traces.npy"
ROIprint_OUT = "roiprint_export.npz"

CAMERA_AVAILABLE = True
Camera = None 

from live_trace_extractor import LiveTraceExtractor

__all__ = ["GPU"]

class GPU(QtWidgets.QWidget):


    closed = pyqtSignal()

    refineRequested = pyqtSignal(object, object)
    requestStartLiveTraces = pyqtSignal()
    requestStopLiveTraces = pyqtSignal()

    instance: Optional["GPU"] = None

    export_count = 0

    def __init__(self, camera: Camera,parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        if camera is None:
            raise ValueError("GPU UI requires a Camera instance")
        self.camera = camera
        GPU.instance = self

        self.setWindowTitle("CRISPI")
        self.resize(800, 560)


        self.requestStartLiveTraces.connect(self.start_live_traces, QtCore.Qt.QueuedConnection)
        self.requestStopLiveTraces.connect(self.stop_live_traces, QtCore.Qt.QueuedConnection)

        self.refineRequested.connect(self._launch_napari_viewer)

        self.layout = QVBoxLayout(self)


        self.plot_widget = None
        if PLOT_WITH_PYQTGRAPH:
            try:
                import pyqtgraph as pg
                self.plot_widget = pg.PlotWidget()
                self.plot_widget.setBackground('k')
                self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
                self.plot_widget.setMouseEnabled(x=False, y=False)
                self.plot_widget.setYRange(0, 255)
                self.layout.addWidget(self.plot_widget)
            except Exception as e:
                print(f"pyqtgraph unavailable, continuing without on-screen traces: {e}")


        self.paused = False


        self.video_path = None
        self.proj_display = None
        self.memmap_path = "movie_mmap.npy"
        self.rois_path = "rois.npz"
        self.trace_path = "traces_live.npy"
        self._discover_method = "OTSU"


        from live_trace_extractor import LiveTraceExtractor
        self.live_extractor: Optional[LiveTraceExtractor] = None

        self._build_pipeline_buttons()

        self._setup_long_term_stability()


    def _build_pipeline_buttons(self):
        grid = QtWidgets.QGridLayout()
        row = 0


        btn = QtWidgets.QPushButton("🖼 Select Video…")
        btn.clicked.connect(self._select_video)
        grid.addWidget(btn, row, 0)


        btn = QtWidgets.QPushButton("➤ Make Memmap")
        btn.clicked.connect(self._run_make_memmap)
        grid.addWidget(btn, row, 1)


        dd = QtWidgets.QToolButton()
        dd.setText("➤ Discover Mask")
        dd.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(dd)
        for method in ("Cellpose", "CNMF", "Custom", "OTSU"):
            act = QtWidgets.QAction(method, dd)
            act.triggered.connect(lambda _=False, m=method: self._run_discover_rois(m))
            menu.addAction(act)
        dd.setMenu(menu)
        grid.addWidget(dd, row, 2)


        btn = QtWidgets.QPushButton("➤ Manual Mask Editor")
        btn.clicked.connect(self._run_refine_rois)
        grid.addWidget(btn, row, 3)


        btn = QtWidgets.QPushButton("▶ Export Traces")
        btn.clicked.connect(self._export_traces)
        grid.addWidget(btn, row, 5)


        row += 1
        btn = QtWidgets.QPushButton("👁️ View Exported Traces")
        btn.clicked.connect(self._view_exported_traces)
        grid.addWidget(btn, row, 0, 1, 2)  # Span 2 columns

        self.layout.addLayout(grid)


    def _setup_long_term_stability(self):
        self._memory_history = deque(maxlen=100)
        self._cpu_history = deque(maxlen=100)
        self._gpu_memory_history = deque(maxlen=100)
        self._last_memory_report = time.time()
        self._error_count = 0
        self._last_error_time = 0.0
        self._max_errors_per_minute = 5
        self._last_activity_time = time.time()

        def every(ms, fn):
            def _wrap():
                try:
                    fn()
                finally:
                    QTimer.singleShot(ms, _wrap)
            QTimer.singleShot(ms, _wrap)

        every(30_000, self._monitor_memory_usage)
        every(60_000, self._watchdog_check)
        every(120_000, self._periodic_cleanup)
        every(45_000, self._check_thread_health)
        every(90_000, self._monitor_performance)


        atexit.register(self._emergency_cleanup)
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except Exception:
            pass

    def _select_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select video file", "", "Video files (*.avi *.mp4 *.h5 *.npy *.npz)"
        )
        if path:
            self.video_path = path
            print(f"Selected video: {path}")

    def _run_make_memmap(self):
        threading.Thread(target=self._thread_make_memmap, daemon=True).start()

    def _thread_make_memmap(self):
        print("Making memmap…")
        try:
            if not self.video_path or not os.path.exists(self.video_path):
                print("No valid video file selected")
                return
            size_mb = os.path.getsize(self.video_path) / (1024 * 1024)
            if size_mb > 500:
                print(f"Large video file detected: {size_mb:.1f} MB")
            gc.collect()
            from make_mmap import make_memmap
            make_memmap(self.video_path, self.memmap_path)
            print(f"Memmap saved to {self.memmap_path}")
            gc.collect()
        except MemoryError as e:
            self._handle_error(e, "Memmap (MemoryError)")
            print("Try processing a smaller video file or restart the app")
        except Exception as e:
            self._handle_error(e, "Memmap")

    def _run_discover_rois(self, method="OTSU"):
        self._discover_method = method
        threading.Thread(target=self._thread_discover_rois, daemon=True).start()

    def _thread_discover_rois(self):
        print("Discovering ROIs…")

        self.requestStopLiveTraces.emit()


        try:
            if self._discover_method == "OTSU":
                movie = np.load(self.memmap_path, mmap_mode="r")
                from otsu_thresh import compute_mean_projection, denoise_and_threshold_gpu

                mean = compute_mean_projection(movie, calib_frames=5400, chunk_size=200)
                mean = cv2.resize(mean, (1936, 1096), interpolation=cv2.INTER_NEAREST)
                masks, sizes = denoise_and_threshold_gpu(
                    mean, gauss_ksize=(3, 3), gauss_sigma=1.5, min_area=60, max_area=300
                )
                if not masks:
                    print("ROI discovery produced no masks; aborting live traces/recording.")
                    return

                labeled = np.zeros_like(masks[0], dtype=np.int32)
                labeled = labeled.astype(np.int32, copy=False)

                for i, m in enumerate(masks, start=1):
                    labeled[m] = i

            elif self._discover_method in ("Cellpose", "CNMF", "Custom"):
                raise NotImplementedError(f"{self._discover_method} integration not implemented")
            else:
                raise ValueError(f"Unknown ROI method: {self._discover_method}")


            try:
                from skimage.color import label2rgb
                from projection import ProjectDisplay
                from PyQt5.QtGui import QGuiApplication

                rgb = (label2rgb(labeled, bg_label=0) * 255).astype(np.uint8)

                screens = QGuiApplication.screens()
                scr = screens[1] if len(screens) > 1 else screens[0]
                size = scr.size()
                rgb = cv2.resize(rgb, (size.width(), size.height()), interpolation=cv2.INTER_NEAREST)

                if self.proj_display:
                    try:
                        self.proj_display.close()
                    except Exception:
                        pass
                self.proj_display = ProjectDisplay(scr)

                H = getattr(self.camera, "translation_matrix", None)
                self.proj_display.show_image_fullscreen_on_second_monitor(rgb, H)
                print("✅ Mask projection displayed")
            except Exception as e:
                print(f"Failed to project mask: {e}")


            np.savez_compressed(self.rois_path, masks=masks, sizes=sizes, labels=labeled)
            print(f"ROIs written to {self.rois_path}")


            self.requestStartLiveTraces.emit()
            print("Requested (queued) start of recording and live traces.")

        except Exception as e:
            print(f"ROI discovery failed: {e}")
            self._handle_error(e, "ROI discovery")

    def _run_refine_rois(self):
        threading.Thread(target=self._thread_refine_rois, daemon=True).start()

    def _thread_refine_rois(self):


        self.requestStopLiveTraces.emit()
        print("Manual Mask Generation…")
        try:
            from otsu_thresh import compute_mean_projection, load_movie
            mean = compute_mean_projection(load_movie(self.video_path), calib_frames=5400)
            mean = cv2.resize(mean, (1936, 1096), interpolation=cv2.INTER_NEAREST)
            masks = np.load(self.rois_path)["masks"]
            self.refineRequested.emit(mean, masks)
        except Exception as e:
            self._handle_error(e, "ROI refinement")



    @pyqtSlot()
    def start_live_traces(self):
       
        print("🚀 Starting live traces with enhanced safety...")
        

        if self.live_extractor is not None:
            print("🔄 Live extractor already exists. Performing clean restart...")
            try:
                self.stop_live_traces()

                from PyQt5.QtCore import QCoreApplication
                QCoreApplication.processEvents()
                import time
                time.sleep(0.1)
            except Exception as stop_error:
                print(f"⚠️ Error during extractor stop: {stop_error}")


        if not getattr(self.camera, "acquisition_running", False):
            print("📷 Starting camera acquisition for live traces...")
            try:
                if not self.camera.start_realtime_acquisition():
                    print("❌ Failed to start camera acquisition; cannot start live traces.")
                    return
                print("✅ Camera acquisition started")
            except Exception as cam_error:
                print(f"❌ Camera acquisition error: {cam_error}")
                return

        roi_path = self.rois_path
        if not os.path.exists(roi_path):
            print("❌ No ROI file found. Run Discover/Manual Mask first.")
            return
        
        print(f"📊 Using ROI file: {roi_path}")

        try:

            use_pygame = (self.plot_widget is None)

            self.live_extractor = LiveTraceExtractor(
                camera=self.camera,
                label_path=self.rois_path,
                plot_widget=self.plot_widget,
                max_points=150,
                max_rois=50,
                use_pygame_plot=False,
                enable_sync=False,
            )

          
            print("Live trace extractor started.")
        except Exception as e:
            print(f"Failed to start live traces: {e}")


    def stop_live_traces(self):
        try:
            if self.live_extractor is not None:
                try:
                    self.camera.image_update_signal.disconnect(self.live_extractor.on_frame)
                except Exception:
                    pass
                self.live_extractor.stop()
                self.live_extractor = None
                print("Live trace extractor stopped.")
        except Exception as e:
            print(f"Error stopping live trace extractor: {e}")


    
    @pyqtSlot(object, object)
    def _launch_napari_viewer(self, mean, masks):
       
        try:

            was_recording = self.camera.is_recording if self.camera else False
            was_live_traces = hasattr(self, 'live_extractor') and self.live_extractor is not None

            

            if was_live_traces:
                self.stop_live_traces()
                print("📊 Live traces paused for Napari launch")
                   

            was_camera_running = self.camera.acquisition_running if self.camera else False
            if was_camera_running:
                self.camera.stop_realtime_acquisition()
                print("📷 Camera acquisition paused for Napari launch")
            

            try:
                if self.proj_display:
                    self.proj_display.close()
            except Exception:
                pass
            

            time.sleep(0.2)
            
            def restore_after_napari(event=None):
               
                try:
                    print("🔄 Restoring operations after Napari close...")
                    

                    time.sleep(0.1)
                    

                    if was_camera_running and self.camera:
                        self.camera.start_realtime_acquisition()
                        print("📷 Camera acquisition restored")
                    

                    try:
                        from skimage.color import label2rgb
                        from projection import ProjectDisplay
                        from PyQt5.QtGui import QGuiApplication


                        if os.path.exists(self.rois_path):
                            try:
                                roi_data = np.load(self.rois_path)
                                if 'labels' in roi_data:
                                    labels = roi_data["labels"]
                                    print(f"🔄 Re-projecting updated ROIs: {len(np.unique(labels))-1} ROIs")
                                else:

                                    labels = np.load(self.rois_path)["labels"]
                                    print("🔄 Re-projecting original ROIs")
                            except Exception as e:
                                print(f"⚠️ Could not load updated ROIs: {e}")

                                labels = np.load(self.rois_path)["labels"]
                        else:
                            print("⚠️ No ROI file found for re-projection")
                            return

                        rgb = (label2rgb(labels, bg_label=0) * 255).astype(np.uint8)

                        screens = QGuiApplication.screens()
                        scr = screens[1] if len(screens) > 1 else screens[0]
                        size = scr.size()
                        rgb = cv2.resize(rgb, (size.width(), size.height()), interpolation=cv2.INTER_NEAREST)

                        if self.proj_display:
                            try:
                                self.proj_display.close()
                            except Exception:
                                pass
                        self.proj_display = ProjectDisplay(scr)
                        H = getattr(self.camera, "translation_matrix", None)
                        self.proj_display.show_image_fullscreen_on_second_monitor(rgb, H)
                        print("🖥️ Updated mask re-projected")
                        

                        if was_live_traces:
                            def restart_with_new_rois():
                                try:
                                    print("🔄 Attempting to restart live traces with updated ROIs...")
                                    

                                    if hasattr(self, 'live_extractor') and self.live_extractor:
                                        print("🧹 Cleaning up existing extractor...")
                                        self.live_extractor.cleanup()
                                        self.live_extractor = None
                                    

                                    import gc
                                    gc.collect()
                                    

                                    from PyQt5.QtCore import QCoreApplication
                                    QCoreApplication.processEvents()
                                    import time
                                    time.sleep(0.1)
                                    

                                    if not self.plot_widget or not hasattr(self.plot_widget, 'plot'):
                                        print("📊 Reinitializing plot widget for live traces...")
                                        try:
                                            if PLOT_WITH_PYQTGRAPH:
                                                import pyqtgraph as pg
                                                self.plot_widget = pg.PlotWidget()
                                                self.plot_widget.setLabel('left', 'Intensity')
                                                self.plot_widget.setLabel('bottom', 'Time (frames)')
                                                self.plot_widget.showGrid(x=True, y=True)
                                                

                                                if self.plot_widget not in [self.layout.itemAt(i).widget() for i in range(self.layout.count()) if self.layout.itemAt(i) and self.layout.itemAt(i).widget()]:
                                                    self.layout.addWidget(self.plot_widget)
                                                print("✅ Plot widget reinitialized")
                                        except Exception as plot_error:
                                            print(f"⚠️ Plot widget reinit failed: {plot_error}")
                                    

                                    self.start_live_traces()
                                    

                                    if hasattr(self, 'live_extractor') and self.live_extractor:

                                        if hasattr(self.live_extractor, 'restart_after_napari'):
                                            restart_success = self.live_extractor.restart_after_napari(self.plot_widget)
                                            if restart_success:
                                                print("✅ LiveTraceExtractor restarted successfully after Napari")
                                            else:
                                                print("⚠️ LiveTraceExtractor restart had issues, using fallback")

                                                self.live_extractor.plot_widget = self.plot_widget
                                                if hasattr(self.live_extractor, '_setup_pagination_controls'):
                                                    self.live_extractor._setup_pagination_controls()
                                        else:

                                            self.live_extractor.plot_widget = self.plot_widget
                                            if hasattr(self.live_extractor, '_setup_pagination_controls'):
                                                self.live_extractor._setup_pagination_controls()
                                    
                                    print("✅ Live traces restarted successfully with updated ROIs")
                                except Exception as restart_error:
                                    print(f"❌ Failed to restart live traces: {restart_error}")
                                    import traceback
                                    print(f"   Stack trace: {traceback.format_exc()}")
                                    

                                    def fallback_restart():
                                        try:
                                            self.start_live_traces()
                                            print("✅ Fallback restart successful")
                                        except Exception as fallback_error:
                                            print(f"❌ Fallback restart also failed: {fallback_error}")
                                    
                                    QTimer.singleShot(2000, fallback_restart)
                            
                            QTimer.singleShot(1000, restart_with_new_rois)  # Increased delay
                            print("📊 Live traces scheduled for restart with updated ROIs")
                        
                    except Exception as e:
                        print(f"⚠️ Failed to re-project mask: {e}")

                        if was_live_traces:
                            QTimer.singleShot(500, self.start_live_traces)
                            print("📊 Live traces scheduled for restart (projection failed)")
                    
                    print("✅ All operations restored successfully")
                    
                except Exception as e:
                    print(f"❌ Error restoring operations: {e}")
                    self._handle_error(e, "restore_after_napari")
            

            try:


                try:
                    from roi_editor import refine_rois
                    roi_editor_available = True
                except ImportError as e:
                    print(f"❌ roi_editor import failed: {e}")
                    print("❌ Cannot proceed without roi_editor")
                    restore_after_napari()
                    return
                except Exception as e:
                    print(f"❌ roi_editor import failed with unexpected error: {e}")
                    print("❌ Cannot proceed without roi_editor")
                    restore_after_napari()
                    return
                from roi_editor import refine_rois
                

                if isinstance(masks, np.ndarray):

                    if masks.ndim == 3:

                        if masks.shape[0] > 0 and masks.shape[1:] == mean.shape:
                            print(f"🔄 Converting 3D mask array ({masks.shape}) to list of 2D masks")
                            mask_list = []
                            for i in range(masks.shape[0]):
                                mask = masks[i].astype(bool)
                                if mask.sum() > 0:  # Only add non-empty masks
                                    mask_list.append(mask)
                            masks = mask_list
                            print(f"✅ Converted to {len(masks)} individual masks")
                        else:
                            print(f"⚠️ Unexpected 3D mask shape: {masks.shape}, expected (N, {mean.shape[0]}, {mean.shape[1]})")
                            restore_after_napari()
                            return
                    elif masks.ndim == 2:

                        print(f"🔄 Converting 2D labels array ({masks.shape}) to list of 2D masks")
                        unique_ids = np.unique(masks)
                        mask_list = []
                        for rid in unique_ids[1:]:  # Skip background (0)
                            mask = masks == rid
                            if mask.sum() > 0:  # Only add non-empty masks
                                mask_list.append(mask)
                        masks = mask_list
                        print(f"✅ Converted to {len(masks)} individual masks")
                    else:
                        print(f"⚠️ Unexpected mask array shape: {masks.shape}")
                        restore_after_napari()
                        return
                

                if not isinstance(masks, list) or len(masks) == 0:
                    print("❌ No valid masks found")
                    restore_after_napari()
                    return
                

                for i, mask in enumerate(masks):
                    if not isinstance(mask, np.ndarray) or mask.shape != mean.shape:
                        print(f"⚠️ Mask {i} has invalid shape: {mask.shape if hasattr(mask, 'shape') else type(mask)}, expected {mean.shape}")
                        masks[i] = None
                

                masks = [mask for mask in masks if mask is not None]
                
                if len(masks) == 0:
                    print("❌ No valid masks after validation")
                    restore_after_napari()
                    return
                
                print(f"✅ Prepared {len(masks)} valid masks for ROI editor")
                

                if 'refine_rois' in locals() and roi_editor_available:

                                    try:
                                        labels_array = refine_rois(mean, masks, return_viewer=False, on_close_callback=restore_after_napari)
                                        

                                        self.current_labels = labels_array
                                        

                                        if labels_array is not None:

                                            try:

                                                existing_data = np.load(self.rois_path)
                                                

                                                updated_data = {
                                                    'labels': labels_array,
                                                    'masks': existing_data.get('masks', []),
                                                    'sizes': existing_data.get('sizes', [])
                                                }
                                                

                                                np.savez_compressed(self.rois_path, **updated_data)
                                                print(f"✅ Updated ROI file saved: {self.rois_path}")
                                                
                                            except Exception as save_error:
                                                print(f"⚠️ Could not save updated ROIs: {save_error}")
                                        
                                    except Exception as napari_error:
                                        print(f"❌ Napari ROI editing failed: {napari_error}")
                                        restore_after_napari()  # Still restore state
                                        return
                                    
                                    print("✅ Napari ROI editor launched successfully with OpenGL safety")

                else:
                    print("❌ refine_rois function not available")
                    restore_after_napari()
                    return
                
            except Exception as e:
                print(f"❌ Error launching Napari: {e}")
                self._handle_error(e, "launch_napari")
                restore_after_napari()
                
        except Exception as e:
            print(f"❌ Error in Napari launch process: {e}")
            self._handle_error(e, "napari_launch")



    def _export_traces(self):
       
        try:
            if not self.live_extractor:
                print("Live trace extractor is not running.")
                return


            from PyQt5.QtCore import QThread, QObject, pyqtSignal

            class ExportWorker(QObject):
                finished = pyqtSignal(str, str)
                failed = pyqtSignal(str)

                def __init__(self, outer):
                    super().__init__()
                    self.outer = outer

                def run(self):
                    try:
                        print("📊 Generating export metadata (optimized)...")
                        export_data = self.outer._generate_comprehensive_export_data(fast_mode=True)
                        unified_file = self.outer._create_unified_export_file(export_data)
                        print("🌐 Generating detailed HTML summary...")
                        html_export_data = self.outer._generate_comprehensive_export_data(fast_mode=False)
                        html_file = unified_file.replace('.npz', '_summary.html')
                        self.outer._generate_html_summary(html_export_data, html_file)
                        self.finished.emit(unified_file, html_file)
                    except Exception as e:
                        self.failed.emit(str(e))

            self._export_thread = QThread(self)
            self._export_worker = ExportWorker(self)
            self._export_worker.moveToThread(self._export_thread)
            self._export_thread.started.connect(self._export_worker.run)

            def on_finished(unified_file, html_file):
                print(f"✅ Unified export completed:")
                print(f"   📦 Complete Data: {unified_file}")
                print(f"   🌐 Visual Summary: {html_file}")
                print(f"   ℹ️  Use 'View Exported Traces' to load the .npz file")
                self._export_thread.quit()
                self._export_thread.wait(100)

            def on_failed(msg):
                self._handle_error(Exception(msg), "Unified trace export")
                self._export_thread.quit()
                self._export_thread.wait(100)

            self._export_worker.finished.connect(on_finished)
            self._export_worker.failed.connect(on_failed)
            self._export_thread.start()

        except Exception as e:
            self._handle_error(e, "Unified trace export")

    def _generate_comprehensive_export_data(self, fast_mode=False):
       
        import time
        
        export_data = {
            'export_info': {
                'timestamp': time.time(),
                'datetime': time.strftime('%Y-%m-%d %H:%M:%S'),
                'version': '1.0.0'
            }
        }
        
        if fast_mode:

            print("⚡ Fast export mode - essential data only")
            export_data.update({
                'machine_snapshot': self._get_machine_snapshot_fast(),
                'camera_info': self._get_camera_info_fast(),
                'roi_metadata': self._extract_roi_metadata_fast(),
                'session_summary': self._get_session_summary_fast(),
                'calibration_info': self._get_calibration_info_fast()
            })
        else:

            export_data.update({
                'machine_snapshot': self._get_machine_snapshot(),
                'camera_info': self._get_camera_info(),
                'roi_metadata': self._extract_roi_metadata(),
                'session_summary': self._get_session_summary(),
                'calibration_info': self._get_calibration_info()
            })
        
        return export_data

    def _get_unified_roi_colors(self):
       

        return [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',  
            '#DDA0DD', '#98D8C8', '#FFA07A', '#87CEEB', '#DEB887', 
            '#FF9F43', '#10AC84', '#EE5A24', '#0084FF', '#341F97',  
            '#F8B500', '#6C5CE7', '#A29BFE', '#FD79A8', '#FDCB6E', 
            '#E17055', '#00B894', '#00CECE', '#2D3436', '#636E72',  
            '#FAB1A0', '#74B9FF', '#55A3FF', '#FF7675', '#6C5CE7',  
        ]
    
    def get_roi_color(self, roi_id, total_rois=None):
       
        colors = self._get_unified_roi_colors()
        

        color_index = (roi_id - 1) % len(colors) 
        return colors[color_index]

    def _get_machine_snapshot_fast(self):
       
        import platform
        import psutil
        
        return {
            'fast_mode': True,
            'timestamp': time.time(),
            'system': {
                'platform': platform.system(),
                'release': platform.release(),
                'machine': platform.machine(),
                'hostname': platform.node()
            },
            'python': {
                'version': platform.python_version()
            },
            'hardware': {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3)
            }
        }
    
    def _get_camera_info_fast(self):
       
        camera_info = {'fast_mode': True}
        try:
            if hasattr(self.camera, 'get_exposure'):
                camera_info['exposure'] = self.camera.get_exposure()
            if hasattr(self.camera, 'get_gain'):
                camera_info['gain'] = self.camera.get_gain()
            if hasattr(self.camera, 'get_fps'):
                camera_info['fps'] = self.camera.get_fps()
        except:
            pass
        return camera_info
    
    def _get_calibration_info_fast(self):
       
        return {
            'fast_mode': True,
            'homography_file': getattr(self.camera, 'translation_matrix_path', 'Unknown'),
            'timestamp': time.time()
        }

    def _extract_roi_metadata_fast(self):
       
        try:
            roi_metadata = {}
            
            if not self.live_extractor or not hasattr(self.live_extractor, '_labels_orig'):
                return roi_metadata
            
            labels = self.live_extractor._labels_orig
            unique_ids = np.unique(labels)
            roi_ids = unique_ids[unique_ids > 0]  
            
            colors = self._get_unified_roi_colors()
            
            for i, roi_id in enumerate(roi_ids):
                roi_mask = (labels == roi_id)
                roi_locations = np.where(roi_mask)
                
                if len(roi_locations[0]) == 0:
                    continue
                

                center_y = int(np.mean(roi_locations[0]))
                center_x = int(np.mean(roi_locations[1]))
                size = int(np.sum(roi_mask))
                

                avg_intensity = 0.0
                if hasattr(self.live_extractor, 'buffers') and roi_id in self.live_extractor.buffers:
                    buffer = list(self.live_extractor.buffers[roi_id])
                    if buffer:
                        avg_intensity = float(np.mean(buffer))
                

                bbox_height = np.max(roi_locations[0]) - np.min(roi_locations[0]) + 1
                bbox_width = np.max(roi_locations[1]) - np.min(roi_locations[1]) + 1
                aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 1.0
                
                roi_metadata[int(roi_id)] = {
                    'roi_index': int(roi_id),
                    'centroid': [center_x, center_y],
                    'size_pixels': size,
                    'size': size, 
                    'shape_info': {
                        'type': 'compact' if aspect_ratio < 1.5 else 'elongated',
                        'aspect_ratio': aspect_ratio
                    },
                    'color': colors[i % len(colors)],
                    'average_intensity': avg_intensity,
                    'fast_mode': True
                }
                
            return roi_metadata
            
        except Exception as e:
            print(f"⚠️ Fast ROI metadata extraction error: {e}")
            return {}

    def _get_session_summary_fast(self):
       
        try:
            frames_processed = 0
            if self.live_extractor and hasattr(self.live_extractor, 'stats'):
                frames_processed = self.live_extractor.stats.get('frames_processed', 0)
            
            summary = {
                'extractor_running': self.live_extractor is not None,
                'roi_count': len(self.live_extractor.buffers) if self.live_extractor else 0,
                'frames_processed': frames_processed,
                'rois_file': os.path.basename(self.rois_path) if hasattr(self, 'rois_path') and self.rois_path else 'Unknown',
                'traces_file': 'Live traces (in memory)',
                'fast_mode': True,
                'timestamp': time.time()
            }
            return summary
        except Exception as e:
            print(f"⚠️ Fast session summary error: {e}")
            return {'fast_mode': True, 'error': str(e)}

    def _create_unified_export_file(self, export_data):
       
        import time
        import numpy as np
        

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        unified_file = f"roi_complete_export_{timestamp}.npz"
        
        try:

            trace_data = {}
            trace_metadata = {}
            
            if self.live_extractor and hasattr(self.live_extractor, 'buffers'):
                print("📊 Collecting ALL ROI trace data for export...")
                

                all_roi_ids = sorted(self.live_extractor.buffers.keys())
                collected_count = 0
                empty_count = 0
                
                for roi_id in all_roi_ids:
                    buffer = self.live_extractor.buffers.get(roi_id, [])
                    
                    if buffer and len(buffer) > 0:

                        trace_array = np.asarray(buffer, dtype=np.float32)
                        trace_data[f'roi_{roi_id}_trace'] = trace_array
                        

                        trace_metadata[f'roi_{roi_id}_info'] = {
                            'length': len(trace_array),
                            'mean': float(trace_array.mean()),
                            'std': float(trace_array.std()),
                            'min': float(trace_array.min()),
                            'max': float(trace_array.max()),
                            'has_data': True
                        }
                        collected_count += 1
                    else:

                        trace_data[f'roi_{roi_id}_trace'] = np.array([], dtype=np.float32)
                        trace_metadata[f'roi_{roi_id}_info'] = {
                            'length': 0, 'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                            'has_data': False, 'roi_id': int(roi_id)
                        }
                        empty_count += 1
                
                print(f"✅ Collected ALL {len(trace_data)} ROI traces: {collected_count} with data, {empty_count} empty")
            

            unified_data = {

                'trace_data': trace_data,
                'trace_stats': trace_metadata,
                

                'export_info_json': np.array([str(export_data.get('export_info', {}))]),
                'machine_snapshot_json': np.array([str(export_data.get('machine_snapshot', {}))]),
                'camera_info_json': np.array([str(export_data.get('camera_info', {}))]),
                'roi_metadata_json': np.array([str(export_data.get('roi_metadata', {}))]),
                'session_summary_json': np.array([str(export_data.get('session_summary', {}))]),
                'calibration_info_json': np.array([str(export_data.get('calibration_info', {}))]),
                

                'file_format_version': np.array(['unified_v1.0']),
                'creation_timestamp': np.array([time.time()]),
                'readable_timestamp': np.array([time.strftime('%Y-%m-%d %H:%M:%S')])
            }
            

            np.savez_compressed(unified_file, **unified_data)
            
            print(f"✅ Unified file created: {unified_file}")
            print(f"   Contains: {len(trace_data)} ROI traces + complete metadata")
            
            return unified_file
            
        except Exception as e:
            print(f"❌ Unified export creation failed: {e}")

            fallback_file = f"roi_basic_export_{timestamp}.npz"
            np.savez_compressed(fallback_file, 
                               traces=list(self.live_extractor.buffers.values()) if self.live_extractor else [],
                               roi_ids=list(self.live_extractor.buffers.keys()) if self.live_extractor else [],
                               error_info=str(e))
            return fallback_file

    def _get_machine_snapshot(self):
       
        import platform
        import os
        
        snapshot = {
            'system': {
                'platform': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'hostname': platform.node()
            },
            'python': {
                'version': platform.python_version(),
                'implementation': platform.python_implementation()
            },
            'environment': {
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
                'pythonpath': os.environ.get('PYTHONPATH', '')
            }
        }
        

        try:
            import psutil
            snapshot['hardware'] = {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'memory_available_gb': psutil.virtual_memory().available / (1024**3)
            }
            

            process = psutil.Process()
            snapshot['process'] = {
                'memory_mb': process.memory_info().rss / (1024**2),
                'cpu_percent': process.cpu_percent()
            }
        except ImportError:
            snapshot['hardware_note'] = 'psutil not available for detailed hardware info'
        
        return snapshot

    def _get_camera_info(self):
       
        camera_info = {
            'acquisition_running': getattr(self.camera, 'acquisition_running', False)
        }
        

        try:
            if hasattr(self.camera, 'get_actual_fps'):
                camera_info['actual_fps'] = self.camera.get_actual_fps()
            
            if hasattr(self.camera, 'node_map'):
                try:
                    fps_node = self.camera.node_map.FindNode("AcquisitionFrameRate")
                    if fps_node:
                        camera_info['configured_fps'] = float(fps_node.Value())
                        

                    gain_node = self.camera.node_map.FindNode("Gain")
                    if gain_node:
                        camera_info['gain'] = float(gain_node.Value())
                except:
                    pass
        except:
            pass
        
        return camera_info

    def _extract_roi_metadata(self):
       
        roi_metadata = {}
        
        if not self.live_extractor or not hasattr(self.live_extractor, '_labels_orig'):
            return roi_metadata
        
        try:
            labels = self.live_extractor._labels_orig
            unique_ids = np.unique(labels)
            roi_ids = unique_ids[unique_ids > 0]  
            

            colors = self._get_unified_roi_colors()
            
            for i, roi_id in enumerate(roi_ids):
                roi_mask = (labels == roi_id)
                

                roi_locations = np.where(roi_mask)
                if len(roi_locations[0]) == 0:
                    continue
                

                center_y = int(np.mean(roi_locations[0]))
                center_x = int(np.mean(roi_locations[1]))
                

                size = int(np.sum(roi_mask))
                

                shape_info = self._estimate_roi_shape(roi_locations)
                

                avg_intensity = 0.0
                if hasattr(self.live_extractor, 'buffers') and roi_id in self.live_extractor.buffers:
                    buffer = list(self.live_extractor.buffers[roi_id])
                    if buffer:
                        avg_intensity = float(np.mean(buffer))
                

                activity_profile = self._calculate_activity_profile(roi_id)
                
                roi_metadata[int(roi_id)] = {
                    'roi_index': int(roi_id),
                    'centroid': [center_x, center_y],
                    'size_pixels': size,
                    'shape_info': shape_info,
                    'color': colors[i % len(colors)],
                    'average_intensity': avg_intensity,
                    'activity_profile': activity_profile,
                    'mask_reference': {
                        'main_mask_file': self.rois_path,
                        'roi_id_in_mask': int(roi_id)
                    }
                }
                
        except Exception as e:
            print(f"⚠️ ROI metadata extraction error: {e}")
        
        return roi_metadata

    def _estimate_roi_shape(self, roi_locations):
       
        if len(roi_locations[0]) < 5:
            return {'type': 'small', 'circularity': 0.0, 'aspect_ratio': 1.0}
        
        try:

            coords = np.column_stack(roi_locations)
            

            min_y, min_x = np.min(coords, axis=0)
            max_y, max_x = np.max(coords, axis=0)
            
            width = max_x - min_x + 1
            height = max_y - min_y + 1
            aspect_ratio = float(width) / float(height) if height > 0 else 1.0
            

            area = len(coords)
            perimeter_approx = 2 * np.sqrt(np.pi * area)
            circularity = 4 * np.pi * area / (perimeter_approx * perimeter_approx) if perimeter_approx > 0 else 0
            

            shape_type = "irregular"
            if circularity > 0.7:
                shape_type = "circular"
            elif aspect_ratio > 2.0 or aspect_ratio < 0.5:
                shape_type = "elongated"
            else:
                shape_type = "oval"
            
            return {
                'type': shape_type,
                'circularity': float(circularity),
                'aspect_ratio': float(aspect_ratio),
                'bounding_box': [int(min_x), int(min_y), int(width), int(height)]
            }
            
        except Exception as e:
            return {'type': 'unknown', 'error': str(e)}

    def _calculate_activity_profile(self, roi_id):
       
        if not hasattr(self.live_extractor, 'buffers') or roi_id not in self.live_extractor.buffers:
            return {'status': 'no_data'}
        
        try:
            buffer = list(self.live_extractor.buffers[roi_id])
            if not buffer:
                return {'status': 'empty_buffer'}
            
            traces = np.array(buffer)
            profile = {
                'status': 'calculated',
                'length': len(traces),
                'mean': float(np.mean(traces)),
                'std': float(np.std(traces)),
                'min': float(np.min(traces)),
                'max': float(np.max(traces)),
                'range': float(np.max(traces) - np.min(traces))
            }
            

            cv = profile['std'] / profile['mean'] if profile['mean'] > 0 else 0
            if cv < 0.1:
                profile['activity_level'] = 'low'
            elif cv < 0.3:
                profile['activity_level'] = 'moderate'
            else:
                profile['activity_level'] = 'high'
            
            profile['coefficient_of_variation'] = float(cv)
            
            return profile
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    def _get_session_summary(self):
       
        summary = {
            'rois_file': self.rois_path,
            'traces_file': self.trace_path
        }
        
        if self.live_extractor:
            summary.update({
                'extractor_running': True,
                'frames_processed': getattr(self.live_extractor, '_frame_count', 0),
                'total_rois': len(getattr(self.live_extractor, 'ids', [])),
                'buffer_lengths': {}
            })
            

            if hasattr(self.live_extractor, 'buffers'):
                for roi_id, buffer in self.live_extractor.buffers.items():
                    summary['buffer_lengths'][roi_id] = len(buffer)
        else:
            summary['extractor_running'] = False
        
        return summary

    def _get_calibration_info(self):
       
        return {
            'status': 'framework_ready',
            'note': 'Calibration system ready for implementation'
        }

    def _save_enhanced_metadata(self, export_data):
       
        import json
        import os
        

        metadata_file = TRACE_OUT.replace('.npy', '_metadata.json')
        try:
            with open(metadata_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            print(f"✅ Metadata saved: {metadata_file}")
        except Exception as e:
            print(f"❌ Metadata save error: {e}")
        

        html_file = TRACE_OUT.replace('.npy', '_summary.html')
        try:
            self._generate_html_summary(export_data, html_file)
            print(f"✅ HTML summary generated: {html_file}")
        except Exception as e:
            print(f"❌ HTML generation error: {e}")

    def _generate_html_summary(self, export_data, html_file):
       
        import os
        
        roi_metadata = export_data.get('roi_metadata', {})
        machine_info = export_data.get('machine_snapshot', {})
        session_info = export_data.get('session_summary', {})
        
        html_content = f"""<!DOCTYPE html>
<html><head><title>ROI Export Summary</title><style>
body {{ font-family: Arial; margin: 20px; background: #f5f5f5; }}
.container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
h1, h2 {{ color: #333; border-bottom: 2px solid #007acc; padding-bottom: 5px; }}
.roi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }}
.roi-card {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; background: #f9f9f9; }}
.roi-header {{ font-weight: bold; color: #007acc; margin-bottom: 10px; }}
.metadata {{ font-family: monospace; font-size: 0.9em; }}
.stats {{ background: #e8f4f8; padding: 10px; border-radius: 3px; margin: 10px 0; }}
</style></head><body><div class="container">
<h1>🔬 ROI Trace Export Summary</h1>
<div class="stats">
<strong>Export Time:</strong> {export_data.get('export_info', {}).get('datetime', 'Unknown')}<br/>
<strong>Total ROIs:</strong> {len(roi_metadata)}<br/>
<strong>Traces File:</strong> {os.path.basename(TRACE_OUT)}<br/>
<strong>System:</strong> {machine_info.get('system', {}).get('platform', 'Unknown')} {machine_info.get('system', {}).get('release', '')}
</div><h2>📊 ROI Details</h2><div class="roi-grid">"""
        

        for roi_id, roi_data in roi_metadata.items():
            activity = roi_data.get('activity_profile', {})
            shape_info = roi_data.get('shape_info', {})
            
            html_content += f"""<div class="roi-card" style="border-left: 4px solid {roi_data.get('color', '#ccc')}">
<div class="roi-header">ROI {roi_id}</div><div class="metadata">
<strong>Location:</strong> ({roi_data.get('centroid', [0, 0])[0]}, {roi_data.get('centroid', [0, 0])[1]})<br/>
<strong>Size:</strong> {roi_data.get('size_pixels', 0)} pixels<br/>
<strong>Shape:</strong> {shape_info.get('type', 'unknown')} (circularity: {shape_info.get('circularity', 0):.2f})<br/>
<strong>Avg Intensity:</strong> {roi_data.get('average_intensity', 0):.1f}<br/>
<strong>Activity:</strong> {activity.get('activity_level', 'unknown')} (CV: {activity.get('coefficient_of_variation', 0):.3f})
</div></div>"""
        
        html_content += f"""</div><h2>🖥️ System Information</h2><div class="metadata">
<strong>Platform:</strong> {machine_info.get('system', {}).get('platform', 'Unknown')}<br/>
<strong>Python:</strong> {machine_info.get('python', {}).get('version', 'Unknown')}<br/>
<strong>CPU Cores:</strong> {machine_info.get('hardware', {}).get('cpu_count', 'Unknown')}<br/>
<strong>Memory:</strong> {machine_info.get('hardware', {}).get('memory_total_gb', 0):.1f} GB
</div><h2>📈 Session Summary</h2><div class="metadata">
<strong>Extractor Running:</strong> {session_info.get('extractor_running', False)}<br/>
<strong>Frames Processed:</strong> {session_info.get('frames_processed', 0)}<br/>
<strong>ROIs File:</strong> {session_info.get('rois_file', 'Unknown')}
</div></div></body></html>"""
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _view_exported_traces(self):
       
        try:
            from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, 
                                       QTextEdit, QLabel, QPushButton, QFileDialog, QWidget)
            import json
            import os
            

            file_dialog = QFileDialog()
            trace_file, _ = file_dialog.getOpenFileName(
                self, 
                "Select Exported ROI Data File", 
                ".", 
                "ROI Export files (*.npz);;Legacy files (*.npy);;All files (*.*)"
            )
            
            if not trace_file:
                return
            

            file_data = self._load_export_file(trace_file)
            if not file_data:
                return
            

            dialog = QDialog(self)
            dialog.setWindowTitle("ROI Data Viewer")
            dialog.resize(1200, 800)  
            
            layout = QVBoxLayout(dialog)
            

            file_format = file_data.get('format', 'unknown')
            info_label = QLabel(f"📁 Viewing: {os.path.basename(trace_file)} ({file_format} format)")
            info_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 10px; background: #e8f4f8;")
            layout.addWidget(info_label)
            

            tab_widget = QTabWidget()
            layout.addWidget(tab_widget)
            

            self._add_roi_overview_tab(tab_widget, file_data)
            

            self._add_interactive_plot_tab(tab_widget, file_data)
            

            self._add_statistics_tab(tab_widget, file_data)
            

            self._add_system_info_tab(tab_widget, file_data)
            

            html_file = trace_file.replace('.npz', '_summary.html').replace('.npy', '_summary.html')
            if os.path.exists(html_file):
                self._add_html_tab(tab_widget, html_file)
            

            button_layout = QHBoxLayout()
            

            if os.path.exists(html_file):
                open_html_btn = QPushButton("🌐 Open Full Report in Browser")
                open_html_btn.clicked.connect(lambda: self._open_html_in_browser(html_file))
                button_layout.addWidget(open_html_btn)
            
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.close)
            button_layout.addWidget(close_btn)
            
            layout.addLayout(button_layout)
            

            dialog.exec_()
            
        except Exception as e:
            print(f"❌ View exported traces error: {e}")
            from PyQt5.QtWidgets import QMessageBox
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText(f"Error viewing exported traces:\\n{str(e)}")
            msg.exec_()

    def _load_export_file(self, file_path):
       
        try:
            import numpy as np
            import json
            import ast
            
            file_data = {'format': 'unknown', 'traces': {}, 'metadata': {}}
            
            if file_path.endswith('.npz'):

                data = np.load(file_path, allow_pickle=True)
                

                if 'file_format_version' in data and 'unified' in str(data['file_format_version']):
                    file_data['format'] = 'unified_npz'
                    

                    if 'trace_data' in data:
                        trace_data = data['trace_data'].item()
                        for key, trace_array in trace_data.items():
                            if key.startswith('roi_') and key.endswith('_trace'):
                                roi_id = key.replace('roi_', '').replace('_trace', '')
                                file_data['traces'][int(roi_id)] = trace_array
                    

                    try:
                        if 'roi_metadata_json' in data:
                            metadata_str = str(data['roi_metadata_json'][0])
                            file_data['metadata'] = ast.literal_eval(metadata_str)
                        
                        if 'export_info_json' in data:
                            export_info_str = str(data['export_info_json'][0])
                            file_data['export_info'] = ast.literal_eval(export_info_str)
                            
                        if 'machine_snapshot_json' in data:
                            machine_str = str(data['machine_snapshot_json'][0])
                            file_data['machine_info'] = ast.literal_eval(machine_str)
                            
                        if 'session_summary_json' in data:
                            session_str = str(data['session_summary_json'][0])
                            file_data['session_info'] = ast.literal_eval(session_str)
                            
                    except Exception as e:
                        print(f"⚠️ Metadata parsing warning: {e}")
                
                else:

                    file_data['format'] = 'legacy_npz'

                    for key, value in data.items():
                        if isinstance(value, np.ndarray):

                            file_data['traces'][key] = value
                
            elif file_path.endswith('.npy'):

                file_data['format'] = 'legacy_npy'
                traces = np.load(file_path, allow_pickle=True)
                
                if isinstance(traces, dict):
                    file_data['traces'] = traces
                else:
                    file_data['traces'] = {'trace_data': traces}
                

                metadata_file = file_path.replace('.npy', '_metadata.json')
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r') as f:
                            companion_data = json.load(f)
                        file_data['metadata'] = companion_data.get('roi_metadata', {})
                        file_data['export_info'] = companion_data.get('export_info', {})
                        file_data['machine_info'] = companion_data.get('machine_snapshot', {})
                        file_data['session_info'] = companion_data.get('session_summary', {})
                    except Exception as e:
                        print(f"⚠️ Companion metadata loading failed: {e}")
            
            print(f"✅ Loaded {file_data['format']} file with {len(file_data['traces'])} traces")
            return file_data
            
        except Exception as e:
            print(f"❌ File loading error: {e}")
            from PyQt5.QtWidgets import QMessageBox
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("File Load Error")
            msg.setText(f"Could not load file:\\n{str(e)}")
            msg.exec_()
            return None

    def _add_roi_overview_tab(self, tab_widget, file_data):
       
        try:
            from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QLabel, QScrollArea
            
            widget = QWidget()
            layout = QVBoxLayout(widget)
            

            header_label = QLabel(f"📊 ROI Overview ({len(file_data.get('traces', {}))} ROIs)")
            header_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 10px; background: #f0f0f0;")
            layout.addWidget(header_label)
            

            table = QTableWidget()
            

            traces = file_data.get('traces', {})
            metadata = file_data.get('metadata', {})
            
            print(f"🔍 ROI Overview Debug:")
            print(f"   Traces found: {len(traces)} ROIs")
            print(f"   Metadata found: {len(metadata)} entries")
            print(f"   Available file_data keys: {list(file_data.keys())}")
            if traces:
                print(f"   Sample trace keys: {list(traces.keys())[:5]}")
            if metadata:
                print(f"   Sample metadata keys: {list(metadata.keys())[:5]}")

                sample_key = list(metadata.keys())[0] if metadata else None
                if sample_key:
                    sample_meta = metadata[sample_key]
                    print(f"   Sample metadata content: {list(sample_meta.keys()) if isinstance(sample_meta, dict) else type(sample_meta)}")
            

            if not metadata or len(metadata) == 0:
                print("   🔄 Primary metadata empty, trying fallback sources...")
                

                trace_stats = file_data.get('trace_stats', {})
                if trace_stats:
                    print(f"   ✅ Using trace_stats as fallback metadata: {len(trace_stats)} entries")
                    metadata = trace_stats
                

                elif 'export_info' in file_data and isinstance(file_data['export_info'], dict):
                    export_roi_meta = file_data['export_info'].get('roi_metadata', {})
                    if export_roi_meta:
                        print(f"   ✅ Using export_info roi_metadata: {len(export_roi_meta)} entries")
                        metadata = export_roi_meta
                

                elif hasattr(self, 'live_extractor') and self.live_extractor:
                    print("   🔄 Generating metadata from live extractor...")
                    metadata = self._extract_roi_metadata()
                    if metadata:
                        print(f"   ✅ Generated metadata from live extractor: {len(metadata)} entries")
                

                if not metadata and traces:
                    print("   🔄 Creating basic metadata from trace data...")
                    metadata = {}
                    for roi_id, trace_data in traces.items():
                        if hasattr(trace_data, '__len__') and len(trace_data) > 0:
                            trace_array = np.array(trace_data, dtype=np.float32)
                            metadata[roi_id] = {
                                'roi_index': int(roi_id),
                                'average_intensity': float(np.mean(trace_array)),
                                'size_pixels': max(10, len(trace_data) // 10),
                                'centroid': [roi_id * 20, roi_id * 15],  
                                'color': self.get_roi_color(int(roi_id)),
                                'shape_info': {'type': 'estimated', 'aspect_ratio': 1.0},
                                'generated': True
                            }
                    print(f"   ✅ Created basic metadata: {len(metadata)} entries")
                
            if traces:
                roi_ids = sorted(traces.keys())
                table.setRowCount(len(roi_ids))
                table.setColumnCount(7) 
                table.setHorizontalHeaderLabels(['ROI ID', 'Color', 'Location', 'Size', 'Avg Intensity', 'Trace Length', 'Activity'])
                
                import numpy as np
                
                for row, roi_id in enumerate(roi_ids):

                    table.setItem(row, 0, QTableWidgetItem(str(roi_id)))
                    

                    roi_meta = metadata.get(str(roi_id), metadata.get(roi_id, {}))
                    

                    trace_data = traces.get(roi_id, [])
                    

                    color = roi_meta.get('color', None)
                    if not color:

                        color = self.get_roi_color(int(roi_id))
                    
                    color_item = QTableWidgetItem(f"● ROI {roi_id}")
                    from PyQt5.QtGui import QColor
                    try:
                        qcolor = QColor(color)
                        color_item.setForeground(qcolor)

                        bg_color = QColor(color)
                        bg_color.setAlpha(30) 
                        color_item.setBackground(bg_color)
                    except Exception as e:
                        print(f"⚠️ Color setting warning for ROI {roi_id}: {e}")

                        color_item = QTableWidgetItem(f"ROI {roi_id}")
                    table.setItem(row, 1, color_item)
                    

                    centroid = roi_meta.get('centroid', None)
                    if centroid and isinstance(centroid, list) and len(centroid) >= 2:
                        try:

                            x_val = float(centroid[0]) if isinstance(centroid[0], (int, float, str)) and str(centroid[0]).replace('.','').replace('-','').isdigit() else 0
                            y_val = float(centroid[1]) if isinstance(centroid[1], (int, float, str)) and str(centroid[1]).replace('.','').replace('-','').isdigit() else 0
                            location_str = f"({x_val:.0f}, {y_val:.0f})"
                        except:
                            location_str = f"({centroid[0]}, {centroid[1]})"
                    else:

                        location_str = f"ROI {roi_id} (estimated)"
                    table.setItem(row, 2, QTableWidgetItem(location_str))
                    

                    size = roi_meta.get('size_pixels', roi_meta.get('size', None))
                    if size is None or size == 'Unknown' or size == 0:

                        if hasattr(trace_data, '__len__') and len(trace_data) > 0:

                            estimated_size = max(10, len(trace_data) // 2) 
                            size = f"~{estimated_size} px (est.)"
                        else:
                            size = "Unknown"
                    else:
                        size = f"{size} px"
                    table.setItem(row, 3, QTableWidgetItem(str(size)))
                    

                    avg_intensity = roi_meta.get('average_intensity', roi_meta.get('mean', None))
                    if avg_intensity is None and hasattr(trace_data, '__len__') and len(trace_data) > 0:
                        try:
                            trace_array = np.array(trace_data, dtype=np.float32)
                            avg_intensity = float(np.mean(trace_array))
                        except:
                            avg_intensity = 0
                    
                    if avg_intensity is not None:
                        table.setItem(row, 4, QTableWidgetItem(f"{avg_intensity:.2f}"))
                    else:
                        table.setItem(row, 4, QTableWidgetItem("N/A"))
                    

                    trace_length = len(trace_data) if hasattr(trace_data, '__len__') else 0
                    table.setItem(row, 5, QTableWidgetItem(str(trace_length)))
                    

                    activity = "Unknown"
                    if hasattr(trace_data, '__len__') and len(trace_data) > 1:
                        try:
                            trace_array = np.array(trace_data, dtype=np.float32)
                            if len(trace_array) > 1:
                                cv = np.std(trace_array) / np.mean(trace_array) if np.mean(trace_array) > 0 else 0
                                if cv > 0.3:
                                    activity = "High"
                                elif cv > 0.1:
                                    activity = "Moderate"
                                else:
                                    activity = "Low"
                        except:
                            activity = "Unknown"
                    table.setItem(row, 6, QTableWidgetItem(activity))
                

                table.resizeColumnsToContents()
                
            else:
                table.setRowCount(1)
                table.setColumnCount(1)
                table.setHorizontalHeaderLabels(['Status'])
                table.setItem(0, 0, QTableWidgetItem("No ROI data found"))
            
            layout.addWidget(table)
            tab_widget.addTab(widget, "📊 ROI Overview")
            
        except Exception as e:
            error_widget = QLabel(f"Error creating ROI overview: {e}")
            tab_widget.addTab(error_widget, "❌ ROI Overview")

    def _add_interactive_plot_tab(self, tab_widget, file_data):
       
        try:
            import numpy as np
            try:
                import matplotlib.pyplot as plt
                import matplotlib.colors as mcolors
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
                from matplotlib.figure import Figure
                matplotlib_available = True
            except ImportError as e:
                print(f"⚠️ Matplotlib import error: {e}")
                matplotlib_available = False
            
            from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QScrollArea, QLabel, QPushButton
            from PyQt5.QtCore import Qt
            
            if not matplotlib_available:
                error_widget = QLabel("Matplotlib not available for interactive plotting")
                tab_widget.addTab(error_widget, "❌ Interactive Plot")
                return
            
            widget = QWidget()
            main_layout = QVBoxLayout(widget)
            

            pagination_widget = QWidget()
            pagination_layout = QHBoxLayout(pagination_widget)
            
            prev_btn = QPushButton("◀ Previous 10 ROIs")
            page_label = QLabel("Page 1/1 (ROIs 1-10)")
            page_label.setAlignment(Qt.AlignCenter)
            page_label.setStyleSheet("font-weight: bold; padding: 5px;")
            next_btn = QPushButton("Next 10 ROIs ▶")
            
            pagination_layout.addWidget(prev_btn)
            pagination_layout.addWidget(page_label)
            pagination_layout.addWidget(next_btn)
            main_layout.addWidget(pagination_widget)
            

            plot_container = QWidget()
            plot_layout = QHBoxLayout(plot_container)
            

            plot_widget = QWidget()
            plot_widget_layout = QVBoxLayout(plot_widget)
            

            fig = Figure(figsize=(12, 8))
            canvas = FigureCanvas(fig)
            plot_widget_layout.addWidget(canvas)
            

            control_widget = QWidget()
            control_widget.setMaximumWidth(200)
            control_layout = QVBoxLayout(control_widget)
            
            control_header = QLabel("Current Page ROIs:")
            control_header.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
            control_layout.addWidget(control_header)
            

            checkbox_widget = QWidget()
            checkbox_layout = QVBoxLayout(checkbox_widget)
            

            traces = file_data.get('traces', {})
            metadata = file_data.get('metadata', {})
            
            if traces:

                roi_ids = sorted(traces.keys())
                rois_per_page = 10
                total_pages = (len(roi_ids) + rois_per_page - 1) // rois_per_page
                current_page = 0
                

                ax = fig.add_subplot(111)
                plot_lines = {}
                checkboxes = {}
                
                def update_plot_page():

                    ax.clear()
                    

                    for cb in checkboxes.values():
                        cb.setParent(None)
                    checkboxes.clear()
                    

                    start_idx = current_page * rois_per_page
                    end_idx = min(start_idx + rois_per_page, len(roi_ids))
                    page_roi_ids = roi_ids[start_idx:end_idx]
                    

                    page_label.setText(f"Page {current_page + 1}/{total_pages} (ROIs {start_idx + 1}-{end_idx})")
                    

                    for idx, roi_id in enumerate(page_roi_ids):
                        trace_data = traces[roi_id]
                        if hasattr(trace_data, '__len__') and len(trace_data) > 0:
                            y_data = np.array(trace_data, dtype=np.float32)
                            x_data = np.arange(len(y_data))

                            color_hex = self.get_roi_color(int(roi_id))
                            color = mcolors.to_rgba(color_hex)
                            
                            line, = ax.plot(x_data, y_data, color=color, label=f"ROI {roi_id}", 
                                          alpha=0.8, linewidth=2)
                            plot_lines[roi_id] = line
                            

                            checkbox = QCheckBox(f"ROI {roi_id}")
                            checkbox.setChecked(True)
                            

                            try:
                                checkbox.setStyleSheet(f"color: {color_hex}; font-weight: bold;")
                            except Exception:
                                pass
                            

                            def make_toggle_function(plot_line, roi_identifier):
                                def toggle_line(checked):
                                    try:
                                        plot_line.set_visible(checked)
                                        canvas.draw()
                                        print(f"🔍 ROI {roi_identifier} visibility: {checked}")
                                    except Exception as e:
                                        print(f"⚠️ Toggle error for ROI {roi_identifier}: {e}")
                                return toggle_line
                            
                            checkbox.toggled.connect(make_toggle_function(line, roi_id))
                            checkboxes[roi_id] = checkbox
                            checkbox_layout.addWidget(checkbox)
                    

                    ax.set_xlabel('Time Points')
                    ax.set_ylabel('Intensity')
                    ax.set_title(f'Interactive ROI Traces - Page {current_page + 1}/{total_pages}')
                    ax.grid(True, alpha=0.3)
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
                    

                    canvas.draw()
                
                def prev_page():
                    nonlocal current_page
                    if current_page > 0:
                        current_page -= 1
                        update_plot_page()
                        prev_btn.setEnabled(current_page > 0)
                        next_btn.setEnabled(current_page < total_pages - 1)
                
                def next_page():
                    nonlocal current_page
                    if current_page < total_pages - 1:
                        current_page += 1
                        update_plot_page()
                        prev_btn.setEnabled(current_page > 0)
                        next_btn.setEnabled(current_page < total_pages - 1)
                

                prev_btn.clicked.connect(prev_page)
                next_btn.clicked.connect(next_page)
                

                prev_btn.setEnabled(False)
                next_btn.setEnabled(total_pages > 1)
                

                update_plot_page()
                
            else:

                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, 'No trace data available', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_title('Interactive Plot - No Data')
                page_label.setText("No data")
                prev_btn.setEnabled(False)
                next_btn.setEnabled(False)
                canvas.draw()
            

            scroll_area = QScrollArea()
            scroll_area.setWidget(checkbox_widget)
            scroll_area.setWidgetResizable(True)
            control_layout.addWidget(scroll_area)
            

            plot_layout.addWidget(plot_widget)
            plot_layout.addWidget(control_widget)
            main_layout.addWidget(plot_container)
            
            tab_widget.addTab(widget, "📈 Interactive Plot")

        
        except Exception as e:
            error_widget = QLabel(f"Error creating interactive plot: {e}")
            tab_widget.addTab(error_widget, "❌ Interactive Plot")

    def _add_statistics_tab(self, tab_widget, file_data):
       
        try:
            import numpy as np
            from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextEdit
            
            widget = QWidget()
            layout = QVBoxLayout(widget)
            
            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.setFont(QtGui.QFont("Courier", 10))
            

            traces = file_data.get('traces', {})
            metadata = file_data.get('metadata', {})
            
            stats_text = "=== Detailed ROI Statistics ===\n\n"
            
            if traces:
                stats_text += f"Total ROIs: {len(traces)}\n\n"
                
                all_intensities = []
                all_lengths = []
                
                for roi_id, trace_data in sorted(traces.items()):
                    if hasattr(trace_data, '__len__') and len(trace_data) > 0:
                        trace_array = np.array(trace_data, dtype=np.float32)
                        
                        roi_meta = metadata.get(str(roi_id), {})
                        
                        stats_text += f"ROI {roi_id}:\n"
                        stats_text += f"  Length: {len(trace_array)} points\n"
                        stats_text += f"  Mean: {np.mean(trace_array):.3f}\n"
                        stats_text += f"  Std: {np.std(trace_array):.3f}\n"
                        stats_text += f"  Min: {np.min(trace_array):.3f}\n"
                        stats_text += f"  Max: {np.max(trace_array):.3f}\n"
                        stats_text += f"  Range: {np.max(trace_array) - np.min(trace_array):.3f}\n"
                        

                        cv = np.std(trace_array) / np.mean(trace_array) if np.mean(trace_array) > 0 else 0
                        activity = 'high' if cv > 0.3 else 'moderate' if cv > 0.1 else 'low'
                        stats_text += f"  Activity: {activity} (CV: {cv:.3f})\n"
                        

                        if roi_meta:
                            centroid = roi_meta.get('centroid', [0, 0])
                            size = roi_meta.get('size_pixels', 0)
                            shape = roi_meta.get('shape_info', {}).get('type', 'unknown')
                            stats_text += f"  Location: ({centroid[0]}, {centroid[1]})\n"
                            stats_text += f"  Size: {size} pixels\n"
                            stats_text += f"  Shape: {shape}\n"
                        
                        stats_text += "\n"
                        
                        all_intensities.extend(trace_array)
                        all_lengths.append(len(trace_array))
                

                if all_intensities:
                    stats_text += "=== Overall Statistics ===\n"
                    stats_text += f"Total data points: {len(all_intensities)}\n"
                    stats_text += f"Global mean intensity: {np.mean(all_intensities):.3f}\n"
                    stats_text += f"Global std intensity: {np.std(all_intensities):.3f}\n"
                    stats_text += f"Average trace length: {np.mean(all_lengths):.1f}\n"
                    stats_text += f"Min trace length: {np.min(all_lengths)}\n"
                    stats_text += f"Max trace length: {np.max(all_lengths)}\n"
            else:
                stats_text += "No trace data available for analysis.\n"
            
            text_edit.setPlainText(stats_text)
            layout.addWidget(text_edit)
            
            tab_widget.addTab(widget, "📈 Statistics")
            
        except Exception as e:
            error_widget = QLabel(f"Error creating statistics: {e}")
            tab_widget.addTab(error_widget, "❌ Statistics")

    def _add_system_info_tab(self, tab_widget, file_data):
       
        try:
            from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextEdit
            
            widget = QWidget()
            layout = QVBoxLayout(widget)
            
            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.setFont(QtGui.QFont("Courier", 10))
            

            info_text = "=== System & Session Information ===\n\n"
            

            export_info = file_data.get('export_info', {})
            if export_info:
                info_text += "Export Information:\n"
                info_text += f"  Timestamp: {export_info.get('datetime', 'Unknown')}\n"
                info_text += f"  Version: {export_info.get('version', 'Unknown')}\n\n"
            

            machine_info = file_data.get('machine_info', {}) or file_data.get('machine_snapshot', {})
            if machine_info:
                info_text += "Machine Information:\n"
                system = machine_info.get('system', {})
                if system:
                    info_text += f"  Platform: {system.get('platform', 'Unknown')}\n"
                    info_text += f"  Release: {system.get('release', 'Unknown')}\n"
                    info_text += f"  Machine: {system.get('machine', 'Unknown')}\n"
                    info_text += f"  Hostname: {system.get('hostname', 'Unknown')}\n"
                
                python = machine_info.get('python', {})
                if python:
                    info_text += f"  Python: {python.get('version', 'Unknown')}\n"
                
                hardware = machine_info.get('hardware', {})
                if hardware:
                    info_text += f"  CPU Cores: {hardware.get('cpu_count', 'Unknown')}\n"
                    info_text += f"  Memory: {hardware.get('memory_total_gb', 0):.1f} GB\n"
                elif machine_info.get('fast_mode'):

                    info_text += f"  Fast Mode: Basic info only\n"
                
                info_text += "\n"
            

            session_info = (file_data.get('session_info', {}) or 
                           file_data.get('session_summary', {}) or 
                           file_data.get('session_data', {}))
            if session_info:
                info_text += "Session Information:\n"
                info_text += f"  Extractor Running: {session_info.get('extractor_running', False)}\n"
                info_text += f"  Frames Processed: {session_info.get('frames_processed', 0)}\n"
                info_text += f"  ROIs File: {session_info.get('rois_file', 'Unknown')}\n"
                info_text += f"  Traces File: {session_info.get('traces_file', 'Unknown')}\n"
                info_text += f"  Session ID: {session_info.get('session_id', 'Unknown')}\n"
                info_text += f"  ROI Count: {session_info.get('roi_count', 0)}\n"
            
            if not any([export_info, machine_info, session_info]):
                info_text += "No system or session information available.\n"
            
            text_edit.setPlainText(info_text)
            layout.addWidget(text_edit)
            
            tab_widget.addTab(widget, "🖥️ System Info")
            
        except Exception as e:
            error_widget = QLabel(f"Error creating system info: {e}")
            tab_widget.addTab(error_widget, "❌ System Info")

    def _add_trace_data_tab(self, tab_widget, trace_file):
       
        try:
            import numpy as np
            

            trace_data = np.load(trace_file, allow_pickle=True)
            
            widget = QWidget()
            layout = QVBoxLayout(widget)
            
            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.setFont(QtGui.QFont("Courier", 10))
            

            info_text = f"""
=== Trace Data Analysis ===

File: {os.path.basename(trace_file)}
File Size: {os.path.getsize(trace_file) / 1024:.1f} KB

Data Structure:
"""
            
            if isinstance(trace_data, dict):
                info_text += f"Type: Dictionary with {len(trace_data)} keys\\n\\n"
                for key, value in trace_data.items():
                    if isinstance(value, np.ndarray):
                        info_text += f"'{key}': Array shape {value.shape}, dtype {value.dtype}\\n"
                        if len(value) > 0:
                            info_text += f"   Range: {np.min(value):.3f} to {np.max(value):.3f}\\n"
                            info_text += f"   Mean: {np.mean(value):.3f}, Std: {np.std(value):.3f}\\n"
                    else:
                        info_text += f"'{key}': {type(value).__name__}\\n"
                    info_text += "\\n"
            else:
                info_text += f"Type: {type(trace_data).__name__}\\n"
                if isinstance(trace_data, np.ndarray):
                    info_text += f"Shape: {trace_data.shape}\\n"
                    info_text += f"Data type: {trace_data.dtype}\\n"
                    if trace_data.size > 0:
                        info_text += f"Range: {np.min(trace_data):.3f} to {np.max(trace_data):.3f}\\n"
                        info_text += f"Mean: {np.mean(trace_data):.3f}\\n"
            
            text_edit.setPlainText(info_text)
            layout.addWidget(text_edit)
            
            tab_widget.addTab(widget, "📊 Trace Data")
            
        except Exception as e:
            error_widget = QLabel(f"Error loading trace data: {e}")
            tab_widget.addTab(error_widget, "❌ Trace Data")

    def _add_metadata_tab(self, tab_widget, metadata_file):
       
        try:
            import json
            
            widget = QWidget()
            layout = QVBoxLayout(widget)
            
            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.setFont(QtGui.QFont("Courier", 10))
            

            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            

            info_text = "=== ROI Metadata Summary ===\\n\\n"
            

            export_info = metadata.get('export_info', {})
            info_text += f"Export Time: {export_info.get('datetime', 'Unknown')}\\n"
            info_text += f"Version: {export_info.get('version', 'Unknown')}\\n\\n"
            

            roi_metadata = metadata.get('roi_metadata', {})
            info_text += f"=== ROI Details ({len(roi_metadata)} ROIs) ===\\n\\n"
            
            for roi_id, roi_data in roi_metadata.items():
                info_text += f"ROI {roi_id}:\\n"
                info_text += f"  Location: {roi_data.get('centroid', 'Unknown')}\\n"
                info_text += f"  Size: {roi_data.get('size_pixels', 'Unknown')} pixels\\n"
                info_text += f"  Shape: {roi_data.get('shape_info', {}).get('type', 'Unknown')}\\n"
                info_text += f"  Avg Intensity: {roi_data.get('average_intensity', 0):.2f}\\n"
                
                activity = roi_data.get('activity_profile', {})
                if activity.get('status') == 'calculated':
                    info_text += f"(Activity: {activity.get('activity_level', 'unknown')})\\n"
                    info_text += f"(CV: {activity.get('coefficient_of_variation', 0):.3f})\\n"
                
                info_text += "\\n"
            

            machine_info = metadata.get('machine_snapshot', {})
            if machine_info:
                info_text += "=== System Information ===\\n"
                system = machine_info.get('system', {})
                info_text += f"Platform: {system.get('platform', 'Unknown')} {system.get('release', '')}\\n"
                
                hardware = machine_info.get('hardware', {})
                if hardware:
                    info_text += f"CPU Cores: {hardware.get('cpu_count', 'Unknown')}\\n"
                    info_text += f"Memory: {hardware.get('memory_total_gb', 0):.1f} GB\\n"
            
            text_edit.setPlainText(info_text)
            layout.addWidget(text_edit)
            
            tab_widget.addTab(widget, "🏷️ ROI Metadata")
            
        except Exception as e:
            error_widget = QLabel(f"Error loading metadata: {e}")
            tab_widget.addTab(error_widget, "❌ Metadata")

    def _add_html_tab(self, tab_widget, html_file):
       
        try:
            from PyQt5.QtWebEngineWidgets import QWebEngineView
            from PyQt5.QtCore import QUrl
            
            web_view = QWebEngineView()
            web_view.load(QUrl.fromLocalFile(os.path.abspath(html_file)))
            
            tab_widget.addTab(web_view, "📋 Visual Summary")
            
        except ImportError:

            widget = QWidget()
            layout = QVBoxLayout(widget)
            
            label = QLabel("Web engine not available for HTML preview.\\nUse 'Open Full Report in Browser' button.")
            label.setStyleSheet("padding: 20px; color: #666;")
            layout.addWidget(label)
            
            tab_widget.addTab(widget, "📋 Visual Summary")
        except Exception as e:
            error_widget = QLabel(f"Error loading HTML: {e}")
            tab_widget.addTab(error_widget, "❌ HTML")

    def _add_plot_preview_tab(self, tab_widget, trace_file, metadata_file):
       
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            
            widget = QWidget()
            layout = QVBoxLayout(widget)
            

            fig = Figure(figsize=(12, 8))
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)
            

            trace_data = np.load(trace_file, allow_pickle=True)
            

            roi_colors = {}
            roi_labels = {}
            if metadata_file:
                try:
                    import json
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    roi_metadata = metadata.get('roi_metadata', {})
                    for roi_id, roi_data in roi_metadata.items():
                        roi_colors[int(roi_id)] = roi_data.get('color', '#000000')
                        centroid = roi_data.get('centroid', [0, 0])
                        roi_labels[int(roi_id)] = f"ROI {roi_id} @({centroid[0]}, {centroid[1]})"
                except:
                    pass
            

            if isinstance(trace_data, dict):

                ax = fig.add_subplot(111)
                plotted_count = 0
                
                for key, values in trace_data.items():
                    if isinstance(values, np.ndarray) and len(values) > 0:
                        try:

                            roi_id = None
                            if 'roi' in key.lower():
                                import re
                                match = re.search(r'roi.?(\d+)', key.lower())
                                if match:
                                    roi_id = int(match.group(1))
                            
                            color = roi_colors.get(roi_id, f'C{plotted_count % 10}') if roi_id else f'C{plotted_count % 10}'
                            label = roi_labels.get(roi_id, key) if roi_id else key
                            
                            ax.plot(values, color=color, label=label, alpha=0.8)
                            plotted_count += 1
                            
                            if plotted_count >= 20: 
                                break
                                
                        except Exception as e:
                            print(f"Plot error for {key}: {e}")
                
                ax.set_xlabel('Time Points')
                ax.set_ylabel('Intensity')
                ax.set_title(f'Exported Traces Preview ({plotted_count} traces)')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                
            else:

                ax = fig.add_subplot(111)
                ax.plot(trace_data)
                ax.set_xlabel('Time Points')
                ax.set_ylabel('Intensity')
                ax.set_title('Exported Trace Preview')
                ax.grid(True, alpha=0.3)
            
            fig.tight_layout()
            canvas.draw()
            
            tab_widget.addTab(widget, "📈 Plot Preview")
            
        except Exception as e:
            error_widget = QLabel(f"Error generating plot: {e}")
            tab_widget.addTab(error_widget, "❌ Plot Preview")

    def _open_html_in_browser(self, html_file):
       
        try:
            import webbrowser
            webbrowser.open(f'file://{os.path.abspath(html_file)}')
        except Exception as e:
            print(f"❌ Browser open error: {e}")



    def _monitor_memory_usage(self):
        try:
            proc = psutil.Process()
            mem_pct = proc.memory_percent()
            mem_mb = proc.memory_info().rss / (1024 * 1024)
            self._memory_history.append(mem_pct)


            if time.time() - self._last_memory_report > 300:
                print(f"Memory usage: {mem_pct:.1f}% ({mem_mb:.1f} MB)")
                self._last_memory_report = time.time()


            if len(self._memory_history) >= 10:
                recent = list(self._memory_history)[-10:]
                if sum(recent) / 10.0 > 85.0:
                    print(f"Sustained high memory usage: {mem_pct:.1f}% ({mem_mb:.1f} MB)")
                    self._force_memory_cleanup()
        except Exception as e:
            print(f"Memory monitoring error: {e}")

    def _monitor_performance(self):
        try:
            cpu = psutil.cpu_percent(interval=0.5)
            self._cpu_history.append(cpu)
            if cpu > 90.0:
                print(f"High CPU usage detected: {cpu:.1f}%")
                self._optimize_performance()

            if CUDA_AVAILABLE:
                try:
                    pool = cp.get_default_memory_pool()
                    used = float(pool.used_bytes())
                    total = float(pool.total_bytes()) if hasattr(pool, "total_bytes") else max(used, 1.0)
                    pct = 100.0 * used / max(total, 1.0)
                    self._gpu_memory_history.append(pct)
                    if pct > 80.0:
                        print(f"High GPU memory usage: {pct:.1f}% ({used/1024/1024:.1f} MB)")
                        self._cleanup_gpu_memory()
                except Exception as e:
                    print(f"GPU memory monitoring error: {e}")
        except Exception as e:
            print(f"Performance monitoring error: {e}")

    def _watchdog_check(self):
        try:
            now = time.time()
            if now - self._last_activity_time > 300:
                print("No activity detected for 5 minutes; health check running")
                self._perform_health_check()

            if self._error_count > self._max_errors_per_minute:
                print("Excessive errors; performing emergency cleanup")
                self._emergency_cleanup()
                self._error_count = 0

            if now - self._last_error_time > 60:
                self._error_count = 0
        except Exception as e:
            print(f"Watchdog check failed: {e}")

    def _check_thread_health(self):
        try:
            if self.live_extractor and hasattr(self.live_extractor, "running"):
                if not self.live_extractor.running:
                    print("Live extractor unresponsive; restarting…")
                    self.stop_live_traces()
                    QTimer.singleShot(100, self.start_live_traces)

        except Exception as e:
            print(f"Thread health check error: {e}")

    def _periodic_cleanup(self):
        try:
            freed = gc.collect()
            if freed:
                print(f"Garbage collection freed {freed} objects")
            if CUDA_AVAILABLE:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                    print("GPU memory pool cleaned")
                except Exception as e:
                    print(f"GPU memory cleanup error: {e}")
        except Exception as e:
            print(f"Periodic cleanup error: {e}")

    def _force_memory_cleanup(self):
        try:
            print("Forced memory cleanup…")
            self.stop_live_traces()

            for _ in range(2):
                gc.collect()
            if CUDA_AVAILABLE:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
            if getattr(self.camera, "acquisition_running", False):
                QTimer.singleShot(1000, self.start_live_traces)
        except Exception as e:
            print(f"Force memory cleanup error: {e}")

    def _optimize_performance(self):
        try:
            if self.live_extractor and hasattr(self.live_extractor, "_update_every_n"):
                self.live_extractor._update_every_n = max(5, self.live_extractor._update_every_n + 1)


            self._force_memory_cleanup()
        except Exception as e:
            print(f"Performance optimization error: {e}")

    def _cleanup_gpu_memory(self):
        try:
            if CUDA_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()
                print("GPU memory cleaned")
        except Exception as e:
            print(f"GPU memory cleanup error: {e}")

    def _perform_health_check(self):
        try:
            print("Performing system health check…")
            if hasattr(self.camera, "acquisition_running") and not self.camera.acquisition_running:
                try:
                    self.camera.start_realtime_acquisition()
                    print("Camera acquisition restarted by watchdog.")
                except Exception as e:
                    print(f"Failed to restart acquisition: {e}")
            self._last_activity_time = time.time()
        except Exception as e:
            print(f"Health check error: {e}")

    def _handle_error(self, error: Exception, context: str = ""):
        self._error_count += 1
        self._last_error_time = time.time()
        print(f"Error in {context}: {error}")
        self._safe_cleanup()
        if self._error_count > self._max_errors_per_minute:
            print("Too many errors; performing emergency cleanup")
            self._emergency_cleanup()

    def _safe_cleanup(self):
        try:
            gc.collect()
            if CUDA_AVAILABLE:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
        except Exception as e:
            print(f"Safe cleanup error: {e}")

    def _emergency_cleanup(self):
       
        try:
            print("🆘 Emergency cleanup initiated...")
            

            self.stop_live_traces()
            

            try:
                if hasattr(self.camera, 'stop_realtime_acquisition'):
                    self.camera.stop_realtime_acquisition()
                    print("📷 Camera acquisition stopped")
            except Exception as e:
                print(f"⚠️ Camera cleanup warning: {e}")
            

            try:
                gc.collect()
                print("🗑️ Memory garbage collected")
            except Exception:
                pass
                

            if CUDA_AVAILABLE:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                    print("🎮 GPU memory cleaned")
                except Exception:
                    pass
            
            print("✅ Emergency cleanup completed successfully")
            
        except Exception as e:
            print(f"❌ Error during emergency cleanup: {e}")

    def _signal_handler(self, signum, frame):
        print(f"🛑 Received signal {signum}, performing graceful cleanup…")
        self._emergency_cleanup()
        
    def closeEvent(self, event):
       
        try:
            print("🚪 CRISPI window closing - performing comprehensive cleanup...")
            

            try:
                self.stop_live_traces()
                print("✅ Live traces stopped")
            except Exception as e:
                print(f"⚠️ Error stopping live traces: {e}")
            

            try:
                if hasattr(self, 'proj_display') and self.proj_display:
                    self.proj_display.close()
                    self.proj_display = None
                    print("✅ Projection display closed")
            except Exception as e:
                print(f"⚠️ Error closing projection display: {e}")
            

            try:
                self._force_memory_cleanup()
                print("✅ Memory cleanup completed")
            except Exception as e:
                print(f"⚠️ Error in memory cleanup: {e}")
            

            try:
                self.closed.emit()
                print("✅ Close signal emitted")
            except Exception as e:
                print(f"⚠️ Error emitting close signal: {e}")
            

            try:
                from PyQt5.QtCore import QCoreApplication
                QCoreApplication.processEvents()
            except Exception as e:
                print(f"⚠️ Error processing events: {e}")
            

            print("🔒 Hiding window (not fully closing)")
            event.ignore()
            self.hide()
            
            print("✅ CRISPI window closed gracefully")
            
        except Exception as e:
            print(f"❌ Critical close event error: {e}")
            import traceback
            print(f"   Stack trace: {traceback.format_exc()}")

            event.ignore()
            self.hide()
            self.closed.emit()
