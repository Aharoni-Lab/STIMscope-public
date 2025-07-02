# gpu_ui.py
import os
import threading
import time
import pyqtgraph as pg
import numpy as np
import PyQt5.QtCore as QtCore
import cv2

from PyQt5.QtWidgets import (
    QGridLayout, QPushButton, QWidget, QTextEdit,
    QVBoxLayout, QFileDialog
)
from PyQt5.QtGui import QTextCursor
from PyQt5.QtCore import pyqtSignal
from live_trace_extractor import LiveTraceExtractor
from live_trace_napari import LiveTraceExtractorNapari
from make_mmap import make_memmap
from otsu_thresh import compute_mean_projection, denoise_and_threshold_gpu
from otsu_thresh import  compute_mean_projection, denoise_and_threshold_gpu
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5 import QtCore
from camera import Camera

class GPU(QWidget):
    newLogpyqtSignal = pyqtSignal(str)
    closed       = pyqtSignal()
    instance     = None
    log_buffer   = []
    export_count = 0
    refineRequested = pyqtSignal(object, object)
    requestStartLiveTraces = pyqtSignal()
    requestStartRecording   = pyqtSignal()
    requestStartLiveTracesNapari = pyqtSignal()

    def __init__(self, camera: Camera, logger=None, log_widget=None):
        super().__init__()
        if camera is None:
            raise ValueError("GPU needs a Camera instance")
        self.camera = camera
        GPU.instance = self
        self.setWindowTitle("CRISPI")
        self.resize(700, 500)
        self.requestStartLiveTraces.connect(self.start_live_traces, QtCore.Qt.QueuedConnection)
        self.requestStartLiveTracesNapari.connect(self.start_live_traces_napari, QtCore.Qt.QueuedConnection)
        self.requestStartRecording.connect(self.camera.start_recording, QtCore.Qt.QueuedConnection)
        self.refineRequested.connect(self._launch_napari_viewer)

        if hasattr(GPU.instance, "roiExported"):
            GPU.instance.roiExported.connect(self.on_roi_exported)
        # layout & log widget
        self.layout = QVBoxLayout(self)
        self.log_widget = log_widget or QTextEdit()
        self.log_widget.setReadOnly(True)
        self.layout.addWidget(self.log_widget)
        self.newLogpyqtSignal.connect(self.write_log_pyqtSlot)
        self.paused = False
        # pipeline state
        self.video_path   = None
        self.proj_display = None
        self.memmap_path  = "movie_mmap.npy"
        self.rois_path    = "rois.npz"
        self.trace_path   = "traces_live.npy"
        self._discover_method = "OTSU" 

        # storage for our extractor
        self.live_extractor = None
        self.live_extractor_napari = None
        # add pause/export controls
        self.log_init()

        # add our pipeline‐button row
        self.init_pipeline_buttons()

        # flush any early logs
        for msg in GPU.log_buffer:
            self.newLogpyqtSignal.emit(msg)
        GPU.log_buffer.clear()

    @pyqtSlot(object)
    def on_roi_exported(self, label_data):
        try:
            GPU.log_INFO("Received new ROIs from Napari export. Re-initializing live traces.")
            if self.live_extractor_napari:
                self.live_extractor_napari.stop()
                self.live_extractor_napari = None
            self.live_extractor_napari = LiveTraceExtractorNapari(
                camera=self.camera,
                label_path=self.rois_path,
                max_points=300,
                use_pygame_plot=True
            )

        except Exception as e:
            GPU.log_ERRO(f"Failed to reinit live traces after export: {e}")



    def stop_live_traces(self):
        """Safely stop the live trace extractor if it's running."""
        if self.live_extractor:
            try:
                if hasattr(self.live_extractor, "stop"):
                    self.live_extractor.stop()
                else:
                    GPU.log_WARN("live_extractor has no stop() method.")
            except Exception as e:
                GPU.log_ERRO(f"Failed to stop live trace extractor: {e}")
            finally:
                self.live_extractor = None

    def stop_live_traces_napari(self):
        """Safely stop the live trace extractor if it's running."""
        if self.live_extractor_napari:
            try:
                if hasattr(self.live_extractor_napari, "stop"):
                    self.live_extractor_napari.stop()
                else:
                    GPU.log_WARN("live_extractor has no stop() method.")
            except Exception as e:
                GPU.log_ERRO(f"Failed to stop live trace extractor: {e}")
            finally:
                self.live_extractor_napari = None


    def init_pipeline_buttons(self):
        """Create buttons for each GPU‐pipeline step."""
        grid = QGridLayout()
        row = 0

        # 1) Select source video
        btn = QPushButton("🖼 Select Video…")
        btn.clicked.connect(self.select_video)
        grid.addWidget(btn, row, 0)

        # 2) Convert → memmap
        btn = QPushButton("➤ Make Memmap")
        btn.clicked.connect(self.run_make_memmap)
        grid.addWidget(btn, row, 1)

        from PyQt5.QtWidgets import QToolButton, QMenu, QAction

        # 3) Detect ROIs
        dd = QToolButton()
        dd.setText("➤ Discover Mask")
        dd.setPopupMode(QToolButton.InstantPopup)

        menu = QMenu(dd)

        for method in ("Cellpose", "CNMF", "Custom", "OTSU"):
            act = QAction(method, dd)
            # When the user picks “Suite2p” (etc.), we call run_discover_rois(method)
            act.triggered.connect(lambda checked=False, m=method: self.run_discover_rois(m))
            menu.addAction(act)

        dd.setMenu(menu)
        grid.addWidget(dd, row, 2)

        # 4) Refine curated ROIs
        btn = QPushButton("➤ Manual Mask Editor")
        btn.clicked.connect(self.run_refine_rois)
        grid.addWidget(btn, row, 3)

        # 5) View traces
        btn = QPushButton("▶ Export Traces")
        btn.clicked.connect(self.run_view_traces)
        grid.addWidget(btn, row, 5)

        self.layout.addLayout(grid)

    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select video file", "", "Video files (*.avi *.mp4 *.h5 *.npy *.npz)"
        )
        if path:
            self.video_path = path
            GPU.log_INFO(f"Selected video: {path}")

    # Generic pattern: spawn a thread, log start, run, log finish
    def run_make_memmap(self):
        threading.Thread(target=self._thread_make_memmap, daemon=True).start()

    def _thread_make_memmap(self):
        GPU.log_NOTI("Making memmap…")
        try:
            make_memmap(self.video_path, self.memmap_path)
            GPU.log_INFO(f"Memmap saved to {self.memmap_path}")
        except Exception as e:
            GPU.log_ERRO(f"Memmap failed: {e}")

    def run_discover_rois(self, method="OTSU"):
        self._discover_method = method
        threading.Thread(target=self._thread_discover_rois, daemon=True).start()

  

    def _thread_discover_rois(self):
        GPU.log_NOTI("Discovering ROIs…")
        self.stop_live_traces()
        self.stop_live_traces_napari()
        if self.camera.is_recording:
            self.camera.stop_recording()
            time.sleep(0.05)
            
        try:

            if self._discover_method == "OTSU":
                # Load memmap video
                movie = np.load(self.memmap_path, mmap_mode='r')
                self.discovered = compute_mean_projection(movie, calib_frames=5400, chunk_size=200)
            

                self.discovered = cv2.resize(self.discovered, (1936, 1096), interpolation=cv2.INTER_NEAREST)

                # Threshold and denoise to get masks
                masks, sizes = denoise_and_threshold_gpu(
                    self.discovered, gauss_ksize=(3,3), gauss_sigma=1.5,
                    min_area=60, max_area=300
                )
            

                labeled_image = np.zeros_like(masks[0], dtype=np.int32)
                for i, mask in enumerate(masks, start=1):
                    labeled_image[mask] = i


            elif self._discover_method == "Cellpose":
                # …call Suite2p’s pipeline on self.memmap_path…
                # labeled_image, masks, sizes = run_suite2p(self.memmap_path, …)
                raise NotImplementedError("Cellpose integration not yet implemented")

            elif self._discover_method == "CNMF":
                # …call CaImAn’s pipeline on self.memmap_path…
                raise NotImplementedError("CNMF integration not yet implemented")

            elif self._discover_method == "Custom":
                # …run your custom ROI detection code…
                raise NotImplementedError("Custom‐method not yet implemented")

            else:
                raise ValueError(f"Unknown ROI method: {self._discover_method}")
            # Save original discovered ROIs
           

            # ==== Project thresholded masks on STIMViewer ====
            from skimage.color import label2rgb
            from projection import ProjectDisplay
            from PyQt5.QtGui import QGuiApplication
            rgb_image = (label2rgb(labeled_image, bg_label=0) * 255).astype(np.uint8)
            screen = QGuiApplication.screens()[1]  # or [0] if only one
            size = screen.size()
            h, w = size.height(), size.width()
            rgb_image = cv2.resize(rgb_image, (w, h), interpolation=cv2.INTER_NEAREST)
            screens = QGuiApplication.screens()
            screen = screens[1] if len(screens) > 1 else screens[0]
            self.proj_display = ProjectDisplay(screen)
            self.proj_display.show_image_fullscreen_on_second_monitor(rgb_image, self.camera.translation_matrix)

            np.savez_compressed(self.rois_path, masks=masks, sizes=sizes, labels=labeled_image)
            GPU.log_INFO(f"ROIs written to {self.rois_path}")
      
            QtCore.QMetaObject.invokeMethod(self, "start_live_traces", QtCore.Qt.QueuedConnection)
            GPU.log_INFO("Live trace extraction requested after ROI discovery.")

            if not self.camera.is_recording:
                QtCore.QMetaObject.invokeMethod(self.camera, "start_recording", QtCore.Qt.QueuedConnection)
                GPU.log_INFO("Recording requested after ROI discovery.")

        except Exception as e:
            GPU.log_ERRO(f"ROI discovery failed: {e}")


    def run_refine_rois(self):
        threading.Thread(target=self._thread_refine_rois, daemon=True).start()
    
    @pyqtSlot()
    def start_live_traces(self):
        print("Camera acquisition_running:", self.camera.acquisition_running)

        if self.live_extractor is not None:
            GPU.log_NOTI("Live trace extractor already running.")
            return

        if not self.camera.acquisition_running:
            GPU.log_WARN("Camera acquisition is not running; attempting to start...")
            started = self.camera.start_realtime_acquisition()
            if not started:
                GPU.log_ERRO("Failed to start camera acquisition. Aborting live trace initialization.")
                return
            else:
                GPU.log_INFO("Camera acquisition started for live trace extraction.")

        roi_path = self.rois_path
        if not os.path.exists(roi_path):
            GPU.log_ERRO("No ROI file found. Run Discover or Manual Mask first.")
            return

        try:
            self.live_extractor = LiveTraceExtractor(
                camera=self.camera,
                label_path=roi_path,
                max_points=300,
                use_pygame_plot=True
            )
            GPU.log_INFO(f"Live trace extraction started using {os.path.basename(roi_path)}.")
        except Exception as e:
            GPU.log_ERRO(f"Failed to start live traces: {e}")

    @pyqtSlot()
    def start_live_traces_napari(self):
        print("Camera acquisition_running:", self.camera.acquisition_running)

        if self.live_extractor_napari is not None:
            GPU.log_NOTI("Live trace extractor already running.")
            return

        if not self.camera.acquisition_running:
            GPU.log_WARN("Camera acquisition is not running; attempting to start...")
            started = self.camera.start_realtime_acquisition()
            if not started:
                GPU.log_ERRO("Failed to start camera acquisition. Aborting live trace initialization.")
                return
            else:
                GPU.log_INFO("Camera acquisition started for live trace extraction.")

        roi_path = self.rois_path
        if not os.path.exists(roi_path):
            GPU.log_ERRO("No ROI file found. Run Discover or Refine ROIs first.")
            return

        try:
            self.live_extractor_napari = LiveTraceExtractorNapari(
                camera=self.camera,
                label_path=roi_path,
                max_points=300,
                use_pygame_plot=True
            )
            GPU.log_INFO(f"Live trace extraction started using {os.path.basename(roi_path)}.")
        except Exception as e:
            GPU.log_ERRO(f"Failed to start live traces: {e}")

    def _wait_until_camera_stops(self, timeout: float = 2.0):
        """
        Block (on the GUI thread) until camera.acquisition_running and camera.is_recording
        are both False, or until `timeout` seconds have elapsed.
        
        This polling ensures the camera’s internal event loop has actually stopped
        before we begin CPU/GPU work.
        """
        start = time.time()
        # First, tell the camera to stop streaming & recording:
        try:
            # It’s often best to stop acquisition before stopping recording,
            # but your Camera API may differ. If your camera stops recording
            # first, then streaming, feel free to swap these two lines.
            if self.camera.acquisition_running:
                self.camera.stop_realtime_acquisition()
            if self.camera.is_recording:
                self.camera.stop_recording()
        except Exception as e:
            GPU.log_WARN(f"Error while requesting camera stop: {e}")

        # Now poll until both flags are False or we exceed `timeout`.
        while True:
            still_streaming = getattr(self.camera, "acquisition_running", False)
            still_recording = getattr(self.camera, "is_recording", False)
            if not still_streaming and not still_recording:
                return True
            if (time.time() - start) > timeout:
                # Timed out waiting for the camera to become idle
                return False
            # Sleep very briefly so we don’t lock up the GUI entirely.
            # Qt will still process events between each short sleep.
            QtCore.QCoreApplication.processEvents()  # allow Qt to update/wheel
            time.sleep(0.02)

    def _thread_refine_rois(self):
        self.stop_live_traces_napari()
        GPU.log_NOTI("Manual Mask Generation…")
        try:
            # load the mean, masks, run your roi_editor logic 
            from otsu_thresh import load_movie, compute_mean_projection
            mean = compute_mean_projection(load_movie(self.video_path), calib_frames=5400)
            mean = cv2.resize(mean, (1936, 1096), interpolation=cv2.INTER_NEAREST)
            masks = np.load(self.rois_path)["masks"]
            
            self.refineRequested.emit(mean, masks)
           
        except Exception as e:
            GPU.log_ERRO(f"ROI refinement failed: {e}")


    @pyqtSlot(object, object)
    def _launch_napari_viewer(self, mean, masks):
        from roi_editor import refine_rois

        # 1) If we were recording, stop it now so the old VideoWriter is released
        if self.camera.is_recording:
            self.camera.stop_recording()
            GPU.log_INFO("Stopped recording before launching Napari.")
            # Small pause to ensure writer closed
            time.sleep(0.05)

        # 2) Stop any live‐trace extractor (PyGame or Napari mode) before pausing the camera
        self.stop_live_traces()           
        self.stop_live_traces_napari()    

        # 3) Pause camera & projector, so Napari can take over
        try:
            if self.proj_display:
                self.proj_display.close()
            self.camera.stop_realtime_acquisition()
            GPU.log_INFO("Paused camera acquisition and projection before Napari.")
        except Exception as e:
            GPU.log_WARN(f"Failed to pause components before Napari: {e}")

        # 4) Launch Napari ROI editor (blocking until Napari window closes)
        _, viewer = refine_rois(mean, masks, return_viewer=True)

        # 5) When Napari closes, execute restore_after_napari()
        def restore_after_napari(event=None):
            try:
                event.accept()

                # 5a) Re‐project the updated mask
                from skimage.color import label2rgb
                from projection import ProjectDisplay
                from PyQt5.QtGui import QGuiApplication

                labels = np.load("rois.npz")["labels"]
                rgb_image = (label2rgb(labels, bg_label=0) * 255).astype(np.uint8)

                screens = QGuiApplication.screens()
                screen = screens[1] if len(screens) > 1 else screens[0]
                h, w = screen.size().height(), screen.size().width()
                rgb_image = cv2.resize(rgb_image, (w, h), interpolation=cv2.INTER_NEAREST)

                # (Re‐)open the projector window
                if self.proj_display:
                    self.proj_display.close()
                self.proj_display = ProjectDisplay(screen)
                self.proj_display.show_image_fullscreen_on_second_monitor(
                    rgb_image, self.camera.translation_matrix
                )
                GPU.log_INFO("Mask re‐projected after Napari closed.")

                # 5b) Restart camera acquisition
                started = self.camera.start_realtime_acquisition()
                if not started:
                    GPU.log_ERRO("Failed to restart camera acquisition after Napari.")
                    return
                GPU.log_INFO("Camera acquisition restarted after Napari.")

                # 5c) Restart recording (new VideoWriter → no PTS conflict)
                self.camera.start_recording()
                GPU.log_INFO("Recording restarted after Napari (new writer).")

                # 5d) Spawn a fresh Pygame‐based LiveTraceExtractorNapari
                if self.live_extractor_napari:
                    self.live_extractor_napari.stop()
                    self.live_extractor_napari = None

                # Delay a few ms so that acquisition is fully up before extractor starts
                QtCore.QTimer.singleShot(250, self._spawn_pygame_extractor)

                GPU.log_INFO("Live‐trace extractor will restart shortly after Napari.")
            except Exception as e:
                GPU.log_ERRO(f"Failed to restore after Napari: {e}")

        viewer.window._qt_window.closeEvent = restore_after_napari


    def _spawn_pygame_extractor(self):
        # Double-check that camera is truly streaming:
        if not self.camera.acquisition_running:
            GPU.log_WARN("Camera not streaming yet; extractor will retry in 50 ms")
            QtCore.QTimer.singleShot(50, self._spawn_pygame_extractor)
            return

        # Now that camera is up, build the Pygame extractor:
        self.live_extractor = LiveTraceExtractor(
            camera=self.camera,
            label_path=self.rois_path,
            max_points=300,
            use_pygame_plot=True
        )
        GPU.log_INFO("LiveTraceExtractor (pygame) started after Napari closed.")

    def run_view_traces(self):

        if not self.live_extractor:
            GPU.log_ERRO("Live trace extractor is not running.")
            return
        try:
            self.live_extractor.export_traces("live_traces.npy")
        except Exception as e:
            GPU.log_ERRO(f"Trace view failed: {e}")


    def closeEvent(self, event):
        # Properly stop recording and acquisition
        if self.live_extractor:
            self.live_extractor.stop()
            self.live_extractor = None
        if self.live_extractor_napari:
            self.live_extractor_napari.stop()
            self.live_extractor_napari = None
        if self.camera:
            self.camera.stop_recording()
            self.camera.stop_realtime_acquisition()
        self.closed.emit()
        event.accept()  # Allow the window to actually close


    def log_init(self):
        self.pause_resume_button = QPushButton("Pause Logging")
        self.pause_resume_button.setCheckable(True)
        self.pause_resume_button.clicked.connect(self.pause_resume_logging)


        self.export = QPushButton("Export Logbook")
        self.export.clicked.connect(self.export_logbook_to_file)

        grid = QGridLayout()
        grid.addWidget(self.pause_resume_button, 2, 0, 2, 1)
        grid.addWidget(self.export, 2, 2, 2, 1)

        self.layout.addStretch()
        self.layout.addLayout(grid)

    def pause_resume_logging(self):
        if self.pause_resume_button.isChecked():
            self.paused = True
            self.pause_resume_button.setText("Resume Logging")
        else:
            self.paused = False
            self.pause_resume_button.setText("Pause Logging")


    def write_log_pyqtSlot(self, log):
        """
        Write the log to the log widget.
        Uses HTML formatting so that colored messages are rendered properly.
        """
        if not self.paused:
            self.log_widget.insertHtml(log)
            self.log_widget.moveCursor(QTextCursor.End)

    def write_log(self, log):
        self.newLogpyqtSignal.emit(log)


    @classmethod
    def _log_generic(cls, level, message):
        """
        Generic logging method using HTML formatting for colored log levels.
        Levels: EMER, ALRT, CRIT, ERRO, WARN, NOTI, INFO, DBUG.
        """
        level_config = {
            "EMER": ("EMERGENCY:", "red"),
            "ALRT": ("ALERT:", "orange"),
            "CRIT": ("CRITICAL:", "darkred"),
            "ERRO": ("ERROR:", "red"),
            "WARN": ("WARNING:", "goldenrod"),
            "NOTI": ("NOTIFICATION:", "blue"),
            "INFO": ("INFORMATIONAL:", "black"),
            "DBUG": ("DEBUG:", "gray"),
        }
        prefix, color = level_config.get(level, ("", "black"))
        # Build an HTML-formatted log message.
        html = f"<span style='color: {color};'><b>{prefix}</b></span> {message}<br>"
        if cls.instance:
            cls.instance.newLogpyqtSignal.emit(html)



    # Convenience methods for each log level:
    @classmethod
    def log_EMER(cls, message):
        cls._log_generic("EMER", message)

    @classmethod
    def log_ALRT(cls, message):
        cls._log_generic("ALRT", message)

    @classmethod
    def log_CRIT(cls, message):
        cls._log_generic("CRIT", message)

    @classmethod
    def log_ERRO(cls, message):
        cls._log_generic("ERRO", message)

    @classmethod
    def log_WARN(cls, message):
        cls._log_generic("WARN", message)

    @classmethod
    def log_NOTI(cls, message):
        cls._log_generic("NOTI", message)

    @classmethod
    def log_INFO(cls, message):
        cls._log_generic("INFO", message)

    @classmethod
    def log_DBUG(cls, message):
        cls._log_generic("DBUG", message)

    def export_logbook_to_file(self):
        """
        Export the log to a file.
        Each export is appended to the file with an export header and separator.
        """
        GPU.export_count += 1
        # Get all logs from the widget as plain text.
        log_text = self.log_widget.toPlainText()
        lines = log_text.splitlines()
        file_path = "export_log.txt"
        try:
            with open(file_path, "a") as f:
                f.write(f"Export: {GPU.export_count}\n")
                f.write("\n".join(lines))
                f.write("\n" + ("-" * 40) + "\n")
            self.write_log(f"<br><i>Log exported successfully to {file_path}</i><br>")
        except Exception as e:
            self.write_log(f"<br><i>Error exporting log: {str(e)}</i><br>")
        print("Logbook exported to file")