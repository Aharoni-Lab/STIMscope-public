import sys, time, gc, threading
from typing import Optional
import os
import cv2
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt, pyqtSlot as Slot
from PyQt5.QtGui import QGuiApplication, QPixmap
import numpy as np
from ids_peak import ids_peak
from camera import Camera

from PyQt5.QtWidgets import (
    QDialog, QLabel, QPushButton, QVBoxLayout, QWidget, QFrame, QSizePolicy
)
from pathlib import Path

ASSETS = (Path(__file__).resolve().parent / "Assets").resolve()


_GPU_AVAILABLE = True

class Interface(QtWidgets.QMainWindow):
  

    messagebox_pyqtSignal = QtCore.pyqtSignal(str, str)
    image_update_signal = QtCore.pyqtSignal(object)
    fps_update_signal = QtCore.pyqtSignal(float)
    sl_decode_done = QtCore.pyqtSignal(bool, str)
    from camera import Camera

    def __init__(self, cam_module: Optional[Camera] = None):

        from PyQt5.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        self._qt_instance = app

        super().__init__()  # only after app exists
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        QtWidgets.QApplication.setQuitOnLastWindowClosed(True)
        self._closing = False
        
        # (Reverted) Global modern styling disabled to restore default compact widgets

        if cam_module is None:
            
            self._camera = Camera(ids_peak.DeviceManager.Instance(), self)
        else:
            self._camera = cam_module


        from video_recorder import VideoRecorder

        def _notify_finalized(path: str):
            QtCore.QTimer.singleShot(0, lambda: QtWidgets.QMessageBox.information(
                self, "Recording Complete", f"Saved video:\n{path}"
            ))

        if not hasattr(self._camera, "video_recorder") or self._camera.video_recorder is None:
            self._camera.video_recorder = VideoRecorder(interface=self, on_finalized=_notify_finalized)

        # Default camera type (can be changed in GUI)
        self.selected_camera_type = "IDS_Peak"

        self.last_frame_time = time.time()
        self.gpu_ui = None
        
        self.gui_init()
       
        
        self._qt_instance.aboutToQuit.connect(self._close)
        try:
            self.sl_decode_done.connect(self._on_sl_decode_done, QtCore.Qt.QueuedConnection)
        except Exception:
            pass

        # No minimum size restriction - allow window to be resized freely
        self.setWindowTitle("STIMViewer-CRISPI")
        
        # Set window icon if available
        icon_path = self._findprinto()
        if icon_path:
            self.setWindowIcon(QtGui.QIcon(str(icon_path)))
    @staticmethod
    def _findprinto():
        candidates = [
            ASSETS / "stimviewer-load.png",
            ASSETS / "UI" / "stimviewer-load.png",
            ASSETS / "Images" / "stimviewer-load.png",
        ]
        for p in candidates:
            if p.exists():
                return p
        return None



    def gui_init(self):
        container = QWidget()

        self._layout = QVBoxLayout(container)
        self.setCentralWidget(container)
        from display import Display

        self.display = Display()
        # Let the display resize freely; fixed max can stress layout/paint
        # Keep a reasonable minimum, but no artificial maximum
        self.display.setMinimumSize(320, 240)
        self.display.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self._layout.addWidget(self.display)
        self.projection = None
        self._projection_active = False  # Track projection state
        self.acquisition_thread = None


        self._button_software_trigger = None
        self._button_start_hardware_acquisition = None
        self._hardware_status = False #False = Display Start, False = End
        self._recording_status = False #False = Display Start, False = End
        # External process handles (non-blocking)
        self._proc_i2c = None
        self._proc_masks = None
        self._proc_projector = None




        self._dropdown_pixel_format = None
        self._dropdown_trigger_line = None # Dropdown for hardware trigger line





        
        self._button_show_gpu_ui = None

        self.messagebox_pyqtSignal.connect(self.message)
        for sig, slot in (("recordingStarted", self._on_recording_started),
                          ("recordingStopped", self._on_recording_stopped),
                          ("autoStartRecording", self._on_auto_start_recording)):
            try:
                getattr(self._camera, sig).connect(slot)
            except Exception:
                pass

        self._frame_count = 0
        self._gain_label = None

        self._gain_slider = None




    def is_gui(self):
        return True
    
    def set_camera(self, cam_module):
        self._camera = cam_module
    
    def _set_compact_width_to_text(self, widget, extra_px: int = 24):
        try:
            fm = widget.fontMetrics()
            text = widget.currentText() if hasattr(widget, 'currentText') else widget.text()
            width = fm.horizontalAdvance(text) + extra_px
            if width > 0:
                widget.setFixedWidth(width)
        except Exception:
            pass
    

    def _create_button_bar(self):
       
        # Helper to force a widget width to match its current text
        def _set_compact_width_to_text(widget, extra_px: int = 24):
            try:
                fm = widget.fontMetrics()
                text = widget.currentText() if hasattr(widget, 'currentText') else widget.text()
                width = fm.horizontalAdvance(text) + extra_px
                if width > 0:
                    widget.setFixedWidth(width)
            except Exception:
                pass


        button_bar = QtWidgets.QWidget(self.centralWidget())
        button_bar_layout = QtWidgets.QGridLayout()


        self._button_start_hardware_acquisition = QtWidgets.QPushButton("Start Hardware Acquisition")
        self._button_start_hardware_acquisition.clicked.connect(self._start_hardware_acquisition)
        try:
            self._button_start_hardware_acquisition.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            self._set_compact_width_to_text(self._button_start_hardware_acquisition)
        except Exception:
            pass


        self._button_start_recording = QtWidgets.QPushButton("Start Recording")
        self._button_start_recording.clicked.connect(self._start_recording)

        # New: External control buttons
        self._button_start_projector = QtWidgets.QPushButton("Start Projection Engine")
        self._button_start_projector.clicked.connect(self._toggle_start_projector)
        self._seq_type_label = QtWidgets.QLabel("Sequence Type")
        self._seq_type_dropdown = QtWidgets.QComboBox()
        self._seq_type_dropdown.addItems(["8-bit Mono", "1-bit RGB"])  # maps to first byte of pattern_cfg
        try:
            self._seq_type_dropdown.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
            self._seq_type_dropdown.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        except Exception:
            pass
        self._button_toggle_overlay = QtWidgets.QPushButton("Enable Overlay")
        self._button_toggle_overlay.setCheckable(True)
        self._button_toggle_overlay.setChecked(True)
        self._button_toggle_overlay.toggled.connect(self._toggle_overlay)
        # Initialize label to current state
        try:
            self._toggle_overlay(self._button_toggle_overlay.isChecked())
        except Exception:
            pass
        self._button_req_hmatrix = QtWidgets.QPushButton("REQ H-Matrix")
        self._button_req_hmatrix.clicked.connect(self._send_hmatrix_to_projector)
        # Mask pattern selection UI
        self._mask_pattern_label = QtWidgets.QLabel("Mask Pattern")
        self._mask_pattern_dropdown = QtWidgets.QComboBox()
        self._mask_pattern_dropdown.addItems([
            "Moving Bar", "Checkerboard", "Solid", "Circle", "Gradient", "Image", "Folder", "Custom Python"
        ])
        self._mask_pattern_dropdown.currentTextChanged.connect(self._on_mask_pattern_changed)
        self._mask_pattern_browse = QtWidgets.QPushButton("Browse…")
        self._mask_pattern_browse.clicked.connect(self._browse_mask_pattern_path)
        self._mask_pattern_browse.setEnabled(False)
        self._mask_pattern_path = ""
        self._button_send_triggers = QtWidgets.QPushButton("Start Projector Trigger")
        self._button_send_triggers.clicked.connect(self._toggle_send_triggers)
        self._button_send_masks = QtWidgets.QPushButton("Send Masks")
        self._button_send_masks.clicked.connect(self._toggle_send_masks)




        
        self._button_show_gpu_ui = QtWidgets.QPushButton("Real-Time Trace")
        self._button_show_gpu_ui.clicked.connect(self.show_gpu_ui)
        self._button_show_gpu_ui.setEnabled(_GPU_AVAILABLE)
        



        self._dropdown_trigger_line = QtWidgets.QComboBox()
        self._label_trigger_line = QtWidgets.QLabel("Change Hardware Trigger Line:")



        self._dropdown_trigger_line.addItem("Line0")
        self._dropdown_trigger_line.addItem("Line1")   
        self._dropdown_trigger_line.addItem("Line2")
        self._dropdown_trigger_line.addItem("Line3")


        self._dropdown_trigger_line.currentIndexChanged.connect(self.change_hardware_trigger_line)
        # Make combo compact to fit content
        try:
            self._dropdown_trigger_line.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
            self._dropdown_trigger_line.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            self._dropdown_trigger_line.currentTextChanged.connect(lambda *_: _set_compact_width_to_text(self._dropdown_trigger_line, 36))
            _set_compact_width_to_text(self._dropdown_trigger_line, 36)
        except Exception:
            pass


        self._dropdown_pixel_format = QtWidgets.QComboBox()
        try:
            formats = self._camera.node_map.FindNode("PixelFormat").Entries()
        except Exception:
            formats = []

        
        na = getattr(ids_peak, "NodeAccessStatus_NotAvailable", None)
        ni = getattr(ids_peak, "NodeAccessStatus_NotImplemented", None)

        for idx in formats:
            try:
                acc = idx.AccessStatus()
                if (na is not None and acc == na) or (ni is not None and acc == ni):
                    continue
                if self._camera.conversion_supported(idx.Value()):
                    self._dropdown_pixel_format.addItem(idx.SymbolicValue())
            except Exception:

                continue
        self._dropdown_pixel_format.currentIndexChanged.connect(self.change_pixel_format)
        # Make combo compact to fit content
        try:
            self._dropdown_pixel_format.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
            self._dropdown_pixel_format.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            self._dropdown_pixel_format.currentTextChanged.connect(lambda *_: _set_compact_width_to_text(self._dropdown_pixel_format, 36))
            _set_compact_width_to_text(self._dropdown_pixel_format, 36)
        except Exception:
            pass


        self._dropdown_pixel_format.setEnabled(True)
        self._dropdown_trigger_line.setEnabled(True)



        

        self._button_software_trigger = QtWidgets.QPushButton("Snapshot")
        self._button_software_trigger.clicked.connect(self._trigger_sw_trigger)
        # Keep buttons compact
        try:
            self._button_software_trigger.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            _set_compact_width_to_text(self._button_software_trigger)
        except Exception:
            pass
        
        

        self._button_calibrate = QtWidgets.QPushButton("Calibrate")
        self._button_calibrate.clicked.connect(self._calibrate)
        try:
            self._button_calibrate.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            # a bit larger than text
            _set_compact_width_to_text(self._button_calibrate, 28)
        except Exception:
            pass

        # Structured-Light calibration & projection buttons
        self._button_sl_calibrate = QtWidgets.QPushButton("Structured-Light Calibrate")
        self._button_sl_calibrate.clicked.connect(self._sl_calibrate)
        try:
            self._button_sl_calibrate.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            _set_compact_width_to_text(self._button_sl_calibrate, 28)
        except Exception:
            pass
        self._button_sl_project_reg = QtWidgets.QPushButton("Project LUT-Warped Registration")
        self._button_sl_project_reg.clicked.connect(self._sl_project_registration)
        try:
            self._button_sl_project_reg.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            _set_compact_width_to_text(self._button_sl_project_reg, 28)
        except Exception:
            pass

        # Project intensity controls
        self._project_intensity_label = QtWidgets.QLabel("Project Intensity")
        self._project_intensity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._project_intensity_slider.setRange(0, 255)
        self._project_intensity_slider.setValue(255)
        self._project_intensity_slider.setSingleStep(1)
        self._project_intensity_slider.setMaximumWidth(150)  # Make slider shorter
        self._project_intensity_slider.valueChanged.connect(self._update_project_intensity)
        
        self._project_intensity_value_label = QtWidgets.QLabel("255")
        self._project_intensity_value_label.setMinimumWidth(30)
        self._project_intensity_value_label.setAlignment(QtCore.Qt.AlignCenter)
        
        self._button_project_on = QtWidgets.QPushButton("Project ON")
        self._button_project_on.clicked.connect(self._project_on)
        
        self._button_project_off = QtWidgets.QPushButton("Project OFF")
        self._button_project_off.clicked.connect(self._project_off)

        # Camera type selection
        self._camera_type_label = QtWidgets.QLabel("Camera Type")
        self.camera_type_dropdown = QtWidgets.QComboBox()
        self.camera_type_dropdown.addItems(["IDS_Peak", "MIPI", "Generic Camera"])
        self.camera_type_dropdown.setCurrentText(self.selected_camera_type)
        self.camera_type_dropdown.currentTextChanged.connect(self._on_camera_type_changed)
        # Make combo compact to fit content
        try:
            self.camera_type_dropdown.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
            self.camera_type_dropdown.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            self.camera_type_dropdown.currentTextChanged.connect(lambda *_: _set_compact_width_to_text(self.camera_type_dropdown, 36))
            _set_compact_width_to_text(self.camera_type_dropdown, 36)
        except Exception:
            pass

        self._gain_label = QtWidgets.QLabel("AG")
        self._gain_label.setMaximumWidth(70)

        self._gain_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical)
        self._gain_slider.setRange(100, 1000)
        self._gain_slider.setSingleStep(1)
        self._gain_slider.valueChanged.connect(self._update_gain)

        

        self._dgain_label = QtWidgets.QLabel("DG")
        self._dgain_label.setMaximumWidth(70)

        self._dgain_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical)
        self._dgain_slider.setRange(100, 1000)
        self._dgain_slider.setSingleStep(1)
        self._dgain_slider.valueChanged.connect(self._update_dgain)


        # Zoom slider removed - using mouse wheel zoom instead



        config_group = QtWidgets.QGroupBox("")
        config_layout = QtWidgets.QGridLayout()
        config_layout.setSpacing(3)  # Reduce spacing
        try:
            config_layout.setHorizontalSpacing(2)  # Tighter space between top-row buttons
        except Exception:
            pass
        config_layout.setContentsMargins(6, 6, 6, 6)  # Reduce margins
        config_group.setLayout(config_layout)
        config_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #d1d1d6;
                border-radius: 6px;
                margin-top: 2px;
                font-weight: 500;
                font-size: 11px;
                color: #1c1c1e;
                background-color: #ffffff;
                padding: 4px;
            }
        """)


        # Row 0: Main action buttons (tightly packed, left-aligned)
        row0_layout = QtWidgets.QHBoxLayout()
        row0_layout.setContentsMargins(0, 0, 0, 0)
        row0_layout.setSpacing(4)
        row0_layout.addWidget(self._button_start_hardware_acquisition)
        row0_layout.addWidget(self._button_calibrate)
        row0_layout.addWidget(self._button_sl_calibrate)
        row0_layout.addWidget(self._button_sl_project_reg)
        row0_widget = QtWidgets.QWidget()
        row0_widget.setLayout(row0_layout)
        config_layout.addWidget(row0_widget,                             0, 0, 1, 2)
        # Row 1: Projection engine and trigger controls
        row1_layout = QtWidgets.QHBoxLayout()
        row1_layout.addWidget(self._button_start_projector)
        row1_layout.addWidget(self._seq_type_label)
        row1_layout.addWidget(self._seq_type_dropdown)
        row1_layout.addWidget(self._button_toggle_overlay)
        row1_layout.addWidget(self._button_req_hmatrix)
        row1_widget = QtWidgets.QWidget()
        row1_widget.setLayout(row1_layout)
        config_layout.addWidget(row1_widget,                             1, 0, 1, 2)
        
        # New Row 2: mask pattern selection and send controls
        row2_layout = QtWidgets.QHBoxLayout()
        row2_layout.addWidget(self._mask_pattern_label)
        row2_layout.addWidget(self._mask_pattern_dropdown)
        row2_layout.addWidget(self._mask_pattern_browse)
        # Shift buttons left: replace stretch with a small spacing
        row2_layout.addSpacing(8)
        row2_layout.addWidget(self._button_send_triggers)
        row2_layout.addWidget(self._button_send_masks)
        row2_widget = QtWidgets.QWidget()
        row2_widget.setLayout(row2_layout)
        config_layout.addWidget(row2_widget,                             2, 0, 1, 2)
        
        # Row 3: Project ON/OFF buttons
        project_buttons_layout = QtWidgets.QHBoxLayout()
        project_buttons_layout.addWidget(self._button_project_on)
        project_buttons_layout.addWidget(self._button_project_off)
        project_buttons_layout.addSpacing(12)
        project_buttons_layout.addWidget(self._project_intensity_label)
        project_buttons_layout.addWidget(self._project_intensity_slider)
        project_buttons_layout.addWidget(self._project_intensity_value_label)
        project_buttons_layout.addStretch()
        project_buttons_widget = QtWidgets.QWidget()
        project_buttons_widget.setLayout(project_buttons_layout)
        config_layout.addWidget(project_buttons_widget,                  3, 0, 1, 2)
        
        # Row 4: Combine Trigger Line, Camera Type, and Camera Format in one row
        self._camera_format_label = QtWidgets.QLabel("Camera Format")
        row_cam_all = QtWidgets.QHBoxLayout()
        row_cam_all.setContentsMargins(0, 0, 0, 0)
        row_cam_all.setSpacing(6)
        row_cam_all.addWidget(self._label_trigger_line)
        row_cam_all.addWidget(self._dropdown_trigger_line)
        row_cam_all.addSpacing(12)
        row_cam_all.addWidget(self._camera_type_label)
        row_cam_all.addWidget(self.camera_type_dropdown)
        row_cam_all.addSpacing(12)
        row_cam_all.addWidget(self._camera_format_label)
        row_cam_all.addWidget(self._dropdown_pixel_format)
        row_cam_all_widget = QtWidgets.QWidget()
        row_cam_all_widget.setLayout(row_cam_all)
        config_layout.addWidget(row_cam_all_widget,                      4, 0, 1, 2, Qt.AlignLeft)


        capture_group = QtWidgets.QGroupBox("")
        capture_layout = QtWidgets.QGridLayout()
        capture_layout.setSpacing(3)  # Reduce spacing
        capture_layout.setContentsMargins(6, 6, 6, 6)  # Reduce margins
        capture_group.setLayout(capture_layout)
        capture_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #d1d1d6;
                border-radius: 6px;
                margin-top: 2px;
                font-weight: 500;
                font-size: 11px;
                color: #1c1c1e;
                background-color: #ffffff;
                padding: 4px;
            }
        """)


        capture_layout.addWidget(self._button_start_recording, 0, 0)
        capture_layout.addWidget(self._button_software_trigger, 0, 1)
        # Keep Start Recording compact and responsive to text changes
        try:
            self._button_start_recording.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            _set_compact_width_to_text(self._button_start_recording)
        except Exception:
            pass
        # Pixel format moved under Camera Type below
        # Place Real-Time Trace on the same row
        capture_layout.addWidget(self._button_show_gpu_ui, 0, 2)


        control_group = QtWidgets.QGroupBox("")
        control_group_layout = QtWidgets.QGridLayout()
        control_group_layout.setSpacing(2)  # Reduce spacing for sliders
        control_group_layout.setContentsMargins(4, 4, 4, 4)  # Reduce margins
        control_group.setLayout(control_group_layout)
        control_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #d1d1d6;
                border-radius: 6px;
                margin-top: 2px;
                font-weight: 500;
                font-size: 11px;
                color: #1c1c1e;
                background-color: #ffffff;
                padding: 4px;
            }
        """)


        self._gain_label.setAlignment(Qt.AlignCenter)
        self._gain_slider.setFixedWidth(15)  # Make narrower
        control_group_layout.addWidget(self._gain_label, 0, 0)
        control_group_layout.addWidget(self._gain_slider, 1, 0)
        self._gain_value_label = QtWidgets.QLabel("1.00")
        self._gain_value_label.setAlignment(Qt.AlignCenter)
        self._gain_value_label.setStyleSheet("font-size: 10px;")
        control_group_layout.addWidget(self._gain_value_label, 2, 0)


        self._dgain_label.setAlignment(Qt.AlignCenter)
        self._dgain_slider.setFixedWidth(15)  # Make narrower
        control_group_layout.addWidget(self._dgain_label, 0, 1)
        control_group_layout.addWidget(self._dgain_slider, 1, 1)
        self._dgain_value_label = QtWidgets.QLabel("1.00")
        self._dgain_value_label.setAlignment(Qt.AlignCenter)
        self._dgain_value_label.setStyleSheet("font-size: 10px;")
        control_group_layout.addWidget(self._dgain_value_label, 2, 1)


        # Zoom controls removed - using mouse wheel zoom instead


        # Set control panel widths for larger buttons
        control_group.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed,
            QtWidgets.QSizePolicy.Preferred
        )
        control_group.setFixedWidth(100)  # Sliders panel
        
        for grp in (config_group, capture_group):
            grp.setSizePolicy(
                QtWidgets.QSizePolicy.Preferred,
                QtWidgets.QSizePolicy.Preferred
            )


        # Remove stretching for more compact layout
        button_bar_layout.setColumnStretch(4, 0)  # No stretching for compact panels
        button_bar_layout.setColumnStretch(5, 0)  # No stretching for sliders
        button_bar_layout.setColumnStretch(6, 0)
        button_bar_layout.setColumnStretch(7, 0)

        # Shift everything to the left to align with video preview
        button_bar_layout.addWidget(config_group, 0, 0, 4, 1)       # Column 0 (leftmost)
        button_bar_layout.addWidget(capture_group, 4, 0, 2, 1)      # Column 0, below config
        button_bar_layout.addWidget(control_group, 0, 1, 7, 1)      # Column 1 (next to config/capture)
        
        # Add spacer to push everything to the left
        spacer = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        button_bar_layout.addItem(spacer, 0, 2, 7, 1)  # Column 2, fill remaining space



        self._button_start_hardware_acquisition.setToolTip("Start/Stop acquiring images using hardware triggering rather than real time(RT) acquisition. Hardware Trigger FPS must stay <45 hz")
        self._button_start_recording.setToolTip("Start/Stop recording video of the live feed.")
        self._button_software_trigger.setToolTip("Save the next processed frame.")
        self._button_send_triggers.setToolTip("Start/Stop sending projector triggers over I2C.")
        self._button_send_masks.setToolTip("Start/Stop sending masks over ZMQ to the projector.")
        self._button_start_projector.setToolTip("Start/Stop the projection engine binary with configured options.")


        self._gain_label.setToolTip("Adjust the analog gain level (brightness).")
        self._dgain_label.setToolTip("Adjust the digital gain level.")
        # Zoom tooltip removed - using mouse wheel zoom instead


        button_bar.setLayout(button_bar_layout)
        self._layout.addWidget(button_bar)

        # SL progress widgets are created in _create_statusbar so they sit on the status bar row
        self._sl_progress = None
        self._sl_status = None

    def _ensure_qprocess(self):
        # Lazy import to avoid startup penalty if unused
        from PyQt5.QtCore import QProcess
        return QProcess

    def _maybe_build_projector(self, proj_dir: str) -> bool:
        try:
            import os, subprocess
            exe = f"{proj_dir}/projector"
            src = f"{proj_dir}/main.cpp"
            need_build = (not os.path.exists(exe))
            if not need_build:
                try:
                    need_build = os.path.getmtime(exe) < os.path.getmtime(src)
                except Exception:
                    need_build = False
            if not need_build:
                return True
            print(f"[PROJ] Building projector in {proj_dir} ...")
            cmd = [
                "g++", "-O2", "-std=c++17", "main.cpp", "-o", "projector",
                "-lglfw", "-lGL", "-lzmq", "-lgpiod", "-lpthread"
            ]
            res = subprocess.run(cmd, cwd=proj_dir, capture_output=True, text=True)
            if res.returncode != 0:
                print("[PROJ] Build failed:\n" + (res.stderr or res.stdout))
                return False
            print("[PROJ] Build succeeded")
            return True
        except Exception as e:
            print(f"[PROJ] Build error: {e}")
            return False

    def _helper_python_path_for_masks(self) -> str:
        # Prefer local venv (contains pyzmq), then active conda, then current python
        try:
            venv_py = (Path(__file__).resolve().parents[1] / "my_UARTvenv" / "bin" / "python").resolve()
            if venv_py.exists():
                return str(venv_py)
        except Exception:
            pass
        try:
            conda_pref = os.environ.get("CONDA_PREFIX")
            if conda_pref:
                cand = Path(conda_pref) / "bin" / "python"
                if cand.exists():
                    return str(cand)
        except Exception:
            pass
        return sys.executable or "/usr/bin/python3"

    def _on_mask_pattern_changed(self, text: str):
        # Enable browse button only for patterns that need a path
        need_path = text in ("Image", "Folder", "Custom Python")
        try:
            self._mask_pattern_browse.setEnabled(need_path)
        except Exception:
            pass

    def _browse_mask_pattern_path(self):
        try:
            from PyQt5.QtWidgets import QFileDialog
            typ = self._mask_pattern_dropdown.currentText()
            if typ == "Image":
                fp, _ = QFileDialog.getOpenFileName(self, "Select Image", str(Path.home()),
                                                    "Images (*.png *.jpg *.jpeg *.bmp)")
                if fp:
                    self._mask_pattern_path = fp
            elif typ == "Folder":
                dirp = QFileDialog.getExistingDirectory(self, "Select Folder", str(Path.home()))
                if dirp:
                    self._mask_pattern_path = dirp
            elif typ == "Custom Python":
                fp, _ = QFileDialog.getOpenFileName(self, "Select Python Script", str(Path.home()),
                                                    "Python (*.py)")
                if fp:
                    self._mask_pattern_path = fp
        except Exception as e:
            print(f"Browse failed: {e}")

    def _helper_python_path_for_i2c(self) -> str:
        """Pick Python for I2C (prefer system where smbus2 is typically available)."""
        for cand in ("/usr/bin/python3", "/usr/local/bin/python3", sys.executable):
            try:
                if os.path.exists(cand):
                    return cand
            except Exception:
                continue
        return sys.executable

    def _attach_proc_signals(self, proc, which: str):
        try:
            from PyQt5.QtCore import QProcess
            proc.setProcessChannelMode(QProcess.MergedChannels)
            proc.readyReadStandardOutput.connect(lambda: self._on_proc_output(proc, which))
        except Exception:
            pass

    def _on_proc_output(self, proc, which: str):
        try:
            data = bytes(proc.readAllStandardOutput()).decode(errors='ignore')
            if data:
                if which == 'i2c':
                    prefix = "[I2C]"
                elif which == 'masks':
                    prefix = "[MASK]"
                else:
                    prefix = "[PROJ]"
                print(f"{prefix} {data.rstrip()}")
        except Exception:
            pass

    def _toggle_send_triggers(self):
        QProcess = self._ensure_qprocess()
        try:
            # Run exact script and capture output/errors in console
            work_dir = str(Path(__file__).resolve().parents[1])
            # Use absolute path explicitly to avoid any ambiguity
            script_path = "/home/aharonilabjetson2/Desktop/MyScripts/MyUART/i2c_test_send_commands.py"
            py = "/usr/bin/python3"

            self._proc_i2c = QProcess(self)
            self._proc_i2c.setWorkingDirectory(work_dir)
            self._attach_proc_signals(self._proc_i2c, 'i2c')
            self._proc_i2c.finished.connect(lambda *_: self._on_proc_finished('i2c'))
            self._proc_i2c.errorOccurred.connect(lambda *_: self._on_proc_finished('i2c'))

            try:
                from PyQt5.QtCore import QProcessEnvironment
                env = QProcessEnvironment.systemEnvironment()
                env.insert("PYTHONUNBUFFERED", "1")
                # Keep a clean PATH so /usr/bin/python3 resolves stable libs
                if not env.contains("PATH"):
                    env.insert("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
                self._proc_i2c.setProcessEnvironment(env)
            except Exception:
                pass

            # Map sequence type to first byte of pattern_cfg (0x02=8-bit Mono, 0x01=1-bit RGB)
            seq_first = "0x02" if (self._seq_type_dropdown.currentText() == "8-bit Mono") else "0x01"
            print(f"[I2C] Launch: {py} {script_path} --seq-first {seq_first}")
            self._proc_i2c.start(py, [script_path, "--seq-first", seq_first])
        except Exception as e:
            print(f"Failed to start I2C trigger script: {e}")
            self._on_proc_finished('i2c')

    def _toggle_start_projector(self):
        QProcess = self._ensure_qprocess()
        try:
            if self._proc_projector is None:
                self._proc_projector = QProcess(self)
                self._proc_projector.finished.connect(lambda *_: self._on_proc_finished('projector'))
                self._proc_projector.errorOccurred.connect(lambda *_: self._on_proc_finished('projector'))
                self._attach_proc_signals(self._proc_projector, 'projector')
                try:
                    from PyQt5.QtCore import QProcessEnvironment
                    env = QProcessEnvironment.systemEnvironment()
                    env.insert("PYTHONUNBUFFERED", "1")
                    self._proc_projector.setProcessEnvironment(env)
                except Exception:
                    pass

                # Launch projector from exact local folder with your args
                proj_dir = "/home/aharonilabjetson2/Desktop/MyScripts/MyUART/ZMQ_sender_mask"
                # Ensure latest binary is built before launch
                if not self._maybe_build_projector(proj_dir):
                    print("Failed to build projector; aborting launch")
                    self._on_proc_finished('projector')
                    return
                self._proc_projector.setWorkingDirectory(proj_dir)
                exe = f"{proj_dir}/projector"
                args = [
                    "--bind=tcp://*:5558",
                    "--swap-interval=1",
                    f"--visible-id={'1' if self._button_toggle_overlay.isChecked() else '0'}",
                    "--overlay-style=digits",
                    # Use projector defaults for size/position (compile-time or runtime)
                    "--overlay-bg=1",
                    "--overlay-bottom=mask",
                    "--overlay-top=proj",
                    "--cam-chip=/dev/gpiochip1",
                    "--cam-line=8",
                    "--cam-edge=rising",
                    "--proj-chip=/dev/gpiochip1",
                    "--proj-line=9",
                    "--proj-edge=rising",
                    "--horiz-flip=1"
                ]
                print(f"[PROJ] Launch: {exe} {' '.join(args)}")
                self._button_start_projector.setText("Stop Projection Engine")
                self._proc_projector.start(exe, args)
            else:
                self._proc_projector.kill()
        except Exception as e:
            print(f"Failed to toggle projector: {e}")
            self._on_proc_finished('projector')

    def _toggle_send_masks(self):
        QProcess = self._ensure_qprocess()
        try:
            if self._proc_masks is None:
                self._proc_masks = QProcess(self)
                self._proc_masks.finished.connect(lambda *_: self._on_proc_finished('masks'))
                self._proc_masks.errorOccurred.connect(lambda *_: self._on_proc_finished('masks'))
                self._attach_proc_signals(self._proc_masks, 'masks')
                self._button_send_masks.setText("Stop Sending Masks")

                work_dir = str(Path(__file__).resolve().parents[1])
                self._proc_masks.setWorkingDirectory(work_dir)
                py = self._helper_python_path_for_masks()
                # Resolve sender script according to dropdown
                script_path = "/home/aharonilabjetson2/Desktop/MyScripts/MyUART/ZMQ_sender_mask/zmq_mask_sender.py"
                args = []
                pat = self._mask_pattern_dropdown.currentText()
                if pat == "Moving Bar":
                    args = []  # defaults
                elif pat == "Checkerboard":
                    args = ["--pattern", "checkerboard"]
                elif pat == "Solid":
                    args = ["--pattern", "solid"]
                elif pat == "Circle":
                    args = ["--pattern", "circle"]
                elif pat == "Gradient":
                    # Use sane defaults for visibility (60 Hz, 6 steps, 20-frame holds, gamma 2.2)
                    args = [
                        "--pattern", "gradient",
                        "--fps", "60",
                        "--gradient-steps", "6",
                        "--gradient-hold", "20",
                        "--gradient-gamma", "2.2"
                    ]
                elif pat == "Image":
                    args = ["--pattern", "image", "--image", self._mask_pattern_path]
                elif pat == "Folder":
                    args = ["--pattern", "folder", "--folder", self._mask_pattern_path]
                elif pat == "Custom Python":
                    script_path = self._mask_pattern_path or script_path
                    args = []

                try:
                    from PyQt5.QtCore import QProcessEnvironment
                    env = QProcessEnvironment.systemEnvironment()
                    env.insert("PYTHONUNBUFFERED", "1")
                    self._proc_masks.setProcessEnvironment(env)
                except Exception:
                    pass

                cmd = [script_path] + args
                print(f"[MASK] Launch: {py} {' '.join(cmd)}")
                self._proc_masks.start(py, cmd)
            else:
                self._proc_masks.kill()
        except Exception as e:
            print(f"Failed to toggle masks: {e}")
            self._on_proc_finished('masks')

    def _on_proc_finished(self, which: str):
        if which == 'i2c':
            try:
                if self._proc_i2c is not None:
                    self._proc_i2c.deleteLater()
            except Exception:
                pass
            self._proc_i2c = None
            if hasattr(self, '_button_send_triggers') and self._button_send_triggers is not None:
                self._button_send_triggers.setText("Send Projection Triggers")
        else:
            if which == 'masks':
                try:
                    if self._proc_masks is not None:
                        self._proc_masks.deleteLater()
                except Exception:
                    pass
                self._proc_masks = None
                if hasattr(self, '_button_send_masks') and self._button_send_masks is not None:
                    self._button_send_masks.setText("Send Masks")
            elif which == 'projector':
                try:
                    if self._proc_projector is not None:
                        self._proc_projector.deleteLater()
                except Exception:
                    pass
                self._proc_projector = None
                if hasattr(self, '_button_start_projector') and self._button_start_projector is not None:
                    self._button_start_projector.setText("Start Projection Engine")

    def _terminate_external_processes(self):
        # Ensure spawned helper scripts are stopped when GUI closes
        try:
            if self._proc_i2c is not None:
                try:
                    self._proc_i2c.kill()
                except Exception:
                    pass
                try:
                    self._proc_i2c.waitForFinished(1000)
                except Exception:
                    pass
        finally:
            self._proc_i2c = None
            try:
                if hasattr(self, '_button_send_triggers') and self._button_send_triggers is not None:
                    self._button_send_triggers.setText("Send Projection Triggers")
            except Exception:
                pass

        try:
            if self._proc_masks is not None:
                try:
                    self._proc_masks.kill()
                except Exception:
                    pass
                try:
                    self._proc_masks.waitForFinished(1000)
                except Exception:
                    pass
        finally:
            self._proc_masks = None
            try:
                if hasattr(self, '_button_send_masks') and self._button_send_masks is not None:
                    self._button_send_masks.setText("Send Masks")
            except Exception:
                pass

        try:
            if self._proc_projector is not None:
                try:
                    self._proc_projector.kill()
                except Exception:
                    pass
                try:
                    self._proc_projector.waitForFinished(2000)
                except Exception:
                    pass
        finally:
            self._proc_projector = None
            try:
                if hasattr(self, '_button_start_projector') and self._button_start_projector is not None:
                    self._button_start_projector.setText("Start Projection Engine")
            except Exception:
                pass

    def _create_statusbar(self):
       
        status_bar = QtWidgets.QWidget(self.centralWidget())
        status_bar.setMaximumHeight(30)
        try:
            status_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        except Exception:
            pass
        status_bar_layout = QtWidgets.QHBoxLayout()
        status_bar_layout.setContentsMargins(5, 2, 5, 2)  # Smaller margins


        separator = QFrame(self)
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._layout.addWidget(separator)


        self.acq_label = QLabel("Acquisition Mode: RealTime", self)
        self.acq_label.setStyleSheet("font-size: 11px; color: #1c1c1e;")
        self.acq_label.setAlignment(Qt.AlignLeft)
        self.acq_label.setToolTip("Current Acquisition Mode")

        # Projector status
        screens = QGuiApplication.screens()
        self.projector_status_label = QLabel(self)
        if len(screens) > 1:
            self.projector_status_label.setText("✅ Projector Connected")
            self.projector_status_label.setStyleSheet("font-size: 11px; color: #27ae60;")
        else:
            self.projector_status_label.setText("❌ No Projector Found")
            self.projector_status_label.setStyleSheet("font-size: 11px; color: #e74c3c;")
        self.projector_status_label.setAlignment(Qt.AlignCenter)
        self.projector_status_label.setToolTip("Projector connection status")

        self.GUIfps_label = QLabel("GUI FPS: 0.00", self)
        self.GUIfps_label.setStyleSheet("font-size: 11px; color: #1c1c1e;")
        self.GUIfps_label.setAlignment(Qt.AlignRight)
        self.GUIfps_label.setToolTip("Calculated FPS over a rolling average of 2 seconds. If set to hardware trigger mode, camera only supports <45 fps.")
        try:
            self.fps_update_signal.connect(self._set_gui_fps, QtCore.Qt.QueuedConnection)
        except Exception:
            pass
        # SL progress widgets in status row
        try:
            self._sl_progress = QtWidgets.QProgressBar(self)
            self._sl_progress.setRange(0, 0)  # indeterminate by default
            self._sl_progress.setVisible(False)
            self._sl_progress.setMaximumWidth(160)
            self._sl_status = QLabel("", self)
            self._sl_status.setStyleSheet("font-size: 11px; color: #1c1c1e;")
        except Exception:
            self._sl_progress = None
            self._sl_status = None

        status_bar_layout.addWidget(self.acq_label)
        status_bar_layout.addSpacing(12)
        status_bar_layout.addWidget(self.projector_status_label)
        status_bar_layout.addSpacing(12)
        if getattr(self, '_sl_progress', None):
            status_bar_layout.addWidget(self._sl_progress)
        if getattr(self, '_sl_status', None):
            status_bar_layout.addWidget(self._sl_status)
        # Push FPS all the way to the right
        status_bar_layout.addStretch(1)
        status_bar_layout.addWidget(self.GUIfps_label)

        status_bar.setLayout(status_bar_layout)
        self._layout.addWidget(status_bar)

    @QtCore.pyqtSlot(float)
    def _set_gui_fps(self, fps: float):
        try:
            self.GUIfps_label.setText(f"GUI FPS: {fps:.2f}")
        except Exception:
            pass

    def _close(self):
        try:
            # Stop helper processes first
            try:
                self._terminate_external_processes()
            except Exception:
                pass
            self._camera.shutdown()
        except Exception:
            pass

    @QtCore.pyqtSlot(bool, str)
    def _on_sl_decode_done(self, ok: bool, msg: str):
        try:
            if getattr(self, '_sl_progress', None):
                self._sl_progress.setVisible(False)
            if getattr(self, '_sl_status', None):
                self._sl_status.setText("✅ SL ready" if ok else f"❌ SL failed: {msg}")
            if hasattr(self, '_button_sl_project_reg') and self._button_sl_project_reg is not None:
                self._button_sl_project_reg.setEnabled(ok)
        except Exception:
            pass

    def closeEvent(self, event):
        try:

            try: self._camera.shutdown()
            except Exception: pass


            try:
                if hasattr(self._camera, "frame_ready"):
                    self._camera.frame_ready.disconnect(self.on_image_received)
                if hasattr(self._camera, "image_ready"):
                    self._camera.image_ready.disconnect(self.on_image_received)
                iface = getattr(self._camera, "_interface", None)
                if iface is not None and hasattr(iface, "frame_ready"):
                    iface.frame_ready.disconnect(self.on_image_received)
            except Exception:
                pass

            if self.projection is not None:
                try: self.projection.close()
                except Exception: pass
            try:
                self._terminate_external_processes()
            except Exception:
                pass
        finally:
            event.accept()

           

    def start_window(self):
        connected = False
        candidate_names = ("frame_ready", "image_ready", "new_frame", "frame", "qsignal_frame", "qsignal_image")


        if getattr(self._camera, "_interface", None) is not self:
            for name in candidate_names:
                sig = getattr(self._camera, name, None)
                if sig is None:
                    continue
                try:
                    sig.connect(self.on_image_received, QtCore.Qt.QueuedConnection)
                    print(f"Connected camera signal: {name} → on_image_received")
                    connected = True
                    break
                except Exception:
                    pass
            if not connected:

                for setter in ("set_frame_callback", "set_image_callback"):
                    cb = getattr(self._camera, setter, None)
                    if callable(cb):
                        try:
                            cb(self.on_image_received)
                            print(f"Installed camera callback via {setter}()")
                            connected = True
                            break
                        except Exception:
                            pass

        if not connected:

            iface = getattr(self._camera, "_interface", None)
            if iface is not None:
                for name in candidate_names:
                    sig = getattr(iface, name, None)
                    if sig is None:
                        continue
                    try:
                        sig.connect(self.on_image_received, QtCore.Qt.QueuedConnection)
                        print(f"Connected nested interface signal: {name}")
                        connected = True
                        break
                    except Exception:
                        pass

        if not connected:
            print("Could not connect any camera frame signal; preview will be blank.")
        else:
            print("Camera connected to UI.")

        self._create_button_bar()
        self._create_statusbar()

        try:
            self.image_update_signal.connect(self.display.on_image_received, QtCore.Qt.QueuedConnection)
            print("Bound image_update_signal → Display.on_image_received")
        except Exception as e1:
            print(f"Primary connect failed ({e1}); falling back to setImage alias")
            try:
                self.image_update_signal.connect(self.display.setImage, QtCore.Qt.QueuedConnection)
                print("Bound image_update_signal → Display.setImage")
            except Exception as e2:
                print(f"Display signal hookup failed: {e2}")
        # Delay creating the projector window until actually needed (calibration/projection)
        # This avoids early windowing/GL issues on some Jetson setups.
        self.projection = None

    def _ensure_projection(self):
        if self.projection is not None:
            return True
        try:
            from projection import ProjectDisplay
            screens = QGuiApplication.screens()
            if not screens:
                print("No screens available for projection")
                return False
            screen = screens[1] if len(screens) > 1 else screens[0]
            try:
                self.projection = ProjectDisplay(screen, parent=self)
            except TypeError:
                self.projection = ProjectDisplay(screen)
                self.projection.setParent(self)
            self.projection.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
            return True
        except Exception as e:
            print(f"Failed to create projection window: {e}")
            self.projection = None
            return False


    def _update_recording_button_text(self):
        """Update the recording button text based on current state"""
        is_recording = getattr(self._camera, "is_recording", False)
        is_armed = getattr(self._camera, "is_armed", False)
        
        print(f"🔍 Updating button text - recording: {is_recording}, armed: {is_armed}")
        
        if is_recording:
            self._button_start_recording.setText("Stop Recording")
        elif is_armed:
            self._button_start_recording.setText("Disarm Recording")
        else:
            self._button_start_recording.setText("Start Recording")

    @QtCore.pyqtSlot()
    def _on_recording_started(self):
        self._recording_status = True
        self._button_start_recording.setText("Stop Recording")
        self._button_start_hardware_acquisition.setEnabled(False)
        self._dropdown_trigger_line.setEnabled(False)

    @QtCore.pyqtSlot()
    def _on_recording_stopped(self):
        self._recording_status = False
        self._update_recording_button_text()
        self._button_start_hardware_acquisition.setEnabled(True)
        if not self._hardware_status:
            self._dropdown_trigger_line.setEnabled(True)

    @QtCore.pyqtSlot()
    def _on_auto_start_recording(self):
        """Handle automatic recording start from hardware trigger"""
        try:
            self._camera.start_recording()
        except Exception as e:
            print(f"Auto-start recording failed: {e}")
    
    def start_interface(self):
        self._gain_slider.setMaximum(int(self._camera.max_gain * 100))
        
        QtCore.QCoreApplication.setApplicationName("STIMViewer")
        self.show()
        self._qt_instance.exec()

    def _trigger_sw_trigger(self):
       
        try:
            if not self._camera:
                self.warning("No camera available for snapshot")
                return
            

            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp}.png"
            

            save_dir = getattr(self._camera, 'save_dir', './Saved_Media')
            os.makedirs(save_dir, exist_ok=True)
            filepath = os.path.join(save_dir, filename)
            

            if hasattr(self._camera, "snapshot"):
                success = self._camera.snapshot(filepath)
                if success:
                    self.information(f"Snapshot saved: {filename}")
                    print(f"✅ Snapshot saved: {filepath}")
                else:
                    self.warning("Snapshot failed - check camera status")
                    print("❌ Snapshot failed")
            elif hasattr(self._camera, "save_image"):
                self._camera.save_image = True
                print("📸 Legacy snapshot triggered")
            elif hasattr(self._camera, "software_trigger"):
                self._camera.software_trigger()
                print("📸 Software trigger sent")
            else:
                self.warning("No snapshot method available")
                print("❌ No snapshot method available")
                
        except Exception as e:
            error_msg = f"Snapshot error: {e}"
            self.warning(error_msg)
            print(f"❌ {error_msg}")


    def _start_hardware_acquisition(self):
        if not self._hardware_status:
            self._camera.stop_realtime_acquisition()
            self._camera.start_hardware_acquisition()
            
            try:
                node_map = self._camera.node_map
                mode_node = node_map.FindNode("TriggerMode")
                source_node = node_map.FindNode("TriggerSource")
                act_node = node_map.FindNode("TriggerActivation")
                
                print("TriggerMode =", mode_node.CurrentEntry().SymbolicValue() if mode_node else "None")
                print("TriggerSource =", source_node.CurrentEntry().SymbolicValue() if source_node else "None")
                print("TriggerActivation =", act_node.CurrentEntry().SymbolicValue() if act_node else "None")
            except Exception as e:
                print(f"Failed to read trigger nodes: {e}")

            self._dropdown_trigger_line.setEnabled(False)
            self.acq_label.setText("Acquisition Mode: Hardware")
            self._button_start_hardware_acquisition.setText("Stop Hardware Acquisition")
            # Reset armed state and update button text for hardware mode
            if hasattr(self._camera, 'is_armed'):
                self._camera.is_armed = False
            self._update_recording_button_text()
        else:
            # Disarm if armed when stopping hardware acquisition
            if getattr(self._camera, "is_armed", False):
                self._camera.disarm_recording()
            
            self._camera.stop_hardware_acquisition()
            self._camera.start_realtime_acquisition()
            self.acq_label.setText("Acquisition Mode: RealTime")
            self._button_start_hardware_acquisition.setText("Start Hardware Acquisition")
            if not self._recording_status:
                self._dropdown_trigger_line.setEnabled(True)
            # Update recording button text for realtime mode
            self._update_recording_button_text()

        self._hardware_status = not self._hardware_status


    def _start_recording(self):
        try:
            if getattr(self._camera, "is_recording", False):
                # Currently recording, stop it
                self._camera.stop_recording()
            elif getattr(self._camera, "is_armed", False):
                # Currently armed, disarm it
                self._camera.disarm_recording()
                self._update_recording_button_text()
            else:
                # Not recording and not armed
                if self._hardware_status:
                    # In hardware mode, arm the system
                    if self._camera.arm_recording():
                        self._update_recording_button_text()
                else:
                    # In realtime mode, start recording directly
                    self._camera.start_recording()
        except Exception as e:
            print(f"Recording toggle failed: {e}")


    def _calibrate(self):
       
        if not self._ensure_projection():
            print("Calibration aborted: projection window unavailable.")
            return
        try:
            img_path = ASSETS / "Generated" / "custom_registration_image.png"
            scr = self.projection.windowHandle().screen() if self.projection.windowHandle() else None
            geo = scr.geometry() if scr else None
            target_w = geo.width() if geo else 1920
            target_h = geo.height() if geo else 1080

            # Always regenerate the calibration image to ensure latest ArUco markers are present
            regen_needed = True
            if img_path.exists():
                try:
                    probe = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    if probe is None:
                        regen_needed = True
                    else:
                        ph, pw = probe.shape[:2]
                        if pw != target_w or ph != target_h:
                            print(f"ℹ️ Registration size mismatch ({pw}x{ph}) != projector ({target_w}x{target_h}), regenerating...")
                            regen_needed = True
                except Exception:
                    regen_needed = True
            else:
                regen_needed = True

            if regen_needed:
                try:
                    from calibration import create_charuco_registration_image
                    create_charuco_registration_image(target_w, target_h)
                    print(f"✅ Custom registration image generated: {img_path}")
                except Exception as e:
                    print(f"Failed to generate registration image: {e}")

            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"Calibration image not readable: {img_path}")
                return

            self.projection.show_image_fullscreen_on_second_monitor(
                img,
                getattr(self._camera, "translation_matrix", None)
            )
            print("projectionnnnnn")


            # Allow time for projector to refresh and camera to capture a few frames
            QtCore.QTimer.singleShot(250, lambda: getattr(self._camera, "start_calibration", lambda: None)())
        except Exception as e:
            print(f"Calibration start failed: {e}")

    
    def _update_project_intensity(self):
        """Update the intensity value label when slider changes."""
        intensity = self._project_intensity_slider.value()
        self._project_intensity_value_label.setText(str(intensity))
        
        # If projection is currently on, update it with new intensity
        if hasattr(self, '_projection_active') and self._projection_active:
            self._project_with_intensity(intensity)
    
    def _project_on(self):
        """Turn on projection with current intensity setting."""
        try:
            if not self._ensure_projection():
                print("Projection window unavailable.")
                return
                
            intensity = self._project_intensity_slider.value()
            self._project_with_intensity(intensity)
            self._projection_active = True
            
        except Exception as e:
            print(f"_project_on failed: {e}")
    
    def _project_off(self):
        """Turn off projection (black screen)."""
        try:
            if not self._ensure_projection():
                print("Projection window unavailable.")
                return
                
            self.projection.show_solid_fullscreen((0, 0, 0))
            self._projection_active = False
            
        except Exception as e:
            print(f"_project_off failed: {e}")

    def _sl_calibrate(self):
        """Run Structured-Light (Gray-code) calibration end-to-end."""
        try:
            from calibration import generate_gray_code_patterns, save_gray_code_patterns, decode_gray_code_from_files, invert_cam_to_proj_lut
        except Exception as e:
            print(f"Structured-light not available: {e}")
            return

        if not self._ensure_projection():
            print("Projection window unavailable.")
            return

        # 1) Generate patterns at projector resolution
        try:
            scr = self.projection.windowHandle().screen() if self.projection.windowHandle() else None
            geo = scr.geometry() if scr else None
            proj_w = geo.width() if geo else 1920
            proj_h = geo.height() if geo else 1080
            patterns = generate_gray_code_patterns(proj_w, proj_h)
            pattern_paths = save_gray_code_patterns(patterns)
            print(f"Generated {len(pattern_paths)} Gray-code patterns")
        except Exception as e:
            print(f"Failed to generate patterns: {e}")
            return

        # Disable LUT-warp button and show progress while running
        try:
            if hasattr(self, '_button_sl_project_reg') and self._button_sl_project_reg is not None:
                self._button_sl_project_reg.setEnabled(False)
            if getattr(self, '_sl_progress', None):
                self._sl_progress.setVisible(True)
                self._sl_status.setText("Capturing structured-light patterns…")
        except Exception:
            pass

        # 2) Project each pattern and capture a camera frame
        capture_paths = []
        last_pidx = None
        for idx, (ppath, meta) in enumerate(zip(pattern_paths, patterns)):
            try:
                # Prefer in-memory pattern image to avoid disk I/O latency
                img = None
                try:
                    img = meta.get("image", None)
                except Exception:
                    img = None
                if img is None:
                    img = cv2.imread(ppath, cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                # If projection engine is running and triggers are armed, stream via ZMQ to sync with projector
                use_engine = hasattr(self, '_proc_projector') and (self._proc_projector is not None)
                if use_engine:
                    try:
                        from projector_client import ProjectorClient
                        # Projector engine expects 1920x1080 luminance frames; client resizes as needed
                        client = ProjectorClient()
                        # Pace strictly: wait for next trigger from last_pidx, then send one frame, then wait until that vis_id appears
                        if last_pidx is None:
                            client.wait_next_trigger(0, timeout_ms=500)
                        else:
                            client.wait_next_trigger(last_pidx, timeout_ms=500)
                        client.send_gray(img, frame_id=idx+1, visible_overlay=self._button_toggle_overlay.isChecked())
                        matched = client.wait_visible(idx+1, timeout_ms=500)
                        if matched is not None:
                            last_pidx = matched
                        client.close()
                    except Exception as ez:
                        print(f"[SL] ZMQ send failed, falling back to local display: {ez}")
                        try:
                            self.projection.show_image_raw_no_warp_no_flip(img)
                        except Exception:
                            self.projection.show_image_fullscreen_on_second_monitor(img, None)
                else:
                    # Local path without engine
                    try:
                        self.projection.show_image_raw_no_warp_no_flip(img)
                    except Exception:
                        self.projection.show_image_fullscreen_on_second_monitor(img, None)
                # Allow minimal UI processing without delaying engine-paced path
                QtCore.QCoreApplication.processEvents()
                if not use_engine:
                    QtCore.QThread.msleep(40)
                # Capture a frame
                save_dir = getattr(self._camera, 'save_dir', './Saved_Media')
                os.makedirs(save_dir, exist_ok=True)
                cap_path = os.path.join(save_dir, f"sl_cap_{idx:03d}.png")
                if hasattr(self._camera, "snapshot"):
                    self._camera.snapshot(cap_path)
                    capture_paths.append(cap_path)
                else:
                    # As a fallback, mark missing
                    capture_paths.append("")
            except Exception as e:
                print(f"Pattern {idx} projection/capture failed: {e}")

        # 3) Decode LUTs (offload to background thread to keep GUI responsive)
        try:
            def _sl_decode_worker(paths, pats, pw, ph, asset_dir):
                try:
                    import numpy as _np, cv2 as _cv2
                    from calibration import decode_gray_code_from_files as _decode, invert_cam_to_proj_lut as _invert
                    cam_h, cam_w = 1080, 1920
                    for _fp in reversed(paths):
                        if not _fp:
                            continue
                        _img = _cv2.imread(_fp, _cv2.IMREAD_GRAYSCALE)
                        if _img is not None:
                            cam_h, cam_w = _img.shape[:2]
                            break
                    print(f"[SL] Decoding Gray-code at {cam_w}x{cam_h} → proj {pw}x{ph}…")
                    proj_x_of_cam, proj_y_of_cam = _decode(paths, pats, cam_h, cam_w, pw, ph)
                    _np.save("/".join([asset_dir, "proj_from_cam_x.npy"]), proj_x_of_cam)
                    _np.save("/".join([asset_dir, "proj_from_cam_y.npy"]), proj_y_of_cam)
                    inv_x, inv_y = _invert(proj_x_of_cam, proj_y_of_cam, pw, ph)
                    _np.save("/".join([asset_dir, "cam_from_proj_x.npy"]), inv_x)
                    _np.save("/".join([asset_dir, "cam_from_proj_y.npy"]), inv_y)
                    print("✅ Structured-light LUTs saved (background)")
                    try:
                        # Notify GUI thread
                        self.sl_decode_done.emit(True, "LUTs saved")
                    except Exception:
                        pass
                except Exception as _e:
                    print(f"Structured-light decoding failed: {_e}")
                    try:
                        self.sl_decode_done.emit(False, str(_e))
                    except Exception:
                        pass

            import threading as _th
            _th.Thread(target=_sl_decode_worker, args=(capture_paths, patterns, proj_w, proj_h, self._camera.asset_dir), daemon=True).start()
            print("[SL] Decoding LUTs in background… GUI remains responsive")
        except Exception as e:
            print(f"Structured-light decoding thread failed to start: {e}")
    
    def _sl_project_registration(self):
        """Prewarp and project the custom registration image using LUTs."""
        try:
            from calibration import prewarp_with_inverse_lut
        except Exception as e:
            print(f"Structured-light prewarp not available: {e}")
            return
        if not self._ensure_projection():
            print("Projection window unavailable.")
            return
        try:
            # Load LUTs
            asset_dir = getattr(self._camera, 'asset_dir', str((Path(__file__).resolve().parent / "Assets" / "Generated").resolve()))
            inv_x = np.load("/".join([asset_dir, "cam_from_proj_x.npy"]))
            inv_y = np.load("/".join([asset_dir, "cam_from_proj_y.npy"]))
            proj_h, proj_w = inv_x.shape[:2]
            # Load registration image in camera space (same as camera preview size preferred). If sizes differ, we will scale.
            img_path = (Path(asset_dir).parent / "Generated" / "custom_registration_image.png").resolve()
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"Registration image not readable: {img_path}")
                return
            # Resize registration to camera frame size if we can detect it from a snapshot
            cam_h, cam_w = img.shape[:2]
            try:
                # Try loading a recent snapshot to infer true camera dims
                save_dir = getattr(self._camera, 'save_dir', './Saved_Media')
                candidates = sorted([p for p in os.listdir(save_dir) if p.endswith('.png')])
                for name in reversed(candidates):
                    probe = cv2.imread(os.path.join(save_dir, name), cv2.IMREAD_GRAYSCALE)
                    if probe is not None:
                        cam_h, cam_w = probe.shape[:2]
                        break
                if (img.shape[1], img.shape[0]) != (cam_w, cam_h):
                    img = cv2.resize(img, (cam_w, cam_h), interpolation=cv2.INTER_LINEAR)
            except Exception:
                pass
            # Prewarp
            warped = prewarp_with_inverse_lut(img, inv_x, inv_y, proj_w, proj_h)
            # Prefer projection engine via ZMQ if running; ensures sync with triggers
            use_engine = hasattr(self, '_proc_projector') and (self._proc_projector is not None)
            if use_engine:
                try:
                    from projector_client import ProjectorClient
                    # Engine expects 1920x1080; client will resize
                    client = ProjectorClient()
                    client.send_gray(warped, frame_id=9999, visible_overlay=self._button_toggle_overlay.isChecked())
                    client.close()
                except Exception as ez:
                    print(f"[SL] ZMQ send failed, falling back to local display: {ez}")
                    try:
                        self.projection.show_image_raw_no_warp_no_flip(warped)
                    except Exception:
                        self.projection.show_image_fullscreen_on_second_monitor(warped, None)
            else:
                # Project raw without flip/warp (LUT already maps correctly)
                try:
                    self.projection.show_image_raw_no_warp_no_flip(warped)
                except Exception:
                    self.projection.show_image_fullscreen_on_second_monitor(warped, None)
            print("✅ Projected LUT-prewarped registration")
        except Exception as e:
            print(f"LUT projection failed: {e}")

    def _project_with_intensity(self, intensity):
        """Project a solid color with the specified intensity (0-255)."""
        try:
            if not self._ensure_projection():
                print("Projection window unavailable.")
                return
                
            # Use the intensity value for all RGB channels (grayscale)
            self.projection.show_solid_fullscreen((intensity, intensity, intensity))
            
        except Exception as e:
            print(f"_project_with_intensity failed: {e}")

    def _apply_modern_style(self):
        # Styling intentionally disabled for revert.
        return

    def _on_camera_type_changed(self, camera_type):
        """Handle camera type selection change."""
        self.selected_camera_type = camera_type
        print(f"Camera type changed to: {camera_type}")
        # Note: Camera type change will take effect on next restart

    def change_pixel_format(self, *_):
        pixel_format = self._dropdown_pixel_format.currentText()
        self._camera.change_pixel_format(pixel_format)



    def change_hardware_trigger_line(self, *_):
        chosen_line = self._dropdown_trigger_line.currentText()
        print(f"Chosen hardware trigger line: {chosen_line}")
        
        self._camera.change_hardware_trigger_line(chosen_line)
    
    @QtCore.pyqtSlot(object)
    def on_image_received(self, image):
     
        try:
            import numpy as np
            import cv2


            def _get_attr(obj, names):
                for n in names:
                    v = getattr(obj, n, None)
                    if callable(v):
                        try:
                            return v()
                        except Exception:
                            continue
                    elif v is not None:
                        return v
                return None

            def _get_int(obj, names):
                v = _get_attr(obj, names)
                try:
                    return int(v)
                except Exception:
                    return None

            def _bayer_code(pf_str: str):
                s = (pf_str or "").upper()
                if "BAYERRG" in s: return cv2.COLOR_BayerRG2RGB
                if "BAYERBG" in s: return cv2.COLOR_BayerBG2RGB
                if "BAYERGB" in s: return cv2.COLOR_BayerGB2RGB
                if "BAYERGR" in s: return cv2.COLOR_BayerGR2RGB
                return None

            def _bit_depth_shift(pf_str: str):
                s = (pf_str or "").upper()

                if "12" in s: return 4
                if "10" in s: return 2
                if "16" in s: return 8
                return 0

            def _numpy_from_ids(img_obj):
                for n in ("get_numpy", "get_numpy_view", "get_numpy_array", "get_numpy_1D"):
                    f = getattr(img_obj, n, None)
                    if callable(f):
                        try:
                            arr = f()
                            if isinstance(arr, np.ndarray):
                                return arr
                        except Exception:
                            pass

                f = getattr(img_obj, "get_buffer", None)
                if callable(f):
                    try:
                        raw = f()
                        if raw is not None:
                            return np.frombuffer(raw, dtype=np.uint8)
                    except Exception:
                        pass
                return None


            pf_str = ""

            if isinstance(image, np.ndarray):
                arr = image
                h, w = arr.shape[:2]
                ch = 1 if arr.ndim == 2 else arr.shape[2]
            else:

                w = _get_int(image, ("Width", "width", "GetWidth", "ImageWidth"))
                h = _get_int(image, ("Height", "height", "GetHeight", "ImageHeight"))
                pf   = _get_attr(image, ("PixelFormat", "pixel_format", "GetPixelFormat", "PixelFormatName"))
                pf_str = str(pf) if pf is not None else ""

                arr = _numpy_from_ids(image)
                if arr is None:
                    print("on_image_received: no buffer -> dropping frame")
                    return

                if arr.ndim == 3:

                    h, w, ch = arr.shape
                elif arr.ndim == 2:

                    ch = 1
                else:

                    channels = 4 if ("BGRA" in pf_str or "RGBA" in pf_str) else 3 if ("BGR" in pf_str or "RGB" in pf_str) else 1
                    if not (w and h):
                        print("on_image_received: unknown WxH for 1D buffer")
                        return
                    expected = w * h * channels
                    if arr.size < expected:
                        print("on_image_received: buffer smaller than expected")
                        return
                    arr = arr[:expected].reshape(h, w, channels) if channels > 1 else arr[:w*h].reshape(h, w)
                    ch = channels



            if arr.dtype == np.uint16:

                shift = _bit_depth_shift(pf_str) if pf_str else 8
                arr8 = (arr >> shift).astype(np.uint8, copy=False)
            elif arr.dtype != np.uint8:
                arr8 = arr.astype(np.uint8, copy=False)
            else:
                arr8 = arr


            bayer = _bayer_code(pf_str)
            if (arr8.ndim == 2 or (arr8.ndim == 3 and arr8.shape[2] == 1)) and bayer is not None:
                try:
                    rgb = cv2.cvtColor(arr8 if arr8.ndim == 2 else arr8[:, :, 0], bayer)
                    qsrc = rgb  
                    h, w = qsrc.shape[:2]
                    fmt = QtGui.QImage.Format_RGB888
                    bpl = int(qsrc.strides[0])  
                    qimg = QtGui.QImage(qsrc.data, w, h, bpl, fmt).copy()
                except Exception as e:
                    print(f"Demosaic failed ({pf_str}), falling back to grayscale: {e}")
                    qsrc = arr8 if arr8.ndim == 2 else arr8[:, :, 0]
                    h, w = qsrc.shape[:2]
                    fmt = QtGui.QImage.Format_Grayscale8
                    bpl = int(qsrc.strides[0])
                    qimg = QtGui.QImage(qsrc.data, w, h, bpl, fmt).copy()
            else:

                if arr8.ndim == 2 or (arr8.ndim == 3 and arr8.shape[2] == 1):
                    qsrc = arr8 if arr8.ndim == 2 else arr8[:, :, 0]
                    h, w = qsrc.shape[:2]
                    fmt = QtGui.QImage.Format_Grayscale8
                    bpl = int(qsrc.strides[0])
                elif arr8.shape[2] == 3:


                    if "BGR" in (pf_str or "").upper():
                        qsrc = cv2.cvtColor(arr8, cv2.COLOR_BGR2RGB)
                    else:

                        qsrc = arr8
                    h, w = qsrc.shape[:2]
                    fmt = QtGui.QImage.Format_RGB888
                    bpl = int(qsrc.strides[0])
                else:


                    if "BGRA" in (pf_str or "").upper():
                        qsrc = cv2.cvtColor(arr8, cv2.COLOR_BGRA2RGBA)
                    else:
                        qsrc = arr8
                    h, w = qsrc.shape[:2]
                    fmt = QtGui.QImage.Format_RGBA8888
                    bpl = int(qsrc.strides[0])

                qimg = QtGui.QImage(qsrc.data, w, h, bpl, fmt).copy()


            try:
                GUIfps = float(self._camera.get_actual_fps())
                self.fps_update_signal.emit(GUIfps)
            except Exception:
                pass


            self.image_update_signal.emit(qimg)

        except Exception as e:
            print(f"on_image_received failed: {e}")


        

    def on_projection_received(self, image, homography_matrix = None):
        """
        Update Projection Image
        """


        try:
            self.projection.show_image_fullscreen_on_second_monitor(image, homography_matrix)
        except Exception as e:
            print(f"Error updating Projection, {e}")

    def warning(self, message: str):
        self.messagebox_pyqtSignal.emit("Warning", message)

    def information(self, message: str):
        self.messagebox_pyqtSignal.emit("Information", message)



    def show_gpu_ui(self):
        try:
            from gpu_ui import GPU
        except ImportError as e:
            print(f"CRISPI UI not available: {e}")
            return

        if not _GPU_AVAILABLE:
            print("CRISPI UI not available in this environment.")
            return
        if self.gpu_ui is None:
            try:
                self.gpu_ui = GPU(camera=self._camera, parent=self)
            except TypeError:
                self.gpu_ui = GPU(camera=self._camera)
                self.gpu_ui.setParent(self)
        self.gpu_ui.setWindowFlags(Qt.Tool)
        self.gpu_ui.show()




    @Slot(str, str)
    def message(self, typ: str, message: str):
        if typ == "Warning":
            QtWidgets.QMessageBox.warning(
                self, "Warning", message, QtWidgets.QMessageBox.Ok)
        else:
            QtWidgets.QMessageBox.information(
                self, "Information", message, QtWidgets.QMessageBox.Ok)


    @Slot(float)
    def change_slider_gain(self, val):
        self._gain_slider.setValue(int(val * 100))

    @Slot(int)
    def _update_gain(self, val):
        value = val / 100
        self._gain_value_label.setText(f"{value:.2f}")
        try:

            self._camera.node_map.FindNode("GainSelector").SetCurrentEntry("AnalogAll")
        except Exception:
            pass
        self._camera.set_gain(value)



    @Slot(float)
    def change_slider_dgain(self, val):
        self._dgain_slider.setValue(int(val * 100))

    @Slot(int)
    def _update_dgain(self, val):
        value = val / 100
        self._dgain_value_label.setText(f"{value:.2f}")
        try:
            self._camera.node_map.FindNode("GainSelector").SetCurrentEntry("DigitalAll")
        except Exception:
            pass
        self._camera.set_gain(value)


    # Zoom slider methods removed - using mouse wheel zoom instead

    def _send_hmatrix_to_projector(self):
        try:
            import numpy as np
            # Prefer in-memory H from last calibration
            H = getattr(self._camera, 'translation_matrix', None)
            if H is None or not hasattr(H, 'shape'):
                # Fallback to npy on disk
                npy_path = (ASSETS / 'Generated' / 'homography_cam2proj.npy').resolve()
                if npy_path.exists():
                    H = np.load(str(npy_path))
            if H is None:
                print("No H-matrix available. Calibrate first.")
                return
            self._camera._send_h_to_projector(H)
        except Exception as e:
            print(f"REQ H-Matrix failed: {e}")

    def _toggle_overlay(self, checked: bool):
        try:
            if not hasattr(self, '_button_toggle_overlay') or self._button_toggle_overlay is None:
                return
            self._button_toggle_overlay.setText("Overlay: On" if checked else "Overlay: Off")
            if hasattr(self, '_proc_projector') and self._proc_projector is not None:
                print("[PROJ] Overlay toggle changed; restart Projection Engine to apply")
        except Exception as e:
            print(f"_toggle_overlay error: {e}")
