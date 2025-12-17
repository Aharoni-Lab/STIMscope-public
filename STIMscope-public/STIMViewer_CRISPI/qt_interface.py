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
        
        # Update exposure text box to show default value
        if hasattr(self, '_exp_line'):
            self._exp_line.setText("33333.333")
        
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
        # Expanded options with explicit labels and defaults; make 8-bit RGB (0x03) the default
        self._seq_type_dropdown.addItems([
            "8-bit RGB (0x03)",
            "8-bit Mono (0x02)",
            "1-bit RGB (0x01)",
            "1-bit Mono (0x00)",
        ])
        try:
            self._seq_type_dropdown.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
            self._seq_type_dropdown.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        except Exception:
            pass
        try:
            self._seq_type_dropdown.currentTextChanged.connect(self._on_seq_type_changed)
        except Exception:
            pass
        self._button_toggle_overlay = QtWidgets.QPushButton("Enable Overlay")
        self._button_toggle_overlay.setCheckable(True)
        self._button_toggle_overlay.setChecked(False)
        self._button_toggle_overlay.toggled.connect(self._toggle_overlay)
        # Initialize label to current state
        try:
            self._toggle_overlay(self._button_toggle_overlay.isChecked())
        except Exception:
            pass
        self._proj_warp_mode = "NONE"  # default: no warp until user selects
        self._button_req_hmatrix = QtWidgets.QPushButton("REQ H-Matrix")
        self._button_req_hmatrix.setCheckable(True)
        self._button_req_hmatrix.setChecked(False)
        self._button_req_hmatrix.toggled.connect(self._on_warp_h_toggled)
        self._button_use_lut = QtWidgets.QPushButton("REQ LUT")
        self._button_use_lut.setCheckable(True)
        self._button_use_lut.setChecked(False)
        self._button_use_lut.toggled.connect(self._on_warp_lut_toggled)
        # Mask pattern selection UI
        self._mask_pattern_label = QtWidgets.QLabel("Mask Pattern")
        self._mask_pattern_dropdown = QtWidgets.QComboBox()
        self._mask_pattern_dropdown.addItems([
            "Seg Mask", "Moving Bar", "Checkerboard", "Solid", "Circle", "Gradient", "Image", "Folder", "Custom"
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
        # Keep trigger/mask buttons compact to text, slightly larger
        try:
            self._button_send_triggers.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            _set_compact_width_to_text(self._button_send_triggers, 28)
        except Exception:
            pass
        try:
            self._button_send_masks.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            _set_compact_width_to_text(self._button_send_masks, 28)
        except Exception:
            pass




        
        self._button_show_gpu_ui = QtWidgets.QPushButton("Real-Time Trace Extraction")
        self._button_show_gpu_ui.clicked.connect(self.show_gpu_ui)
        self._button_show_gpu_ui.setEnabled(_GPU_AVAILABLE)
        try:
            self._button_show_gpu_ui.setStyleSheet(
                """
                QPushButton {
                    color: #000000; /* keep text black */
                    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #f5eeff, stop:1 #ece2ff);
                    border: 1px solid #cdbcf3;
                    border-radius: 6px;
                    padding: 4px 10px;
                }
                QPushButton:hover {
                    color: #000000;
                    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #f2e9ff, stop:1 #e4d6ff);
                    border: 1px solid #b49cf0;
                }
                QPushButton:pressed {
                    color: #000000;
                    background-color: #dbcaff;
                }
                QPushButton:disabled {
                    color: #b8b6c9;
                    background-color: #fafafa;
                    border: 1px solid #eeeeee;
                }
                """
            )
        except Exception:
            pass
        



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
        self._button_sl_project_reg = QtWidgets.QPushButton("Project LUT-Warped")
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
        # Move Start Projection Engine next to Start Hardware Acquisition (right side)
        row0_layout.addWidget(self._button_start_projector)
        # The calibration-related buttons are moved to a dedicated top panel
        # (Calibrate, Structured-Light Calibrate, Subpixel, Project LUT-Warped)
        try:
            self._chk_phase_refine = QtWidgets.QCheckBox("Subpixel")
            self._chk_phase_refine.setChecked(False)
            self._chk_phase_refine.setToolTip("Enable sinusoidal phase refinement for subpixel LUT. If results degrade, uncheck.")
        except Exception:
            pass
        row0_widget = QtWidgets.QWidget()
        row0_widget.setLayout(row0_layout)
        config_layout.addWidget(row0_widget,                             0, 0, 1, 2, Qt.AlignLeft)
        # Row 1: Projection engine and trigger controls
        row1_layout = QtWidgets.QHBoxLayout()
        row1_layout.addWidget(self._seq_type_label)
        row1_layout.addWidget(self._seq_type_dropdown)
        row1_layout.addWidget(self._button_toggle_overlay)
        row1_layout.addWidget(self._button_req_hmatrix)
        row1_layout.addWidget(self._button_use_lut)
        row1_widget = QtWidgets.QWidget()
        row1_widget.setLayout(row1_layout)
        config_layout.addWidget(row1_widget,                             1, 0, 1, 2)
        
        # New Row 2: mask pattern selection and send controls
        row2_layout = QtWidgets.QHBoxLayout()
        try:
            row2_layout.setSpacing(2)  # tighter gap between label and dropdown
            row2_layout.setContentsMargins(0, 0, 0, 0)
        except Exception:
            pass
        # Hardware trigger out toggle (left side of Mask Pattern)
        self._button_hw_trig = QtWidgets.QPushButton("HW Trigger Out")
        self._button_hw_trig.setCheckable(True)
        self._button_hw_trig.setChecked(False)
        try:
            self._button_hw_trig.setToolTip("Toggle GPIO trigger out on every projector frame (BOARD pin 22)")
        except Exception:
            pass
        self._button_hw_trig.toggled.connect(self._toggle_hw_trigger_out)
        row2_layout.addWidget(self._button_hw_trig)
        try:
            self._mask_pattern_label.setContentsMargins(0, 0, 0, 0)
            self._mask_pattern_label.setStyleSheet("margin:0px; padding-right:2px;")
        except Exception:
            pass
        # Tight pair: label + dropdown with zero spacing
        try:
            mp_pair_widget = QtWidgets.QWidget()
            mp_pair_layout = QtWidgets.QHBoxLayout(mp_pair_widget)
            mp_pair_layout.setContentsMargins(0, 0, 0, 0)
            mp_pair_layout.setSpacing(0)
            try:
                self._mask_pattern_label.setContentsMargins(0, 0, 0, 0)
                self._mask_pattern_label.setStyleSheet("margin:0px; padding-right:1px;")
            except Exception:
                pass
            mp_pair_layout.addWidget(self._mask_pattern_label)
            mp_pair_layout.addWidget(self._mask_pattern_dropdown)
            row2_layout.addWidget(mp_pair_widget)
        except Exception:
            # Fallback: add directly
            row2_layout.addWidget(self._mask_pattern_label)
            row2_layout.addWidget(self._mask_pattern_dropdown)
        row2_layout.addWidget(self._mask_pattern_browse)
        # Shift buttons left: replace stretch with a small spacing
        row2_layout.addSpacing(8)
        # New: Set Trig Params button (kept on HW Trigger Out row)
        self._button_set_trig_params = QtWidgets.QPushButton("Set Trig Params")
        try:
            self._button_set_trig_params.setToolTip("Configure TriggerDelay (µs) and ExposureTime (µs)")
        except Exception:
            pass
        self._button_set_trig_params.clicked.connect(self._open_trig_params_dialog)
        row2_layout.addWidget(self._button_set_trig_params)
        row2_widget = QtWidgets.QWidget()
        row2_widget.setLayout(row2_layout)
        config_layout.addWidget(row2_widget,                             2, 0, 1, 2)
        
        # New Row (under HW Trigger Out row): start projector trigger and send masks
        row2b_layout = QtWidgets.QHBoxLayout()
        row2b_layout.setContentsMargins(0, 0, 0, 0)
        row2b_layout.setSpacing(6)
        row2b_layout.addWidget(self._button_send_triggers)
        row2b_layout.addWidget(self._button_send_masks)
        row2b_layout.addStretch(1)
        row2b_widget = QtWidgets.QWidget()
        row2b_widget.setLayout(row2b_layout)
        config_layout.addWidget(row2b_widget,                            3, 0, 1, 2, Qt.AlignLeft)
        
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
        config_layout.addWidget(project_buttons_widget,                  4, 0, 1, 2)
        
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
        config_layout.addWidget(row_cam_all_widget,                      5, 0, 1, 2, Qt.AlignLeft)


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
        # Removed from panel; accessible via Sensor Settings window
        self._gain_value_label = QtWidgets.QLabel("1.00")
        self._gain_value_label.setAlignment(Qt.AlignCenter)
        self._gain_value_label.setStyleSheet("font-size: 10px;")
        # not added to layout


        self._dgain_label.setAlignment(Qt.AlignCenter)
        self._dgain_slider.setFixedWidth(15)  # Make narrower
        # Removed from panel; accessible via Sensor Settings window
        self._dgain_value_label = QtWidgets.QLabel("1.00")
        self._dgain_value_label.setAlignment(Qt.AlignCenter)
        self._dgain_value_label.setStyleSheet("font-size: 10px;")
        # not added to layout

        # Exposure entry (µs)
        self._exp_label = QtWidgets.QLabel("EXP (µs)")
        self._exp_label.setAlignment(Qt.AlignCenter)
        # Removed from panel; accessible via Sensor Settings window
        self._exp_line = QtWidgets.QLineEdit("33333.333")
        self._exp_line.setAlignment(Qt.AlignCenter)
        self._exp_line.setValidator(QtGui.QDoubleValidator(1.0, 1e9, 3))
        self._exp_line.editingFinished.connect(self._apply_exposure_from_text)
        # not added to layout

        # Buttons row (horizontal)
        btn_row = QtWidgets.QHBoxLayout()
        self._button_sensor_settings = QtWidgets.QPushButton("Sensor Settings")
        self._button_sensor_settings.clicked.connect(self._open_sensor_settings)
        btn_row.addWidget(self._button_sensor_settings)
        self._button_troubleshoot = QtWidgets.QPushButton("Troubleshooting")
        try:
            self._button_troubleshoot.setToolTip("Open troubleshooting tools: GPIO test, engine/camera status, performance graphs")
        except Exception:
            pass
        self._button_troubleshoot.clicked.connect(self._open_troubleshoot_window)
        btn_row.addWidget(self._button_troubleshoot)
        # Add ASIFT Calibration button to the right of Troubleshooting
        self._button_asift = QtWidgets.QPushButton("ASIFT Calibration")
        try:
            self._button_asift.setToolTip("Compute 3x3 H using Affine-SIFT and apply to projector")
        except Exception:
            pass
        self._button_asift.clicked.connect(self._asift_calibrate)
        btn_row.addWidget(self._button_asift)
        control_group_layout.addLayout(btn_row, 5, 0, 1, 2)


        # Zoom controls removed - using mouse wheel zoom instead


        # Set control panel widths for larger buttons
        control_group.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Fixed
        )
        
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

        # New: Top calibration panel (above hardware trigger/config zone)
        try:
            calib_panel = QtWidgets.QWidget()
            calib_panel.setObjectName("calib_panel")
            calib_layout = QtWidgets.QHBoxLayout(calib_panel)
            calib_layout.setContentsMargins(6, 6, 6, 6)
            calib_layout.setSpacing(6)
            # Style similar to other panels but without a title area
            calib_panel.setStyleSheet(
                "border: 1px solid #d1d1d6;"
                "border-radius: 6px;"
                "margin-top: 2px;"
                "font-size: 11px;"
                "color: #1c1c1e;"
                "background-color: #ffffff;"
                "padding: 4px;"
                " QPushButton { font-weight: normal; color: #000000;"
                "   background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f5f5f7, stop:1 #eaeaef);"
                "   border: 1px solid #cfcfd6; border-radius: 6px; padding: 4px 10px; }"
                " QPushButton:hover {"
                "   background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #f1f1f6);"
                "   border: 1px solid #bdbdca; }"
                " QPushButton:pressed { background-color: #e6e6ee; }"
                " QPushButton:disabled { color: #b8b6c9; background-color: #fafafa; border: 1px solid #eeeeee; }"
            )
            # Move calibration-related controls here
            calib_layout.addWidget(self._button_calibrate)
            calib_layout.addWidget(self._button_sl_calibrate)
            try:
                calib_layout.addWidget(self._chk_phase_refine)
            except Exception:
                pass
            calib_layout.addWidget(self._button_sl_project_reg)
            # Place the new panel at the very top-left
            button_bar_layout.addWidget(calib_panel, 0, 0, 1, 1)
        except Exception:
            pass

        # Shift everything to the left to align with video preview; push existing panels down
        button_bar_layout.addWidget(config_group, 1, 0, 4, 1)       # Column 0 (under calibration panel)
        button_bar_layout.addWidget(capture_group, 5, 0, 2, 1)      # Column 0, below config
        # Keep control panel as a separate panel below the left column panels
        button_bar_layout.addWidget(control_group,                  7, 0, 1, 1, Qt.AlignLeft)
        
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
        try:
            self._exp_label.setToolTip("Exposure in microseconds. Default 33333.333 (≈30 FPS).")
            self._exp_line.setToolTip("Type exposure in µs and press Enter.")
        except Exception:
            pass
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
                # Link order matters: GLEW before GL on Linux
                "-lglfw", "-lGLEW", "-lGL", "-lzmq", "-lgpiod", "-lpthread"
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
        need_path = text in ("Image", "Folder", "Custom")
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
            elif typ == "Custom":
                # Allow selecting either a Python sender or a compiled custom sender (including no extension)
                fp, _ = QFileDialog.getOpenFileName(self, "Select Sender (Python or Executable)", str(Path.home()),
                                                    "All Files (*)")
                if fp:
                    self._mask_pattern_path = fp
        except Exception as e:
            print(f"Browse failed: {e}")

    def _toggle_hw_trigger_out(self, checked: bool):
        """Enable/disable GPIO trigger out on Jetson BOARD pin 22.
        When enabled, each engine frame send will emit a short pulse.
        """
        try:
            import Jetson.GPIO as GPIO
            pin = 22  # J30 pin 22 -> GPIO17
            if checked:
                GPIO.setmode(GPIO.BOARD)
                GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
                self._hw_trig_pin = pin
                self._hw_trig_enabled = True
                print("[HWTRIG] Enabled on BOARD pin 22")
                # Start background subscriber that pulses on every projector visibility event
                try:
                    import threading as _th
                    import zmq as _zmq
                    self._hw_trig_stop = _th.Event()

                    def _loop():
                        last_pidx = 0
                        try:
                            ctx = _zmq.Context.instance()
                            sub = ctx.socket(_zmq.SUB)
                            sub.setsockopt(_zmq.LINGER, 0)
                            sub.setsockopt_string(_zmq.SUBSCRIBE, "")
                            sub.connect("tcp://127.0.0.1:5562")
                        except Exception as _e:
                            print(f"[HWTRIG] SUB init error: {_e}")
                            return
                        while not self._hw_trig_stop.is_set():
                            try:
                                msg = sub.recv(flags=_zmq.NOBLOCK)
                                s = msg.decode('utf-8', errors='ignore')
                                # Minimal JSON parse
                                pidx = None
                                vis = None
                                try:
                                    import json as _json
                                    d = _json.loads(s)
                                    pidx = int(d.get('pidx', 0))
                                    vis = int(d.get('vis_id', -1))
                                except Exception:
                                    pass
                                if pidx is not None and pidx > last_pidx and vis is not None and vis >= 0:
                                    try:
                                        GPIO.output(pin, GPIO.HIGH)
                                        import time as _t
                                        _t.sleep(0.001)
                                        GPIO.output(pin, GPIO.LOW)
                                    except Exception as _e:
                                        print(f"[HWTRIG] Pulse error: {_e}")
                                    last_pidx = pidx
                            except Exception:
                                # No message yet
                                import time as _t
                                _t.sleep(0.005)

                    self._hw_trig_thread = _th.Thread(target=_loop, daemon=True)
                    self._hw_trig_thread.start()
                except Exception as _e:
                    print(f"[HWTRIG] Subscriber start error: {_e}")
            else:
                try:
                    GPIO.output(getattr(self, '_hw_trig_pin', pin), GPIO.LOW)
                    GPIO.cleanup(getattr(self, '_hw_trig_pin', pin))
                except Exception:
                    pass
                self._hw_trig_enabled = False
                print("[HWTRIG] Disabled and cleaned up")
                # Stop background subscriber
                try:
                    if hasattr(self, '_hw_trig_stop') and self._hw_trig_stop is not None:
                        self._hw_trig_stop.set()
                    if hasattr(self, '_hw_trig_thread') and self._hw_trig_thread is not None:
                        self._hw_trig_thread.join(timeout=0.5)
                except Exception:
                    pass
        except Exception as e:
            print(f"[HWTRIG] Setup error: {e}")

    def _test_hw_trigger_pulse(self):
        try:
            import Jetson.GPIO as GPIO, time as _t
            pin = 22
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
            print("[HWTRIG] Test: 5 pulses on BOARD 22")
            for _ in range(5):
                GPIO.output(pin, GPIO.HIGH); _t.sleep(0.01)
                GPIO.output(pin, GPIO.LOW);  _t.sleep(0.01)
            # leave low
        except Exception as e:
            print(f"[HWTRIG] Test pulse error: {e}")

    # ---------------- Troubleshooting Window ----------------
    def _open_troubleshoot_window(self):
        try:
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGridLayout, QWidget, QMessageBox
            import psutil, os, cv2, numpy as _np
        except Exception as e:
            print(f"Troubleshooting UI error: {e}")
            return

        # Optional plotting
        try:
            import pyqtgraph as pg
            _HAS_PG = True
        except Exception:
            _HAS_PG = False

        dlg = QDialog(self)
        dlg.setWindowTitle("Troubleshooting")
        dlg.setMinimumSize(680, 420)
        lay = QVBoxLayout(dlg)

        # Row: quick actions & engine monitor toggle
        row = QHBoxLayout()
        btn_test = QtWidgets.QPushButton("Test HW Trigger Out Pulse")
        btn_test.clicked.connect(self._test_hw_trigger_pulse)
        btn_mon = QtWidgets.QPushButton("Start Engine Monitor")
        btn_mon.setCheckable(True)
        status_lbl = QLabel("Engine: idle")
        last_lbl = QLabel("Last: pidx=-- vis=-- rate=-- Hz")
        # Trigger indicator button (non-interactive)
        ind_btn = QtWidgets.QPushButton("Projector Trigger: OFF")
        ind_btn.setEnabled(False)
        ind_btn.setStyleSheet("QPushButton{background-color:#ff4d4f; color:white; border-radius:6px; padding:4px 8px;}")
        row.addWidget(btn_test)
        row.addSpacing(10)
        row.addWidget(btn_mon)
        row.addSpacing(10)
        row.addWidget(status_lbl)
        row.addSpacing(10)
        row.addWidget(ind_btn)
        row.addStretch()
        lay.addLayout(row)

        # Live graphs (CPU, GPU, Mem)
        grid = QGridLayout()
        if _HAS_PG:
            pg.setConfigOptions(antialias=True)
            def _small_plot(title, pen_color):
                w = pg.PlotWidget()
                w.setTitle(title)
                w.setMinimumSize(160, 100)
                w.setMaximumHeight(110)
                c = w.plot(pen=pg.mkPen(pen_color, width=2))
                w.getPlotItem().hideButtons()
                w.getPlotItem().setLabel('bottom', '')
                w.getPlotItem().setLabel('left', '')
                w.getPlotItem().getAxis('left').setStyle(showValues=False)
                w.getPlotItem().getAxis('bottom').setStyle(showValues=False)
                return w, c
            cpu_plot, cpu_curve = _small_plot("CPU %", '#2ecc71')
            mem_plot, mem_curve = _small_plot("Mem %", '#3498db')
            gpu_plot, gpu_curve = _small_plot("GPU %", '#9b59b6')
            grid.addWidget(cpu_plot, 0, 0)
            grid.addWidget(mem_plot, 0, 1)
            grid.addWidget(gpu_plot, 0, 2)
        else:
            lbl_cpu = QLabel("CPU: -- %")
            lbl_mem = QLabel("Mem: -- %")
            lbl_gpu = QLabel("GPU: -- %")
            grid.addWidget(lbl_cpu, 0, 0)
            grid.addWidget(lbl_mem, 0, 1)
            grid.addWidget(lbl_gpu, 0, 2)
        lay.addLayout(grid)

        # ---------------- Structured-Light Validation ----------------
        def _load_luts():
            asset_dir = getattr(self._camera, 'asset_dir', str((Path(__file__).resolve().parent / "Assets" / "Generated").resolve()))
            xfp = os.path.join(asset_dir, "cam_from_proj_x.npy")
            yfp = os.path.join(asset_dir, "cam_from_proj_y.npy")
            if not (os.path.isfile(xfp) and os.path.isfile(yfp)):
                QMessageBox.warning(dlg, "LUTs Missing", "cam_from_proj_{x,y}.npy not found. Run Structured-Light calibration first.")
                return None, None, asset_dir
            try:
                inv_x = _np.load(xfp).astype(_np.float32)
                inv_y = _np.load(yfp).astype(_np.float32)
                return inv_x, inv_y, asset_dir
            except Exception as e:
                QMessageBox.critical(dlg, "LUT Load Error", str(e))
                return None, None, asset_dir

        from PyQt5.QtWidgets import QGridLayout as _QGrid
        sl_row = _QGrid()
        sl_title = QLabel("Structured-Light Validation:")
        try: sl_title.setStyleSheet("font-weight:600;")
        except Exception: pass
        lay.addWidget(sl_title)

        btn_diag = QPushButton("LUT Diagnostics")
        btn_proj = QPushButton("Project Grid (LUT)")
        btn_eval = QPushButton("Capture + Evaluate")
        btn_rterr = QPushButton("Round-Trip Error (Maps)")
        btn_probe = QPushButton("Pixel Probe (1px)")
        btn_dots  = QPushButton("Dot Array Test")
        btn_rtphy = QPushButton("Round-Trip (Physical)")
        btn_edge  = QPushButton("Edge Strip Test")
        btn_calib_char = QPushButton("Calib Grid Characterization")
        # arrange buttons in two rows
        btns = [btn_diag, btn_proj, btn_eval, btn_rterr, btn_probe, btn_dots, btn_rtphy, btn_edge, btn_calib_char]
        for i, b in enumerate(btns):
            r = i // 4
            c = i % 4
            sl_row.addWidget(b, r, c)
        lay.addLayout(sl_row)

        # Zoomable preview (with mouse wheel zoom + double-click reset)
        from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
        class _ZoomGraphicsView(QGraphicsView):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                try:
                    self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
                    self.setDragMode(QGraphicsView.ScrollHandDrag)
                except Exception:
                    pass
            def wheelEvent(self, ev):
                try:
                    angle = ev.angleDelta().y() / 120.0
                    factor = 1.25 ** max(-3.0, min(3.0, angle))
                    self.scale(factor, factor)
                    ev.accept()
                except Exception:
                    super().wheelEvent(ev)
            def mouseDoubleClickEvent(self, ev):
                try:
                    self.setTransform(QtGui.QTransform())
                    # Fit current pixmap item if present
                    items = self.scene().items()
                    for it in items:
                        if isinstance(it, QGraphicsPixmapItem):
                            self.fitInView(it, Qt.KeepAspectRatio)
                            break
                    ev.accept()
                except Exception:
                    super().mouseDoubleClickEvent(ev)

        sl_scene = QGraphicsScene()
        sl_view = _ZoomGraphicsView(sl_scene)
        sl_view.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, on=True)
        sl_view.setMinimumSize(360, 220)
        sl_view.setStyleSheet("border:1px solid #d1d1d6;")
        sl_pix = QGraphicsPixmapItem()
        sl_scene.addItem(sl_pix)
        lay.addWidget(sl_view)
        # Save current calibration preview (original resolution) as TIFF
        try:
            from PyQt5.QtWidgets import QFileDialog, QMessageBox
            btn_save_tiff = QPushButton("Save Current View (TIFF)")
            try:
                btn_save_tiff.setToolTip("Save the current calibration preview image at original resolution in .tiff format")
            except Exception:
                pass
            def _on_save_current_tiff():
                try:
                    pm = sl_pix.pixmap()
                    if pm is None or pm.isNull():
                        QMessageBox.warning(dlg, "Save Image", "No image available to save.")
                        return
                    try:
                        save_dir = getattr(self._camera, 'save_dir', './Saved_Media')
                    except Exception:
                        save_dir = './Saved_Media'
                    try:
                        os.makedirs(save_dir, exist_ok=True)
                    except Exception:
                        pass
                    default_name = time.strftime("calibration_%Y%m%d_%H%M%S.tiff")
                    fp, _ = QFileDialog.getSaveFileName(
                        dlg,
                        "Save Calibration Image (TIFF)",
                        os.path.join(save_dir, default_name),
                        "TIFF Image (*.tiff *.tif);;All Files (*)"
                    )
                    if not fp:
                        return
                    # Ensure .tiff extension
                    fpl = fp.lower()
                    if not (fpl.endswith(".tiff") or fpl.endswith(".tif")):
                        fp = fp + ".tiff"
                    ok = False
                    try:
                        ok = pm.save(fp, "TIFF")
                    except Exception:
                        ok = False
                    if not ok:
                        try:
                            qimg = pm.toImage()
                            ok = qimg.save(fp, "TIFF")
                        except Exception:
                            ok = False
                    if not ok:
                        QMessageBox.warning(dlg, "Save Failed", "Could not save image to TIFF.")
                        return
                    QMessageBox.information(dlg, "Saved", f"Saved image:\n{fp}")
                except Exception as _e:
                    try:
                        QMessageBox.warning(dlg, "Save Failed", str(_e))
                    except Exception:
                        print(f"[TSAVE] Save failed: {_e}")
            btn_save_tiff.clicked.connect(_on_save_current_tiff)
            lay.addWidget(btn_save_tiff)
        except Exception as _e:
            print(f"[TSAVE] Save button init failed: {_e}")

        # Metrics output (textbox - not on top of the image)
        metrics_lbl = QLabel("Metrics / Logs:")
        metrics_box = QtWidgets.QPlainTextEdit(dlg)
        try:
            metrics_box.setReadOnly(True)
            metrics_box.setMaximumHeight(120)
        except Exception:
            pass
        lay.addWidget(metrics_lbl)
        lay.addWidget(metrics_box)

        def _to_pix(img_bgr):
            try:
                rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            except Exception:
                rgb = img_bgr
            h, w = rgb.shape[:2]
            from PyQt5.QtGui import QImage, QPixmap
            qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
            return QPixmap.fromImage(qimg.copy())

        def _on_lut_diag():
            try:
                from calibration import visualize_lut_quality as _viz
            except Exception:
                _viz = None
            inv_x, inv_y, asset_dir = _load_luts()
            if inv_x is None or _viz is None:
                return
            diag = _viz(inv_x, inv_y, os.path.join(asset_dir, "lut_diagnostics.png"))
            try:
                pm = _to_pix(diag)
                sl_pix.setPixmap(pm)
                sl_view.fitInView(sl_pix, Qt.KeepAspectRatio)
            except Exception:
                pass

        def _infer_cam_size():
            try:
                save_dir = getattr(self._camera, 'save_dir', './Saved_Media')
                names = sorted([p for p in os.listdir(save_dir) if p.endswith('.png')])
                for nm in reversed(names):
                    fp = os.path.join(save_dir, nm)
                    img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        return img.shape[1], img.shape[0]
            except Exception:
                pass
            try:
                return int(self._camera.sensor_width), int(self._camera.sensor_height)
            except Exception:
                return 1920, 1080

        def _make_cam_grid(cam_w, cam_h, cell=32, pitch=None):
            """
            Build a binary checkerboard-like grid image in camera space.
            - cell: side length of each bright square in pixels
            - pitch: center-to-center spacing (>= cell). If None or <= cell, fall back to contiguous chessboard.
            """
            g = _np.zeros((cam_h, cam_w), _np.uint8)
            cell = int(max(1, cell))
            if pitch is None or int(pitch) <= cell:
                # Classic contiguous checkerboard
                for y in range(0, cam_h, cell):
                    for x in range(0, cam_w, cell):
                        if ((x//cell)+(y//cell)) & 1:
                            y1 = min(y+cell, cam_h)
                            x1 = min(x+cell, cam_w)
                            g[y:y1, x:x1] = 255
                return g
            # Spaced squares with given pitch (>= cell)
            pitch = int(max(cell, int(pitch)))
            for y in range(0, cam_h, pitch):
                for x in range(0, cam_w, pitch):
                    # Alternate parity across pitched grid cells
                    if ((x//pitch) + (y//pitch)) & 1:
                        y1 = min(y+cell, cam_h)
                        x1 = min(x+cell, cam_w)
                        g[y:y1, x:x1] = 255
            return g

        def _on_project_grid():
            try:
                from calibration import prewarp_with_inverse_lut as _prewarp
            except Exception:
                _prewarp = None
            inv_x, inv_y, _ = _load_luts()
            if inv_x is None or _prewarp is None:
                return
            cam_w, cam_h = _infer_cam_size()
            try:
                _cell = max(1, int(sp_cell.value()))
            except Exception:
                _cell = 16
            try:
                _pitch = max(_cell, int(sp_pitch.value()))
            except Exception:
                _pitch = _cell
            grid = _make_cam_grid(cam_w, cam_h, cell=_cell, pitch=_pitch)
            proj_h, proj_w = inv_x.shape
            warped = _prewarp(cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR), inv_x, inv_y, proj_w, proj_h)
            # Prefer sending to the projection engine to avoid GL/X context conflicts
            use_engine = hasattr(self, '_proc_projector') and (self._proc_projector is not None)
            if use_engine:
                try:
                    # Clear H so prewarped content is not warped again
                    import zmq as _zmq
                    _ctx = _zmq.Context.instance(); _s = _ctx.socket(_zmq.REQ)
                    _s.setsockopt(_zmq.LINGER, 0)
                    _s.connect("tcp://127.0.0.1:5560"); _s.send(b"IDENTITY"); _ = _s.recv(); _s.close()
                except Exception:
                    pass
                try:
                    from projector_client import ProjectorClient
                    client = ProjectorClient()
                    # Engine expects 1920x1080 luminance; client will resize.
                    client.send_gray(warped, frame_id=7777, visible_id=0, immediate=True)
                    client.close()
                    return
                except Exception:
                    pass
            # Fallback: draw via Qt projector window
            try:
                self.projection.show_image_raw_no_warp_no_flip(warped)
            except Exception:
                self.projection.show_image_fullscreen_on_second_monitor(warped, None)

        # ---------------- Homography (H) Validation (simple calibration) ----------------
        h_title = QLabel("Calibration (H) Validation:")
        try: h_title.setStyleSheet("font-weight:600;")
        except Exception: pass
        lay.addWidget(h_title)
        h_row = _QGrid()
        btn_h_proj = QPushButton("Project Grid (H)")
        btn_h_eval = QPushButton("Capture + Evaluate (H)")
        btn_h_dots = QPushButton("Dot Array Test (H)")
        h_row.addWidget(btn_h_proj, 0, 0)
        h_row.addWidget(btn_h_eval, 0, 1)
        h_row.addWidget(btn_h_dots, 0, 4)
        # Grid pitch control
        lbl_cell = QLabel("Cell (px):")
        sp_cell = QtWidgets.QSpinBox(dlg)
        try:
            sp_cell.setRange(1, 256)
            sp_cell.setSingleStep(1)
            sp_cell.setValue(16)
            sp_cell.setToolTip("Grid square size in camera pixels")
        except Exception:
            pass
        h_row.addWidget(lbl_cell, 0, 2)
        h_row.addWidget(sp_cell, 0, 3)
        # Pitch control (>= Cell)
        lbl_pitch = QLabel("Pitch (px):")
        sp_pitch = QtWidgets.QSpinBox(dlg)
        try:
            sp_pitch.setRange(1, 512)
            sp_pitch.setSingleStep(1)
            sp_pitch.setValue(int(sp_cell.value()))
            sp_pitch.setToolTip("Center-to-center spacing of squares; must be >= Cell")
        except Exception:
            pass
        def _sync_pitch_min():
            try:
                sp_pitch.setMinimum(int(sp_cell.value()))
                if int(sp_pitch.value()) < int(sp_cell.value()):
                    sp_pitch.setValue(int(sp_cell.value()))
            except Exception:
                pass
        try:
            sp_cell.valueChanged.connect(_sync_pitch_min)
        except Exception:
            pass
        h_row.addWidget(lbl_pitch, 0, 5)
        h_row.addWidget(sp_pitch, 0, 6)
        lay.addLayout(h_row)

        def _on_h_project_grid():
            try:
                import cv2, numpy as _np
            except Exception:
                QMessageBox.warning(dlg, "Dependencies", "OpenCV not available")
                return
            H = getattr(self._camera, 'translation_matrix', None)
            if not isinstance(H, _np.ndarray) or H.shape != (3, 3):
                QMessageBox.warning(dlg, "Calibration", "No homography available. Run Calibrate first.")
                return
            cam_w, cam_h = _infer_cam_size()
            try:
                _cell = max(1, int(sp_cell.value()))
            except Exception:
                _cell = 16
            try:
                _pitch = max(_cell, int(sp_pitch.value()))
            except Exception:
                _pitch = _cell
            grid = _make_cam_grid(cam_w, cam_h, cell=_cell, pitch=_pitch)
            img = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)
            # Ensure local projector window exists and use H path (no LUT)
            if not self._ensure_projection():
                # Fallback: show warped preview inside troubleshooting
                try:
                    h, w = img.shape[:2]
                    prev = cv2.warpPerspective(img, H.astype(_np.float64), (w, h))
                    pm = _to_pix(prev); sl_pix.setPixmap(pm); sl_view.fitInView(sl_pix, Qt.KeepAspectRatio)
                except Exception:
                    QMessageBox.warning(dlg, "Projection", "Projection window unavailable")
                return
            try:
                self.projection.show_image_fullscreen_on_second_monitor(img, H)
            except Exception as e:
                # Also show preview in troubleshooting for confirmation
                try:
                    h, w = img.shape[:2]
                    prev = cv2.warpPerspective(img, H.astype(_np.float64), (w, h))
                    pm = _to_pix(prev); sl_pix.setPixmap(pm); sl_view.fitInView(sl_pix, Qt.KeepAspectRatio)
                except Exception:
                    pass
                QMessageBox.warning(dlg, "Projection", str(e))

        # Hold last H evaluation images for mode switching
        _h_last_grid = {'img': None}
        _h_last_cap = {'img': None}
        _h_last_overlap = {'img': None}
        # Track whether we've fitted the view once for this set; preserves zoom on toggles
        _h_view_fit = {'done': False}

        # Crosstalk metric: mean/p95 of neighbor(off)/on intensities across pitched grid
        def _compute_crosstalk(cap_gray, cell, pitch):
            try:
                import numpy as _np
            except Exception:
                return None
            if cap_gray is None or getattr(cap_gray, 'ndim', 0) != 2:
                return None
            h, w = cap_gray.shape
            cell = int(max(1, int(cell)))
            pitch = int(max(cell, int(pitch)))
            img = cap_gray.astype(_np.float32)
            ratios = []
            on_means = []
            off_means = []
            for y0 in range(0, h - cell + 1, pitch):
                for x0 in range(0, w - cell + 1, pitch):
                    if ((x0 // pitch) + (y0 // pitch)) & 1:
                        on_roi = img[y0:y0+cell, x0:x0+cell]
                        on_mean = float(on_roi.mean())
                        if on_mean <= 1e-6:
                            continue
                        for dx, dy in ((pitch,0),(-pitch,0),(0,pitch),(0,-pitch)):
                            xn = x0 + dx; yn = y0 + dy
                            if xn < 0 or yn < 0 or xn + cell > w or yn + cell > h:
                                continue
                            off_roi = img[yn:yn+cell, xn:xn+cell]
                            off_mean = float(off_roi.mean())
                            ratios.append(off_mean / on_mean)
                            on_means.append(on_mean)
                            off_means.append(off_mean)
            if not ratios:
                return None
            ratios = _np.array(ratios, dtype=_np.float32)
            return {
                'ratio_mean': float(_np.mean(ratios)),
                'ratio_p95': float(_np.percentile(ratios, 95)),
                'samples': int(ratios.size),
                'on_mean': float(_np.mean(on_means)) if on_means else 0.0,
                'off_mean': float(_np.mean(off_means)) if off_means else 0.0
            }

        def _update_h_preview(mode: str):
            src = None
            if mode == 'ref' and _h_last_grid['img'] is not None:
                src = _h_last_grid['img']
            elif mode == 'cap' and _h_last_cap['img'] is not None:
                src = _h_last_cap['img']
            elif mode == 'ov' and _h_last_overlap['img'] is not None:
                src = _h_last_overlap['img']
            if src is not None:
                try:
                    pm = _to_pix(src)
                    sl_pix.setPixmap(pm)
                    if not _h_view_fit['done']:
                        sl_view.fitInView(sl_pix, Qt.KeepAspectRatio)
                        _h_view_fit['done'] = True
                except Exception:
                    pass

        def _on_h_capture_eval():
            try:
                import cv2, numpy as _np, time as _t
            except Exception:
                QMessageBox.warning(dlg, "Dependencies", "OpenCV not available")
                return
            H = getattr(self._camera, 'translation_matrix', None)
            if not isinstance(H, _np.ndarray) or H.shape != (3, 3):
                QMessageBox.warning(dlg, "Calibration", "No homography available. Run Calibrate first.")
                return
            cam_w, cam_h = _infer_cam_size()
            try:
                _cell = max(1, int(sp_cell.value()))
            except Exception:
                _cell = 16
            try:
                _pitch = max(_cell, int(sp_pitch.value()))
            except Exception:
                _pitch = _cell
            grid = _make_cam_grid(cam_w, cam_h, cell=_cell, pitch=_pitch)
            img = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)
            if self._ensure_projection():
                try:
                    self.projection.show_image_fullscreen_on_second_monitor(img, H)
                    _t.sleep(0.15)
                except Exception:
                    pass
            cap = _capture_gray()
            if cap is None:
                QMessageBox.warning(dlg, "Capture Failed", "No camera snapshot available")
                return
            if cap.shape != grid.shape:
                try:
                    cap = cv2.resize(cap, (grid.shape[1], grid.shape[0]), interpolation=cv2.INTER_AREA)
                except Exception:
                    pass
            # Crosstalk (report in textbox, not overlay)
            try:
                ctk = _compute_crosstalk(cap, _cell, _pitch)
                if ctk:
                    metrics_box.appendPlainText(
                        f"Crosstalk (H): cell={_cell}px, pitch={_pitch}px -> mean={ctk['ratio_mean']*100:.1f}%, "
                        f"p95={ctk['ratio_p95']*100:.1f}% (N={ctk['samples']})"
                    )
            except Exception as _e:
                try:
                    metrics_box.appendPlainText(f"Crosstalk (H) error: {_e}")
                except Exception:
                    pass
            # Threshold to binary masks
            try:
                _, cap_bin = cv2.threshold(cap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            except Exception:
                cap_bin = (cap > 128).astype(_np.uint8) * 255
            grid_bin = (grid > 128).astype(_np.uint8) * 255
            diff = (cap_bin.astype(_np.int16) - grid_bin.astype(_np.int16)).astype(_np.float32)
            mse = float(_np.mean((diff/255.0)**2)) * (255.0*255.0)
            psnr = 99.0 if mse <= 1e-9 else float(10.0 * _np.log10((255.0*255.0)/mse))
            # Build color-coded overlap: green where both 1, red where mismatch, black elsewhere
            both = ((cap_bin == 255) & (grid_bin == 255))
            xor  = ((cap_bin == 255) ^ (grid_bin == 255))
            vis = _np.zeros((cam_h, cam_w, 3), _np.uint8)
            vis[both] = (0, 255, 0)      # green (BGR)
            vis[xor]  = (0, 0, 255)      # red (BGR)
            try:
                _h_last_grid['img'] = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)
                _h_last_cap['img']  = cv2.cvtColor(_np.clip(cap, 0, 255).astype(_np.uint8), cv2.COLOR_GRAY2BGR)
                _h_last_overlap['img'] = vis
                # Reset fit for new images; subsequent toggles preserve zoom
                _h_view_fit['done'] = False
                _update_h_preview('ov')
            except Exception:
                pass

        def _on_h_dot_array_test():
            try:
                import cv2, numpy as _np, time as _t
            except Exception:
                QMessageBox.warning(dlg, "Dependencies", "OpenCV not available")
                return
            H = getattr(self._camera, 'translation_matrix', None)
            if not isinstance(H, _np.ndarray) or H.shape != (3, 3):
                QMessageBox.warning(dlg, "Calibration", "No homography available. Run Calibrate first.")
                return
            cam_w, cam_h = _infer_cam_size()
            try:
                pitch = max(1, int(sp_cell.value()))
            except Exception:
                pitch = 16
            # Build dot array in camera space
            ref = _np.zeros((cam_h, cam_w), _np.uint8)
            # Choose a conservative radius relative to pitch
            radius = max(2, int(round(pitch * 0.18)))
            try:
                for y in range(radius + 1, cam_h - radius - 1, pitch):
                    for x in range(radius + 1, cam_w - radius - 1, pitch):
                        cv2.circle(ref, (int(x), int(y)), radius, 255, thickness=-1)
            except Exception:
                # Fallback: sparse centers without cv2
                ref[::pitch, ::pitch] = 255
            img = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
            if self._ensure_projection():
                try:
                    self.projection.show_image_fullscreen_on_second_monitor(img, H)
                    _t.sleep(0.15)
                except Exception:
                    pass
            cap = _capture_gray()
            if cap is None:
                QMessageBox.warning(dlg, "Capture Failed", "No camera snapshot available")
                return
            if cap.shape != ref.shape:
                try:
                    cap = cv2.resize(cap, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
                except Exception:
                    pass
            # Threshold both
            try:
                _, cap_bin = cv2.threshold(cap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            except Exception:
                cap_bin = (cap > 128).astype(_np.uint8) * 255
            ref_bin = (ref > 128).astype(_np.uint8) * 255
            # Compute simple metrics
            diff = (cap_bin.astype(_np.int16) - ref_bin.astype(_np.int16)).astype(_np.float32)
            mse = float(_np.mean((diff/255.0)**2)) * (255.0*255.0)
            psnr = 99.0 if mse <= 1e-9 else float(10.0 * _np.log10((255.0*255.0)/mse))
            # Overlap viz
            both = ((cap_bin == 255) & (ref_bin == 255))
            xor  = ((cap_bin == 255) ^ (ref_bin == 255))
            vis = _np.zeros((cam_h, cam_w, 3), _np.uint8)
            vis[both] = (0, 255, 0)
            vis[xor]  = (0, 0, 255)
            try:
                _h_last_grid['img'] = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
                _h_last_cap['img']  = cv2.cvtColor(_np.clip(cap, 0, 255).astype(_np.uint8), cv2.COLOR_GRAY2BGR)
                _h_last_overlap['img'] = vis
                _h_view_fit['done'] = False
                _update_h_preview('ov')
            except Exception:
                pass

        btn_h_proj.clicked.connect(_on_h_project_grid)
        btn_h_eval.clicked.connect(_on_h_capture_eval)
        btn_h_dots.clicked.connect(_on_h_dot_array_test)

        # H view mode (Reference / Captured / Overlap)
        try:
            from PyQt5.QtWidgets import QHBoxLayout as _QHBox, QRadioButton as _QRB, QButtonGroup as _QBG
            mode_row = _QHBox()
            mode_row.addWidget(QLabel("View:"))
            rb_ref = _QRB("Reference")
            rb_cap = _QRB("Captured")
            rb_ov  = _QRB("Overlap")
            rb_ov.setChecked(True)
            bg = _QBG(dlg)
            bg.addButton(rb_ref); bg.addButton(rb_cap); bg.addButton(rb_ov)
            mode_row.addWidget(rb_ref); mode_row.addWidget(rb_cap); mode_row.addWidget(rb_ov)
            # Legend
            leg = QLabel("Legend: \nGreen=overlap, Red=mismatch")
            try: leg.setStyleSheet("color:#1c1c1e;")
            except Exception: pass
            mode_row.addSpacing(12); mode_row.addWidget(leg)
            lay.addLayout(mode_row)
            def _on_mode_change():
                if rb_ref.isChecked():
                    _update_h_preview('ref')
                elif rb_cap.isChecked():
                    _update_h_preview('cap')
                else:
                    _update_h_preview('ov')
            rb_ref.toggled.connect(_on_mode_change)
            rb_cap.toggled.connect(_on_mode_change)
            rb_ov.toggled.connect(_on_mode_change)
        except Exception:
            pass

        def _on_calib_char():
            try:
                import numpy as _np, cv2
                from scipy.spatial import cKDTree
            except Exception as e:
                QMessageBox.warning(dlg, "Dependencies", f"Missing scipy or cv2: {e}")
                return
            try:
                # Build camera grid points
                cam_w, cam_h = _infer_cam_size()
                cell = 64
                pts = []
                for y in range(cell//2, cam_h, cell):
                    for x in range(cell//2, cam_w, cell):
                        pts.append([x, y, 1.0])
                P = _np.array(pts, dtype=_np.float64).T  # 3xN
                # Load H (camera->projector)
                H = getattr(self._camera, 'translation_matrix', None)
                if not isinstance(H, _np.ndarray) or H.shape != (3,3):
                    try:
                        from pathlib import Path as _P
                        npy = (_P(__file__).resolve().parent / 'Assets' / 'Generated' / 'homography_cam2proj.npy').as_posix()
                        if os.path.isfile(npy):
                            H = _np.load(npy)
                    except Exception:
                        H = None
                if H is None:
                    QMessageBox.warning(dlg, "Calibration", "No homography available. Run Calibrate first.")
                    return
                # Map to projector space
                X = H @ P; X /= _np.clip(X[2:3, :], 1e-9, None)
                proj_xy = X[:2, :].T
                # Ideal projector grid
                try:
                    proj_w = int(getattr(self, '_proj_w', 1920))
                    proj_h = int(getattr(self, '_proj_h', 1080))
                except Exception:
                    proj_w, proj_h = 1920, 1080
                gx = _np.arange(cell//2, proj_w, cell)
                gy = _np.arange(cell//2, proj_h, cell)
                grid_xy = _np.stack(_np.meshgrid(gx, gy), axis=-1).reshape(-1, 2)
                # Nearest neighbor errors
                try:
                    tree = cKDTree(grid_xy)
                    dists, _ = tree.query(proj_xy, k=1)
                except Exception:
                    dists = _np.linalg.norm(proj_xy[:, None, :] - grid_xy[None, :, :], axis=2).min(axis=1)
                rmse = float(_np.sqrt(_np.mean(dists**2))) if dists.size else 0.0
                # Visualization
                vis = _np.zeros((proj_h, proj_w, 3), _np.uint8)
                for y in range(cell//2, proj_h, cell):
                    cv2.line(vis, (0, y), (proj_w-1, y), (64,64,64), 1)
                for x in range(cell//2, proj_w, cell):
                    cv2.line(vis, (x, 0), (x, proj_h-1), (64,64,64), 1)
                for (x, y) in proj_xy.astype(_np.int32):
                    if 0 <= x < proj_w and 0 <= y < proj_h:
                        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 255), -1)
                pm = _to_pix(vis); sl_pix.setPixmap(pm); sl_view.fitInView(sl_pix, Qt.KeepAspectRatio)
            except Exception as e:
                QMessageBox.critical(dlg, "Calibration Characterization", str(e))

        def _on_capture_evaluate():
            # Structured-light LUT: project prewarped grid, capture, and overlap
            try:
                from calibration import prewarp_with_inverse_lut as _prewarp
            except Exception:
                _prewarp = None
            inv_x, inv_y, _ = _load_luts()
            if inv_x is None or _prewarp is None:
                QMessageBox.warning(dlg, "LUT Missing", "cam_from_proj LUTs not available. Run SL calibration first.")
                return
            # Build grid with chosen cell
            cam_w, cam_h = _infer_cam_size()
            try:
                _cell = max(1, int(sp_cell.value()))
            except Exception:
                _cell = 16
            try:
                _pitch = max(_cell, int(sp_pitch.value()))
            except Exception:
                _pitch = _cell
            grid = _make_cam_grid(cam_w, cam_h, cell=_cell, pitch=_pitch)
            grid_rgb = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)
            proj_h, proj_w = inv_x.shape
            warped = _prewarp(grid_rgb, inv_x, inv_y, proj_w, proj_h)
            # Try to project via engine; fallback to local window
            sent = _send_to_engine_gray(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))
            if not sent:
                try:
                    if not self._ensure_projection():
                        raise RuntimeError("Projection window unavailable")
                    self.projection.show_image_raw_no_warp_no_flip(warped)
                except Exception:
                    pass
            # Short wait and capture
            try:
                import time as _t
                _t.sleep(0.15)
            except Exception:
                pass
            cap = _capture_gray()
            if cap is None:
                QMessageBox.warning(dlg, "Capture Failed", "Could not read snapshot.")
                return
            if cap.shape[:2] != (cam_h, cam_w):
                try:
                    cap = cv2.resize(cap, (cam_w, cam_h), interpolation=cv2.INTER_AREA)
                except Exception:
                    pass
            # Crosstalk (report to textbox)
            try:
                ctk = _compute_crosstalk(cap, _cell, _pitch)
                if ctk:
                    metrics_box.appendPlainText(
                        f"Crosstalk (LUT): cell={_cell}px, pitch={_pitch}px -> mean={ctk['ratio_mean']*100:.1f}%, "
                        f"p95={ctk['ratio_p95']*100:.1f}% (N={ctk['samples']})"
                    )
            except Exception as _e:
                try:
                    metrics_box.appendPlainText(f"Crosstalk (LUT) error: {_e}")
                except Exception:
                    pass
            # Build binary masks and overlap vis
            try:
                _, cap_bin = cv2.threshold(cap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            except Exception:
                cap_bin = (cap > 128).astype(_np.uint8) * 255
            grid_bin = (grid > 128).astype(_np.uint8) * 255
            both = ((cap_bin == 255) & (grid_bin == 255))
            xor  = ((cap_bin == 255) ^ (grid_bin == 255))
            vis = _np.zeros((cam_h, cam_w, 3), _np.uint8)
            vis[both] = (0, 255, 0)
            vis[xor]  = (0, 0, 255)
            diff = (cap_bin.astype(_np.int16) - grid_bin.astype(_np.int16)).astype(_np.float32)
            mse = float(_np.mean((diff/255.0)**2)) * (255.0*255.0)
            psnr = 99.0 if mse <= 1e-9 else float(10.0 * _np.log10((255.0*255.0)/mse))
            # Update preview with overlap and store ref/cap for view toggles
            try:
                _h_last_grid['img'] = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)
                _h_last_cap['img']  = cv2.cvtColor(_np.clip(cap, 0, 255).astype(_np.uint8), cv2.COLOR_GRAY2BGR)
                _h_last_overlap['img'] = vis
                # Preserve current zoom on toggles; fit only once for new set
                _h_view_fit = {'done': False}
                pm = _to_pix(vis)
                sl_pix.setPixmap(pm)
                sl_view.fitInView(sl_pix, Qt.KeepAspectRatio)
                _h_view_fit['done'] = True
            except Exception:
                pass

        def _on_round_trip():
            try:
                asset_dir = getattr(self._camera, 'asset_dir', str((Path(__file__).resolve().parent / "Assets" / "Generated").resolve()))
                fpx = os.path.join(asset_dir, "proj_from_cam_x.npy")
                fpy = os.path.join(asset_dir, "proj_from_cam_y.npy")
                inv_x, inv_y, _ = _load_luts()
                if inv_x is None or (not (os.path.isfile(fpx) and os.path.isfile(fpy))):
                    QMessageBox.warning(dlg, "Missing Maps", "Need proj_from_cam and cam_from_proj maps.")
                    return
                fx = _np.load(fpx).astype(_np.float32); fy = _np.load(fpy).astype(_np.float32)
                cam_h, cam_w = fx.shape
                step = max(4, min(cam_w, cam_h)//200)
                ys = _np.arange(0, cam_h, step, dtype=_np.int32)
                xs = _np.arange(0, cam_w, step, dtype=_np.int32)
                yy, xx = _np.meshgrid(ys, xs, indexing='ij')
                px = fx[yy, xx]; py = fy[yy, xx]
                ph, pw = inv_x.shape
                x0 = _np.floor(px).astype(_np.int32); y0 = _np.floor(py).astype(_np.int32)
                dx = px - x0; dy = py - y0
                x1 = _np.clip(x0+1, 0, pw-1); y1 = _np.clip(y0+1, 0, ph-1)
                def _bil(inmap):
                    v00 = inmap[_np.clip(y0,0,ph-1), _np.clip(x0,0,pw-1)]
                    v10 = inmap[y0, x1]; v01 = inmap[y1, x0]; v11 = inmap[y1, x1]
                    return (1-dx)*(1-dy)*v00 + dx*(1-dy)*v10 + (1-dx)*dy*v01 + dx*dy*v11
                rx = _bil(inv_x); ry = _bil(inv_y)
                err = _np.sqrt((_np.maximum(0, rx) - xx.astype(_np.float32))**2 + (_np.maximum(0, ry) - yy.astype(_np.float32))**2)
                mean_err = float(_np.mean(err[_np.isfinite(err)]))
                p95_err = float(_np.percentile(err[_np.isfinite(err)], 95))
                QMessageBox.information(dlg, "Round-Trip Error", f"Mean error: {mean_err:.2f} px\n95th %: {p95_err:.2f} px")
            except Exception as e:
                QMessageBox.warning(dlg, "Round-Trip Error", str(e))

        btn_diag.clicked.connect(_on_lut_diag)
        btn_proj.clicked.connect(_on_project_grid)
        btn_eval.clicked.connect(_on_capture_evaluate)
        btn_rterr.clicked.connect(_on_round_trip)
        btn_calib_char.clicked.connect(_on_calib_char)

        def _send_to_engine_gray(img_gray):
            try:
                from projector_client import ProjectorClient
                client = ProjectorClient()
                client.send_gray(img_gray, frame_id=8888, visible_id=0, immediate=True)
                client.close()
                return True
            except Exception:
                return False

        def _capture_gray():
            # Prefer RAM-backed path to avoid heavy disk I/O during probes
            try:
                tmp_dir = "/dev/shm"
                if os.path.isdir(tmp_dir) and os.access(tmp_dir, os.W_OK):
                    cap_path = os.path.join(tmp_dir, "sl_validation_cap.png")
                else:
                    save_dir = getattr(self._camera, 'save_dir', './Saved_Media')
                    os.makedirs(save_dir, exist_ok=True)
                    cap_path = os.path.join(save_dir, "sl_validation_cap.png")
            except Exception:
                save_dir = getattr(self._camera, 'save_dir', './Saved_Media')
                os.makedirs(save_dir, exist_ok=True)
                cap_path = os.path.join(save_dir, "sl_validation_cap.png")
            self._camera.snapshot(cap_path)
            return cv2.imread(cap_path, cv2.IMREAD_GRAYSCALE)

        def _on_pixel_probe():
            # Memory-safe pixel probe: avoid full-frame prewarp per point and reuse client/buffers
            # Uses forward LUT to place a subpixel dot in projector space via bilinear weights
            try:
                asset_dir = getattr(self._camera, 'asset_dir', str((Path(__file__).resolve().parent / "Assets" / "Generated").resolve()))
                fpx = os.path.join(asset_dir, "proj_from_cam_x.npy")
                fpy = os.path.join(asset_dir, "proj_from_cam_y.npy")
                fx = _np.load(fpx).astype(_np.float32)
                fy = _np.load(fpy).astype(_np.float32)
            except Exception as e:
                QMessageBox.warning(dlg, "Missing Maps", f"Need proj_from_cam_{'{x,y}'} maps: {e}")
                return
            inv_x, inv_y, _ = _load_luts()
            if inv_x is None:
                return
            proj_h, proj_w = inv_x.shape
            cam_w, cam_h = fx.shape[1], fx.shape[0]
            step = max(96, min(cam_w, cam_h)//12)
            points = [(x, y) for y in range(step//2, cam_h, step) for x in range(step//2, cam_w, step)]
            # Limit total samples aggressively to avoid overloading system
            try:
                max_samples = 40
                if len(points) > max_samples:
                    stride = int(_np.ceil(len(points) / float(max_samples)))
                    points = points[::max(1, stride)]
            except Exception:
                pass
            # Preallocate projector-space grayscale buffer
            proj_img = _np.zeros((proj_h, proj_w), _np.uint8)
            vis = _np.zeros((cam_h, cam_w, 3), _np.uint8)
            errors = []
            # Reuse ZMQ client if available
            client = None
            try:
                from projector_client import ProjectorClient
                client = ProjectorClient()
            except Exception:
                client = None
            # Optional progress dialog
            try:
                from PyQt5.QtWidgets import QProgressDialog
                prog = QProgressDialog("Probing pixels…", "Cancel", 0, len(points), dlg)
                prog.setWindowModality(Qt.WindowModal)
                prog.setAutoClose(False)
                prog.setAutoReset(False)
                prog.show()
            except Exception:
                prog = None
            import gc as _gc, time as _t
            from PyQt5.QtWidgets import QApplication as _QApp
            t_start = _t.time()
            consecutive_fail = 0
            for i, (x0, y0) in enumerate(points):
                # Hard overall time cap (e.g., ~8s)
                if (_t.time() - t_start) > 8.0:
                    break
                # Early cancel check to keep UI responsive
                if prog is not None:
                    try:
                        if prog.wasCanceled():
                            break
                    except Exception:
                        pass
                # Build sparse subpixel dot in projector space using forward LUT
                px = float(fx[y0, x0]); py = float(fy[y0, x0])
                if not _np.isfinite(px) or not _np.isfinite(py):
                    continue
                if px < 0 or py < 0 or px > (proj_w - 1.001) or py > (proj_h - 1.001):
                    continue
                xz = int(_np.floor(px)); yz = int(_np.floor(py))
                dx = px - xz; dy = py - yz
                xz1 = min(proj_w - 1, xz + 1); yz1 = min(proj_h - 1, yz + 1)
                # Clear buffer and write four bilinear weights scaled to 255
                proj_img.fill(0)
                w00 = (1.0 - dx) * (1.0 - dy)
                w10 = dx * (1.0 - dy)
                w01 = (1.0 - dx) * dy
                w11 = dx * dy
                proj_img[yz,  xz ] = int(255.0 * w00)
                proj_img[yz,  xz1] = int(255.0 * w10)
                proj_img[yz1, xz ] = int(255.0 * w01)
                proj_img[yz1, xz1] = int(255.0 * w11)
                # Send to engine (reuse client) or fallback to Qt projector
                sent = False
                if client is not None:
                    try:
                        client.send_gray(proj_img, frame_id=8888, visible_id=0, immediate=True)
                        sent = True
                    except Exception:
                        sent = False
                if not sent:
                    try:
                        self.projection.show_image_raw_no_warp_no_flip(cv2.cvtColor(proj_img, cv2.COLOR_GRAY2BGR))
                    except Exception:
                        try:
                            self.projection.show_image_fullscreen_on_second_monitor(cv2.cvtColor(proj_img, cv2.COLOR_GRAY2BGR), None)
                        except Exception:
                            pass
                # Allow a short time for the projector to present the dot
                try:
                    _t.sleep(0.02)
                except Exception:
                    pass
                # Capture and compute subpixel center near (x0,y0)
                cap = _capture_gray()
                if cap is None:
                    consecutive_fail += 1
                    if consecutive_fail >= 20:
                        break
                    continue
                x1 = max(0, x0 - 4); x2 = min(cam_w, x0 + 5)
                y1 = max(0, y0 - 4); y2 = min(cam_h, y0 + 5)
                roi = cap[y1:y2, x1:x2].astype(_np.float32)
                if roi.size == 0:
                    consecutive_fail += 1
                    if consecutive_fail >= 20:
                        break
                    continue
                yy, xx = _np.mgrid[y1:y2, x1:x2]
                w = _np.maximum(0.0, roi - roi.mean())
                # Require sufficient local signal; skip if no visible dot
                amp = float(roi.max() - roi.mean())
                if not _np.isfinite(amp) or amp < 25.0 or w.sum() <= 1e-3:
                    consecutive_fail += 1
                    if consecutive_fail >= 20:
                        break
                    continue
                s = w.sum()
                cx = float((w * xx).sum() / s); cy = float((w * yy).sum() / s)
                errors.append(_np.hypot(cx - x0, cy - y0))
                consecutive_fail = 0
                cv2.circle(vis, (int(cx), int(cy)), 2, (0,255,0), -1)
                cv2.arrowedLine(vis, (x0, y0), (int(cx), int(cy)), (0,255,255), 1, tipLength=0.3)
                # UI/progress and periodic GC to keep memory in check
                if prog is not None:
                    try:
                        prog.setValue(i + 1)
                        _QApp.processEvents()
                        if prog.wasCanceled():
                            break
                    except Exception:
                        pass
                if (i & 7) == 7:
                    try: _gc.collect()
                    except Exception: pass
                # Small throttle to reduce CPU pressure
                try: _t.sleep(0.002)
                except Exception: pass
            try:
                if client is not None:
                    client.close()
            except Exception:
                pass
            if errors:
                mean_err = float(_np.mean(errors)); p95 = float(_np.percentile(errors, 95))
                QMessageBox.information(dlg, "Pixel Probe", f"Samples: {len(errors)}\nMean: {mean_err:.2f} px\n95th %: {p95:.2f} px")
                try:
                    pm = _to_pix(vis)
                    sl_pix.setPixmap(pm)
                    sl_view.fitInView(sl_pix, Qt.KeepAspectRatio)
                except Exception:
                    pass

        def _on_dot_array():
            try:
                from calibration import prewarp_with_inverse_lut as _prewarp
            except Exception:
                QMessageBox.warning(dlg, "Missing", "prewarp not available")
                return
            inv_x, inv_y, _ = _load_luts()
            if inv_x is None:
                return
            cam_w, cam_h = _infer_cam_size()
            spacing = max(24, min(cam_w, cam_h)//24)
            dot_r = 3
            img = _np.zeros((cam_h, cam_w), _np.uint8)
            pts = []
            for y in range(spacing//2, cam_h, spacing):
                for x in range(spacing//2, cam_w, spacing):
                    cv2.circle(img, (x,y), dot_r, 255, -1); pts.append((x,y))
            proj_h, proj_w = inv_x.shape
            warped = _prewarp(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), inv_x, inv_y, proj_w, proj_h)
            sent = _send_to_engine_gray(warped)
            if not sent:
                try:
                    self.projection.show_image_raw_no_warp_no_flip(warped)
                except Exception:
                    self.projection.show_image_fullscreen_on_second_monitor(warped, None)
            cap = _capture_gray()
            if cap is None:
                return
            # Threshold and find blobs
            _, bw = cv2.threshold(cap, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            num, labels, stats, cent = cv2.connectedComponentsWithStats(bw, connectivity=8)
            centers = cent[1:, :] if num>1 else _np.zeros((0,2), _np.float32)
            used = _np.zeros(len(centers), dtype=bool)
            errors = []
            overlay = cv2.cvtColor(cap, cv2.COLOR_GRAY2BGR)
            for (x,y) in pts:
                # find nearest center
                if centers.shape[0]==0:
                    continue
                d2 = _np.sum((centers - _np.array([[x,y]], _np.float32))**2, axis=1)
                idx = int(_np.argmin(d2))
                c = centers[idx]
                if used[idx]:
                    continue
                used[idx] = True
                err = float(_np.hypot(c[0]-x, c[1]-y))
                errors.append(err)
                cv2.circle(overlay, (int(c[0]), int(c[1])), 3, (0,255,0), -1)
                cv2.arrowedLine(overlay, (x,y), (int(c[0]), int(c[1])), (0,255,255), 1, tipLength=0.3)
            if errors:
                mean_err = float(_np.mean(errors)); p95 = float(_np.percentile(errors, 95))
                QMessageBox.information(dlg, "Dot Array", f"Samples: {len(errors)}\nMean: {mean_err:.2f} px\n95th %: {p95:.2f} px")
                try:
                    pm = _to_pix(overlay)
                    sl_pix.setPixmap(pm)
                    sl_view.fitInView(sl_pix, Qt.KeepAspectRatio)
                except Exception:
                    pass

        def _on_round_trip_physical():
            try:
                from calibration import prewarp_with_inverse_lut as _prewarp
            except Exception:
                QMessageBox.warning(dlg, "Missing", "prewarp not available")
                return
            inv_x, inv_y, _ = _load_luts()
            if inv_x is None:
                return
            cam_w, cam_h = _infer_cam_size()
            grid = _make_cam_grid(cam_w, cam_h)
            proj_h, proj_w = inv_x.shape
            warped = _prewarp(cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR), inv_x, inv_y, proj_w, proj_h)
            sent = _send_to_engine_gray(warped)
            if not sent:
                try:
                    self.projection.show_image_raw_no_warp_no_flip(warped)
                except Exception:
                    self.projection.show_image_fullscreen_on_second_monitor(warped, None)
            cap = _capture_gray()
            if cap is None:
                return
            # Map the captured camera image into projector space with inv LUT and compare to warped(gray)
            cap_bgr = cv2.cvtColor(cap, cv2.COLOR_GRAY2BGR)
            pred = cv2.remap(cap_bgr, inv_x, inv_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            pred_gray = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
            diff = (warped_gray.astype(_np.float32) - pred_gray.astype(_np.float32))
            mse = float(_np.mean(diff*diff)); psnr = 99.0 if mse<=1e-9 else 10.0*_np.log10((255.0*255.0)/mse)
            QMessageBox.information(dlg, "Round-Trip (Physical)", f"MSE: {mse:.1f}\nPSNR: {psnr:.2f} dB")
            try:
                pm = _to_pix(cv2.cvtColor(pred_gray, cv2.COLOR_GRAY2BGR))
                sl_pix.setPixmap(pm)
                sl_view.fitInView(sl_pix, Qt.KeepAspectRatio)
            except Exception:
                pass

        def _on_edge_strip():
            try:
                from calibration import prewarp_with_inverse_lut as _prewarp
            except Exception:
                QMessageBox.warning(dlg, "Missing", "prewarp not available")
                return
            inv_x, inv_y, _ = _load_luts()
            if inv_x is None:
                return
            cam_w, cam_h = _infer_cam_size()
            positions = [int(cam_w*0.25), int(cam_w*0.5), int(cam_w*0.75)]
            img = _np.zeros((cam_h, cam_w), _np.uint8)
            for x in positions:
                img[:, max(0, x-1):min(cam_w, x+1)] = 255
            proj_h, proj_w = inv_x.shape
            warped = _prewarp(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), inv_x, inv_y, proj_w, proj_h)
            sent = _send_to_engine_gray(warped)
            if not sent:
                try:
                    self.projection.show_image_raw_no_warp_no_flip(warped)
                except Exception:
                    self.projection.show_image_fullscreen_on_second_monitor(warped, None)
            cap = _capture_gray()
            if cap is None:
                return
            errs = []
            for x0 in positions:
                x1 = max(0, x0-20); x2 = min(cam_w, x0+21)
                roi = cap[:, x1:x2].astype(_np.float32)
                gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
                prof = _np.mean(_np.abs(gx), axis=0)
                # subpixel via quadratic fit around peak
                i = int(_np.argmax(prof))
                i0 = max(1, min(len(prof)-2, i))
                y1 = prof[i0-1]; y2 = prof[i0]; y3 = prof[i0+1]
                denom = (y1 - 2*y2 + y3)
                delta = 0.0 if abs(denom) < 1e-6 else 0.5 * (y1 - y3) / denom
                xpos = x1 + i0 + delta
                errs.append(abs(xpos - x0))
            if errs:
                mean_err = float(_np.mean(errs)); p95 = float(_np.percentile(errs, 95))
                QMessageBox.information(dlg, "Edge Strip", f"Lines: {len(errs)}\nMean: {mean_err:.2f} px\n95th %: {p95:.2f} px")

        btn_probe.clicked.connect(_on_pixel_probe)
        btn_dots.clicked.connect(_on_dot_array)
        btn_rtphy.clicked.connect(_on_round_trip_physical)
        btn_edge.clicked.connect(_on_edge_strip)

        # State for monitors
        from collections import deque
        cpu_hist = deque(maxlen=120)
        mem_hist = deque(maxlen=120)
        gpu_hist = deque(maxlen=120)
        trig_times = deque(maxlen=200)
        last_pidx = [0]
        running = {"engine": False}

        # GPU via NVML if available
        _HAS_NVML = False
        try:
            import pynvml
            pynvml.nvmlInit()
            _nvdev = pynvml.nvmlDeviceGetHandleByIndex(0)
            _HAS_NVML = True
        except Exception:
            _HAS_NVML = False

        def _sample_perf():
            try:
                cpu_hist.append(psutil.cpu_percent(interval=None))
                mem_hist.append(psutil.virtual_memory().percent)
            except Exception:
                cpu_hist.append(0.0)
                mem_hist.append(0.0)
            if _HAS_NVML:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(_nvdev)
                    gpu_hist.append(float(util.gpu))
                except Exception:
                    gpu_hist.append(0.0)
            else:
                # Fallback: Tegra GPU load from sysfs (0-255 -> %)
                try:
                    with open("/sys/devices/gpu.0/load", "r") as f:
                        val = f.read().strip()
                        v = float(val) if val else 0.0
                        gpu_hist.append(min(100.0, max(0.0, (v / 255.0) * 100.0)))
                except Exception:
                    gpu_hist.append(0.0)
            if _HAS_PG:
                cpu_curve.setData(list(range(len(cpu_hist))), list(cpu_hist))
                mem_curve.setData(list(range(len(mem_hist))), list(mem_hist))
                gpu_curve.setData(list(range(len(gpu_hist))), list(gpu_hist))
            else:
                try:
                    lbl_cpu.setText(f"CPU: {cpu_hist[-1]:.1f} %")
                    lbl_mem.setText(f"Mem: {mem_hist[-1]:.1f} %")
                    lbl_gpu.setText(f"GPU: {gpu_hist[-1]:.1f} %")
                except Exception:
                    pass

        # Engine subscriber thread
        last_event_ts = {"t": 0.0}
        engine_status = {"text": "idle"}

        def _set_indicator(on: bool):
            try:
                if on:
                    ind_btn.setText("Projector Trigger: ON")
                    ind_btn.setStyleSheet("QPushButton{background-color:#52c41a; color:white; border-radius:6px; padding:4px 8px;}")
                else:
                    ind_btn.setText("Projector Trigger: OFF")
                    ind_btn.setStyleSheet("QPushButton{background-color:#ff4d4f; color:white; border-radius:6px; padding:4px 8px;}")
            except Exception:
                pass

        def _start_engine_sub():
            import threading as _th, zmq as _zmq, json, time as _t
            running["engine"] = True
            engine_status["text"] = "connecting…"
            def _loop():
                try:
                    ctx = _zmq.Context.instance()
                    sub = ctx.socket(_zmq.SUB)
                    sub.setsockopt(_zmq.LINGER, 0)
                    sub.setsockopt_string(_zmq.SUBSCRIBE, "")
                    sub.connect("tcp://127.0.0.1:5562")
                except Exception as e:
                    engine_status["text"] = f"error {e}"
                    running["engine"] = False
                    return
                engine_status["text"] = "monitoring"
                while running["engine"]:
                    try:
                        msg = sub.recv(flags=_zmq.NOBLOCK)
                        d = json.loads(msg.decode('utf-8', errors='ignore'))
                        p = int(d.get('pidx', 0))
                        v = int(d.get('vis_id', -1))
                        if p > last_pidx[0]:
                            last_pidx[0] = p
                            from time import time as now
                            ts = now()
                            trig_times.append(ts)
                            last_event_ts["t"] = ts
                            # UI updated on main thread via timer
                    except Exception:
                        _t.sleep(0.02)
                try:
                    sub.close(0)
                except Exception:
                    pass
                engine_status["text"] = "stopped"
            th = _th.Thread(target=_loop, daemon=True)
            th.start()
            dlg._engine_thread = th

        def _stop_engine_sub():
            running["engine"] = False

        def _toggle_engine_monitor(checked: bool):
            if checked:
                btn_mon.setText("Stop Engine Monitor")
                _start_engine_sub()
            else:
                btn_mon.setText("Start Engine Monitor")
                _stop_engine_sub()

        btn_mon.toggled.connect(_toggle_engine_monitor)

        # Periodic perf updates and trigger indicator decay
        try:
            from PyQt5.QtCore import QTimer
            tm = QTimer(dlg)
            def _tick():
                _sample_perf()
                # turn indicator OFF if no triggers for 0.5s
                try:
                    import time as _t
                    if running["engine"]:
                        if (_t.time() - last_event_ts.get("t", 0.0)) > 0.5:
                            _set_indicator(False)
                        else:
                            _set_indicator(True)
                    # update engine status and last rate text
                    status_lbl.setText(f"Engine: {engine_status.get('text','')}" )
                    # compute rate over last second for display
                    if trig_times:
                        t1 = trig_times[-1]
                        n = len([t for t in trig_times if t1 - t <= 1.0])
                        last_lbl.setText(f"Last: pidx={last_pidx[0]} vis=? rate={n} Hz")
                except Exception:
                    pass
            tm.timeout.connect(_tick)
            tm.start(1000)
        except Exception:
            pass

        def _on_close():
            try:
                _stop_engine_sub()
            except Exception:
                pass

        try:
            dlg.finished.connect(lambda *_: _on_close())
        except Exception:
            pass

        dlg.show()

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

    def _open_trig_params_dialog(self):
        try:
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QGridLayout, QLabel, QLineEdit, QCheckBox, QPushButton
            dlg = QDialog(self)
            dlg.setWindowTitle("Trigger Parameters")
            try:
                dlg.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.WindowTitleHint | QtCore.Qt.WindowCloseButtonHint)
                dlg.setModal(False)
                dlg.setWindowModality(QtCore.Qt.NonModal)
                dlg.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
            except Exception:
                pass

            lay = QVBoxLayout(dlg)
            grid = QGridLayout()

            # Enable toggles and inputs
            chk_delay = QCheckBox("Enable TriggerDelay (µs)")
            edt_delay = QLineEdit()
            edt_delay.setPlaceholderText("e.g. 0")
            chk_exp = QCheckBox("Enable ExposureTime (µs)")
            edt_exp = QLineEdit()
            edt_exp.setPlaceholderText("e.g. 33333.33")

            # Populate from stored values if present
            try:
                if getattr(self, '_trig_delay_enabled', False):
                    chk_delay.setChecked(True)
                edt_delay.setText(str(getattr(self, '_trig_delay_us', "")))
            except Exception:
                pass
            try:
                if getattr(self, '_trig_exp_enabled', False):
                    chk_exp.setChecked(True)
                edt_exp.setText(str(getattr(self, '_trig_exp_us', "")))
            except Exception:
                pass

            grid.addWidget(chk_delay, 0, 0)
            grid.addWidget(edt_delay, 0, 1)
            grid.addWidget(chk_exp,   1, 0)
            grid.addWidget(edt_exp,   1, 1)

            lay.addLayout(grid)

            btn_apply = QPushButton("Apply")
            btn_close = QPushButton("Close")
            row = QtWidgets.QHBoxLayout()
            row.addStretch(1)
            row.addWidget(btn_apply)
            row.addWidget(btn_close)
            lay.addLayout(row)

            def _apply():
                try:
                    self._trig_delay_enabled = bool(chk_delay.isChecked())
                    self._trig_exp_enabled   = bool(chk_exp.isChecked())
                    # Parse numbers if provided
                    try:
                        self._trig_delay_us = float(edt_delay.text()) if edt_delay.text().strip() else None
                    except Exception:
                        self._trig_delay_us = None
                    try:
                        self._trig_exp_us = float(edt_exp.text()) if edt_exp.text().strip() else None
                    except Exception:
                        self._trig_exp_us = None

                    print(f"[CAM] Trig params set: delay_enabled={self._trig_delay_enabled} delay_us={self._trig_delay_us} exp_enabled={self._trig_exp_enabled} exp_us={self._trig_exp_us}")

                    # If in hardware mode and running, apply immediately
                    try:
                        if getattr(self._camera, 'acquisition_running', False) and getattr(self._camera, 'acquisition_mode', 0) == 1:
                            self._apply_trig_params_to_camera()
                    except Exception:
                        pass
                except Exception as e:
                    print(f"Failed to apply trig params: {e}")

            btn_apply.clicked.connect(_apply)
            btn_close.clicked.connect(dlg.close)

            dlg.show()
        except Exception as e:
            print(f"Failed to open Trigger Parameters dialog: {e}")

    def _apply_trig_params_to_camera(self):
        try:
            nm = getattr(self._camera, 'node_map', None)
            if nm is None:
                return
            # Apply TriggerDelay if enabled and value is valid
            if getattr(self, '_trig_delay_enabled', False) and getattr(self, '_trig_delay_us', None) is not None:
                try:
                    nm.FindNode("TriggerDelay").SetValue(float(self._trig_delay_us))
                    print(f"[CAM] Applied TriggerDelay = {float(self._trig_delay_us)} µs")
                except Exception as e:
                    print(f"[CAM] Failed to set TriggerDelay: {e}")
            # Apply ExposureTime if enabled and value is valid
            if getattr(self, '_trig_exp_enabled', False) and getattr(self, '_trig_exp_us', None) is not None:
                try:
                    nm.FindNode("ExposureAuto").SetCurrentEntry("Off")
                except Exception:
                    pass
                try:
                    nm.FindNode("ExposureTime").SetValue(float(self._trig_exp_us))
                    print(f"[CAM] Applied ExposureTime = {float(self._trig_exp_us)} µs")
                except Exception as e:
                    print(f"[CAM] Failed to set ExposureTime: {e}")
        except Exception:
            pass
    def _on_seq_type_changed(self, text: str):
        try:
            sel = text
            if "0x03" in sel or sel.startswith("8-bit RGB"):
                seq_first = "0x03"
            elif "0x02" in sel or sel.startswith("8-bit Mono"):
                seq_first = "0x02"
            elif "0x00" in sel or sel.startswith("1-bit Mono"):
                seq_first = "0x00"
            else:
                seq_first = "0x01"  # 1-bit RGB
            print(f"[I2C] Sequence type changed: {sel} -> {seq_first}")
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

            # Map sequence type to first byte of pattern_cfg
            # 0x00 = 1-bit Mono, 0x01 = 1-bit RGB, 0x02 = 8-bit Mono, 0x03 = 8-bit RGB
            sel = self._seq_type_dropdown.currentText()
            if "0x03" in sel or sel.startswith("8-bit RGB"):
                seq_first = "0x03"
            elif "0x02" in sel or sel.startswith("8-bit Mono"):
                seq_first = "0x02"
            elif "0x00" in sel or sel.startswith("1-bit Mono"):
                seq_first = "0x00"
            else:
                seq_first = "0x01"  # 1-bit RGB
            print(f"[I2C] Applying sequence mode: {sel} -> {seq_first}")
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
                        "--gradient-steps", "3",
                        "--gradient-hold", "30",
                        "--gradient-gamma", "2.2"
                    ]
                elif pat == "Image":
                    args = ["--pattern", "image", "--image", self._mask_pattern_path]
                elif pat == "Folder":
                    args = ["--pattern", "folder", "--folder", self._mask_pattern_path]
                elif pat == "Seg Mask":
                    # Send latest segmentation labels/masks from rois.npz
                    try:
                        roi_path = str((Path.cwd() / "rois.npz").resolve())
                        args = ["--pattern", "segmask", "--roi-npz", roi_path]
                    except Exception:
                        args = ["--pattern", "segmask", "--roi-npz", "rois.npz"]
                elif pat == "Custom":
                    script_path = self._mask_pattern_path or script_path
                    args = []
                    # If file endswith .py, run with Python; else treat as executable
                    try:
                        if script_path.lower().endswith('.py'):
                            cmd_prog = py
                            cmd_args = [script_path] + args
                            print(f"[MASK] Launch (python): {cmd_prog} {' '.join(cmd_args)}")
                            self._proc_masks.start(cmd_prog, cmd_args)
                        else:
                            from PyQt5.QtCore import QFileInfo
                            fi = QFileInfo(script_path)
                            cmd_prog = fi.absoluteFilePath()
                            print(f"[MASK] Launch (exec): {cmd_prog} {' '.join(args)}")
                            self._proc_masks.start(cmd_prog, args)
                        return
                    except Exception as e:
                        print(f"Custom sender launch failed: {e}")

                # If LUT mode is active, pass prewarp dir
                try:
                    if getattr(self, '_proj_warp_mode', 'H') == 'LUT':
                        asset_dir = getattr(self._camera, 'asset_dir', str((Path(__file__).resolve().parent / "Assets" / "Generated").resolve()))
                        args += ["--prewarp-lut-dir", asset_dir]
                        # Ensure engine H is cleared
                        try:
                            import zmq as _zmq
                            _ctx = _zmq.Context.instance(); _s = _ctx.socket(_zmq.REQ)
                            _s.setsockopt(_zmq.LINGER, 0)
                            _s.connect("tcp://127.0.0.1:5560"); _s.send(b"IDENTITY"); _ = _s.recv(); _s.close()
                        except Exception:
                            pass
                except Exception:
                    pass

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

        self.GUIfps_label = QLabel("Frame rate: 0.00", self)
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
            self.GUIfps_label.setText(f"Frame rate: {fps:.2f}")
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
            
            # Do not force exposure in hardware trigger mode; exposure is controlled by the trigger timing
            
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
            # Apply pending trigger params if enabled
            try:
                self._apply_trig_params_to_camera()
            except Exception:
                pass
        else:
            # Disarm if armed when stopping hardware acquisition
            if getattr(self._camera, "is_armed", False):
                self._camera.disarm_recording()
            
            self._camera.stop_hardware_acquisition()
            self._camera.start_realtime_acquisition()
            
            # Set default exposure to 33333.33 µs
            try:
                if hasattr(self._camera, "set_exposure_us"):
                    self._camera.set_exposure_us(33333.33)
                else:
                    # Fallback: try node map if available
                    nm = getattr(self._camera, "node_map", None)
                    if nm is not None:
                        nm.FindNode("ExposureTime").SetValue(33333.33)
                print(f"[CAM] Default exposure set to 33333.33 µs")
                # Update the exposure text box to show the set value
                if hasattr(self, '_exp_line'):
                    self._exp_line.setText("33333.333")
            except Exception as e:
                print(f"Default exposure set failed: {e}")
            
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

            # Respect current warp mode: H uses homography, LUT uses prewarped content (no H)
            if getattr(self, '_proj_warp_mode', 'H') == 'H':
                self.projection.show_image_fullscreen_on_second_monitor(
                    img,
                    getattr(self._camera, "translation_matrix", None)
                )
            else:
                self.projection.show_image_fullscreen_on_second_monitor(
                    img,
                    None
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
        """Run Structured-Light calibration end-to-end (Gray + Phase subpixel)."""
        try:
            from calibration import (
                generate_gray_code_patterns,
                generate_phase_shift_patterns,
                save_structured_light_patterns,
                decode_gray_code_from_files,
                decode_phase_shift_from_files,
                invert_cam_to_proj_lut,
            )
        except Exception as e:
            print(f"Structured-light not available: {e}")
            return

        if not self._ensure_projection():
            print("Projection window unavailable.")
            return

        # 1) Generate patterns at projector resolution (Gray + Phase)
        try:
            scr = self.projection.windowHandle().screen() if self.projection.windowHandle() else None
            geo = scr.geometry() if scr else None
            proj_w = geo.width() if geo else 1920
            proj_h = geo.height() if geo else 1080
            gray_patterns = generate_gray_code_patterns(proj_w, proj_h)
            use_phase = getattr(self, '_chk_phase_refine', None) is not None and self._chk_phase_refine.isChecked()
            if use_phase:
                # Enable phase-shift patterns for subpixel refinement
                phase_patterns = generate_phase_shift_patterns(
                    proj_w, proj_h, num_phases=3, cycles_x=1, cycles_y=1, gamma=1.0
                )
                patterns = gray_patterns + phase_patterns
            else:
                patterns = gray_patterns
            pattern_paths = save_structured_light_patterns(patterns)
            print(f"Generated {len(pattern_paths)} structured-light patterns (Gray+Phase)")
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
        # If using engine, clear any homography so patterns are unwarped on output
        try:
            use_engine = hasattr(self, '_proc_projector') and (self._proc_projector is not None)
            if use_engine:
                try:
                    import zmq as _zmq
                    _ctx = _zmq.Context.instance(); _s = _ctx.socket(_zmq.REQ)
                    _s.setsockopt(_zmq.LINGER, 0)
                    _s.connect("tcp://127.0.0.1:5560")
                    _s.send(b"IDENTITY")
                    _ = _s.recv()
                    _s.close()
                except Exception:
                    pass
        except Exception:
            pass
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
                        # Force engine overlay OFF for SL, and request immediate scheduling
                        client.send_gray(img, frame_id=idx+1, visible_id=0, immediate=True)
                        matched = client.wait_visible(idx+1, timeout_ms=500)
                        if matched is not None:
                            last_pidx = matched
                        # Allow camera to expose the just-shown pattern before snapshot
                        try:
                            QtCore.QThread.msleep(60)
                        except Exception:
                            pass
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
                    from calibration import (
                        decode_gray_code_from_files as _decode_gray,
                        decode_phase_shift_from_files as _decode_phase,
                        invert_cam_to_proj_lut as _invert,
                    )
                    # Split captures: Gray-code vs Phase (optional)
                    pairs = [(p, m) for p, m in zip(paths, pats)]
                    gray_pairs  = [(p, m) for (p, m) in pairs if isinstance(m, dict) and ('bit' in m)]
                    phase_pairs = [(p, m) for (p, m) in pairs if isinstance(m, dict) and (m.get('type') == 'phase')]
                    paths_gray  = [p for (p, _) in gray_pairs]
                    meta_gray   = [m for (_, m) in gray_pairs]
                    paths_phase = [p for (p, _) in phase_pairs]
                    meta_phase  = [m for (_, m) in phase_pairs]

                    cam_h, cam_w = 1080, 1920
                    for _fp in reversed(paths_gray):  # Only check Gray patterns
                        if not _fp:
                            continue
                        _img = _cv2.imread(_fp, _cv2.IMREAD_GRAYSCALE)
                        if _img is not None:
                            cam_h, cam_w = _img.shape[:2]
                            break
                    print(f"[SL] Decoding Gray-code at {cam_w}x{cam_h} → proj {pw}x{ph}…")
                    proj_x_of_cam, proj_y_of_cam = _decode_gray(paths_gray, meta_gray, cam_h, cam_w, pw, ph)
                    
                    # Optionally apply phase-shift refinement only if present and valid
                    try:
                        if len(paths_phase) > 0 and len(meta_phase) > 0:
                            print(f"[SL] Decoding Phase-shift for subpixel refinement…")
                            px_phase, py_phase, ax, ay = _decode_phase(paths_phase, meta_phase, cam_h, cam_w, pw, ph, num_phases=3, amp_thresh=5.0)
                            # Adaptive amplitude gating: use stricter threshold if coverage is low
                            amp_thr = 5.0
                            # Estimate potential coverage
                            cov_x = float((_np.sum(ax > amp_thr)) / (ax.size if ax.size else 1))
                            cov_y = float((_np.sum(ay > amp_thr)) / (ay.size if ay.size else 1))
                            # If coverage < 20%, try lower threshold 3.0 to rescue weak areas
                            if cov_x < 0.2 or cov_y < 0.2:
                                amp_thr = 3.0
                            use_x = (px_phase >= 0.0) & (ax > amp_thr)
                            use_y = (py_phase >= 0.0) & (ay > amp_thr)
                            applied_x = int(_np.sum(use_x)); applied_y = int(_np.sum(use_y))
                            # Only apply if meaningful coverage (e.g., >10% of pixels)
                            min_cov = 0.10
                            if (applied_x / float(px_phase.size if px_phase.size else 1) > min_cov) or (applied_y / float(py_phase.size if py_phase.size else 1) > min_cov):
                                proj_x_of_cam = proj_x_of_cam.astype(_np.float32, copy=True)
                                proj_y_of_cam = proj_y_of_cam.astype(_np.float32, copy=True)
                                if applied_x > 0:
                                    proj_x_of_cam[use_x] = px_phase[use_x]
                                if applied_y > 0:
                                    proj_y_of_cam[use_y] = py_phase[use_y]
                                print(f"[SL] Phase refinement applied: {applied_x} X px, {applied_y} Y px (thr={amp_thr})")
                            else:
                                print(f"[SL] Phase refinement skipped due to low coverage (X={applied_x}, Y={applied_y})")
                        else:
                            print("[SL] Phase patterns not included; using Gray-code only")
                    except Exception as _pe:
                        print(f"[SL] Phase refinement skipped: {_pe}")
                        print("[SL] Using Gray-code only (phase refinement failed)")
                    _np.save("/".join([asset_dir, "proj_from_cam_x.npy"]), proj_x_of_cam)
                    _np.save("/".join([asset_dir, "proj_from_cam_y.npy"]), proj_y_of_cam)
                    inv_x, inv_y = _invert(proj_x_of_cam, proj_y_of_cam, pw, ph)
                    _np.save("/".join([asset_dir, "cam_from_proj_x.npy"]), inv_x)
                    _np.save("/".join([asset_dir, "cam_from_proj_y.npy"]), inv_y)
                    
                    # Generate diagnostic visualization
                    try:
                        from calibration import visualize_lut_quality
                        diag_path = "/".join([asset_dir, "lut_diagnostic.png"])
                        visualize_lut_quality(inv_x, inv_y, diag_path)
                    except Exception as diag_e:
                        print(f"Could not generate diagnostic: {diag_e}")
                    
                    print("✅ Structured-light LUTs (subpixel) saved (background)")
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
            # Prewarp with error handling
            try:
                warped = prewarp_with_inverse_lut(img, inv_x, inv_y, proj_w, proj_h)
            except Exception as warp_e:
                print(f"Warping failed: {warp_e}")
                # Try simple resize as fallback
                warped = cv2.resize(img, (proj_w, proj_h), interpolation=cv2.INTER_LINEAR)
                print("Using simple resize as fallback")
            
            # Prefer projection engine via ZMQ if running; ensures sync with triggers
            use_engine = hasattr(self, '_proc_projector') and (self._proc_projector is not None)
            if use_engine:
                try:
                    from projector_client import ProjectorClient
                    # Engine expects 1920x1080; client will resize
                    client = ProjectorClient()
                    # Clear engine homography so the prewarped image is not warped again
                    try:
                        import zmq as _zmq
                        _ctx = _zmq.Context.instance(); _s = _ctx.socket(_zmq.REQ)
                        _s.setsockopt(_zmq.LINGER, 0)
                        _s.connect("tcp://127.0.0.1:5560"); _s.send(b"IDENTITY"); _ = _s.recv(); _s.close()
                    except Exception:
                        pass
                    if getattr(self, '_button_hw_trig', None) and self._button_hw_trig.isChecked():
                        client.enable_gpio_trigger(22)
                    client.send_gray(
                        warped,
                        frame_id=9999,
                        visible_id=int(bool(self._button_toggle_overlay.isChecked()))
                    )
                    # Optionally wait for visibility, but pulsing is now handled by background subscriber when enabled
                    _ = client.wait_visible(9999, timeout_ms=250)
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

    def _apply_exposure_from_text(self):
        try:
            txt = self._exp_line.text().strip()
            if not txt:
                return
            exp_us = float(txt)
            if not (exp_us > 0):
                return
            # Apply to camera
            if hasattr(self._camera, "set_exposure_us"):
                self._camera.set_exposure_us(exp_us)
            else:
                # Fallback: try node map if available
                try:
                    nm = getattr(self._camera, "node_map", None)
                    if nm is not None:
                        nm.FindNode("ExposureTime").SetValue(exp_us)
                except Exception:
                    pass
            print(f"[CAM] Exposure set to {exp_us:.3f} µs")
        except Exception as e:
            print(f"Exposure apply failed: {e}")

    def _open_sensor_settings(self):
        try:
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QGridLayout, QLabel, QPushButton
            dlg = QDialog(self)
            dlg.setWindowTitle("Sensor Settings")
            # Make it a movable, modeless top-level window
            try:
                dlg.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.WindowTitleHint | QtCore.Qt.WindowCloseButtonHint)
                dlg.setModal(False)
                dlg.setWindowModality(QtCore.Qt.NonModal)
                dlg.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
            except Exception:
                pass
            lay = QVBoxLayout(dlg)
            grid = QGridLayout()

            # Reuse existing widgets by creating new controls bound to same slots
            # AG slider
            ag_label = QtWidgets.QLabel("Analog Gain")
            ag_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            ag_slider.setRange(self._gain_slider.minimum(), self._gain_slider.maximum())
            ag_slider.setValue(self._gain_slider.value())
            ag_slider.valueChanged.connect(self._update_gain)
            ag_val = QtWidgets.QLabel(self._gain_value_label.text())
            ag_slider.valueChanged.connect(lambda v: ag_val.setText(f"{v/100:.2f}"))

            grid.addWidget(ag_label, 0, 0)
            grid.addWidget(ag_slider, 0, 1)
            grid.addWidget(ag_val, 0, 2)

            # DG slider
            dg_label = QtWidgets.QLabel("Digital Gain")
            dg_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            dg_slider.setRange(self._dgain_slider.minimum(), self._dgain_slider.maximum())
            dg_slider.setValue(self._dgain_slider.value())
            dg_slider.valueChanged.connect(self._update_dgain)
            dg_val = QtWidgets.QLabel(self._dgain_value_label.text())
            dg_slider.valueChanged.connect(lambda v: dg_val.setText(f"{v/100:.2f}"))

            grid.addWidget(dg_label, 1, 0)
            grid.addWidget(dg_slider, 1, 1)
            grid.addWidget(dg_val, 1, 2)

            # Exposure textbox
            exp_label = QtWidgets.QLabel("Exposure (µs)")
            exp_line = QtWidgets.QLineEdit(self._exp_line.text())
            exp_line.setValidator(QtGui.QDoubleValidator(1.0, 1e9, 3))

            def _apply_local_exp():
                try:
                    self._exp_line.setText(exp_line.text())
                    self._apply_exposure_from_text()
                except Exception:
                    pass

            grid.addWidget(exp_label, 2, 0)
            grid.addWidget(exp_line, 2, 1, 1, 2)
            try:
                set_btn = QPushButton("Set")
                set_btn.clicked.connect(_apply_local_exp)
                grid.addWidget(set_btn, 2, 3)
            except Exception:
                pass

            lay.addLayout(grid)
            btns = QtWidgets.QHBoxLayout()
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dlg.accept)
            btns.addStretch(1)
            btns.addWidget(close_btn)
            lay.addLayout(btns)
            # Keep a reference so it stays alive when shown modelessly
            self._sensor_settings_dlg = dlg
            try:
                dlg.show()
                dlg.raise_()
                dlg.activateWindow()
            except Exception:
                dlg.show()
        except Exception as e:
            print(f"Sensor Settings UI error: {e}")


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

    def _asift_calibrate(self):
        """Compute 3x3 H via ASIFT (fallback SIFT), update camera H and projector.

        - Loads reference/capture paths from Assets/Generated
        - Uses ZMQ_sender_mask.asift_calibration backend
        - Writes homography_cam2proj.txt next to existing files
        """
        try:
            from pathlib import Path
            import os
            import numpy as np
            import cv2
            # Import backend (ensure repository path is on sys.path or installed)
            try:
                from ZMQ_sender_mask.asift_calibration import run_asift_calibration_and_send
            except Exception as e:
                # Attempt to add local MyUART workspace to sys.path
                try:
                    import sys as _sys
                    _sys.path.insert(0, "/home/aharonilabjetson2/Desktop/MyScripts/MyUART")
                    from ZMQ_sender_mask.asift_calibration import run_asift_calibration_and_send
                except Exception as e2:
                    print(f"ASIFT backend import failed: {e2}")
                    return

            assets = Path(__file__).resolve().parent / "Assets" / "Generated"
            ref_path = (assets / "custom_registration_image.png").as_posix()
            cam_path = (assets / "calibration_capture_image.png").as_posix()
            save_txt = (assets / "homography_cam2proj.txt").as_posix()

            ok, H = run_asift_calibration_and_send(ref_path, cam_path, endpoint="tcp://127.0.0.1:5560", save_txt=save_txt)
            if not ok or H is None:
                print("ASIFT calibration failed: no H")
                return

            # Update in-memory camera H so the rest of UI uses the new matrix
            try:
                if hasattr(self, "_camera") and (self._camera is not None):
                    self._camera.translation_matrix = H
            except Exception:
                pass

            # Send to projector immediately
            try:
                self._camera._send_h_to_projector(H)
            except Exception as esend:
                print(f"Could not send ASIFT H to projector: {esend}")
            print(f"ASIFT Calibration OK. Wrote: {save_txt}")

            # Immediately apply H to the registration image and project it for confirmation
            try:
                if not self._ensure_projection():
                    print("ASIFT confirm: projection window unavailable.")
                    return
                img_path = (Path(__file__).resolve().parent / "Assets" / "Generated" / "custom_registration_image.png").as_posix()
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"ASIFT confirm: cannot read {img_path}")
                    return
                # Use current warp mode H; show image with H
                try:
                    Hn = H / H[2, 2] if abs(float(H[2, 2])) > 1e-12 else H
                except Exception:
                    Hn = H
                try:
                    self.projection.show_image_fullscreen_on_second_monitor(img, Hn)
                except Exception:
                    # Fallback to interface method
                    self.on_projection_received(img, Hn)
                print("ASIFT confirm: projected registration with new H")
            except Exception as econf:
                print(f"ASIFT confirm failed: {econf}")
        except Exception as e:
            print(f"ASIFT Calibration error: {e}")

    def _select_warp_h(self):
        # Toggle behavior: if already active, turn off; else activate H and deactivate LUT
        try:
            if getattr(self, '_proj_warp_mode', 'H') == 'H' and self._button_req_hmatrix.isChecked():
                # Deactivate
                self._proj_warp_mode = "NONE"
                self._button_req_hmatrix.setChecked(False)
                print("[PROJ] Warp mode: None (no H applied)")
            else:
                self._proj_warp_mode = "H"
                if hasattr(self, '_button_req_hmatrix'):
                    self._button_req_hmatrix.setChecked(True)
                if hasattr(self, '_button_use_lut'):
                    self._button_use_lut.setChecked(False)
                # Send H to projector immediately
                self._send_hmatrix_to_projector()
                print("[PROJ] Warp mode: Homography (H)")
        except Exception as e:
            print(f"Warp H select failed: {e}")

    def _select_warp_lut(self):
        # Toggle behavior: if already active, turn off; else activate LUT and deactivate H
        try:
            if getattr(self, '_proj_warp_mode', 'H') == 'LUT' and self._button_use_lut.isChecked():
                self._proj_warp_mode = "NONE"
                self._button_use_lut.setChecked(False)
                print("[PROJ] Warp mode: None (no H; content not assumed prewarped)")
            else:
                self._proj_warp_mode = "LUT"
                if hasattr(self, '_button_req_hmatrix'):
                    self._button_req_hmatrix.setChecked(False)
                if hasattr(self, '_button_use_lut'):
                    self._button_use_lut.setChecked(True)
                print("[PROJ] Warp mode: LUT (engine will display prewarped content)")
        except Exception as e:
            print(f"Warp LUT select failed: {e}")

    def _on_warp_h_toggled(self, checked: bool):
        if checked:
            # activate H
            self._proj_warp_mode = "H"
            try:
                if hasattr(self, '_button_use_lut'):
                    self._button_use_lut.setChecked(False)
            except Exception:
                pass
            self._send_hmatrix_to_projector()
            print("[PROJ] Warp mode: Homography (H)")
        else:
            # if H turned off and LUT not active → NONE
            if (getattr(self, '_button_use_lut', None) is None) or (not self._button_use_lut.isChecked()):
                self._proj_warp_mode = "NONE"
                print("[PROJ] Warp mode: None")

    def _on_warp_lut_toggled(self, checked: bool):
        if checked:
            self._proj_warp_mode = "LUT"
            try:
                if hasattr(self, '_button_req_hmatrix'):
                    self._button_req_hmatrix.setChecked(False)
            except Exception:
                pass
            print("[PROJ] Warp mode: LUT (engine will display prewarped content)")
        else:
            if (getattr(self, '_button_req_hmatrix', None) is None) or (not self._button_req_hmatrix.isChecked()):
                self._proj_warp_mode = "NONE"
                print("[PROJ] Warp mode: None")

    def _toggle_overlay(self, checked: bool):
        try:
            if not hasattr(self, '_button_toggle_overlay') or self._button_toggle_overlay is None:
                return
            self._button_toggle_overlay.setText("Overlay: On" if checked else "Overlay: Off")
            if hasattr(self, '_proc_projector') and self._proc_projector is not None:
                print("[PROJ] Overlay toggle changed; restart Projection Engine to apply")
        except Exception as e:
            print(f"_toggle_overlay error: {e}")
