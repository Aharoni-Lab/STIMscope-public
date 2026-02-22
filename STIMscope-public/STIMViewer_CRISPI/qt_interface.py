
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

        dlg = QDialog()
        dlg.setWindowTitle("STIMViewer")
        layout = QVBoxLayout(dlg)

        logo = QLabel()  
        logo.setAlignment(Qt.AlignCenter)
        logo_path = self._findprinto()
        if logo_path:
            logo.setPixmap(QPixmap(str(logo_path)))
        else:
            print(f"Logo not found in {ASSETS}")
        layout.addWidget(logo)



        hbox = QtWidgets.QHBoxLayout()


        cam_label = QLabel("Camera Type:")
        self.camera_type_dropdown = QtWidgets.QComboBox()
        self.camera_type_dropdown.addItems(["IDS_Peak", "MIPI", "Generic Camera"])

        cam_layout = QtWidgets.QVBoxLayout()
        cam_layout.addWidget(cam_label)
        cam_layout.addWidget(self.camera_type_dropdown)
        hbox.addLayout(cam_layout)
        

        screens = QGuiApplication.screens()
        projector_status = QLabel()
        if len(screens) > 1:
            projector_status.setText("✅ Projector Connected")
            projector_status.setStyleSheet("color: green; font-weight: bold;")
        else:
            projector_status.setText("❌ No Projector Found")
            projector_status.setStyleSheet("color: red; font-weight: bold;")
        projector_status.setAlignment(Qt.AlignCenter)
        hbox.addWidget(projector_status)


        btn = QPushButton('Start STIMViewer')
        btn.clicked.connect(dlg.accept)
        hbox.addWidget(btn)


        layout.addLayout(hbox)


        if dlg.exec() != QDialog.Accepted:
            raise RuntimeError("User cancelled startup")

        self.selected_camera_type = self.camera_type_dropdown.currentText()

        self.last_frame_time = time.time()
        self.gpu_ui = None
        
        self.gui_init()
       
        
        self._qt_instance.aboutToQuit.connect(self._close)

        self.setMinimumSize(700, 650)
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
        self._layout.addWidget(self.display)
        self.projection = None
        self.acquisition_thread = None


        self._button_software_trigger = None
        self._button_start_hardware_acquisition = None
        self._hardware_status = False #False = Display Start, False = End
        self._recording_status = False #False = Display Start, False = End



        self._dropdown_pixel_format = None
        self._dropdown_trigger_line = None # Dropdown for hardware trigger line





        
        self._button_show_gpu_ui = None

        self.messagebox_pyqtSignal.connect(self.message)
        for sig, slot in (("recordingStarted", self._on_recording_started),
                          ("recordingStopped", self._on_recording_stopped)):
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
    

    def _create_button_bar(self):
       

        button_bar = QtWidgets.QWidget(self.centralWidget())
        button_bar_layout = QtWidgets.QGridLayout()


        self._button_start_hardware_acquisition = QtWidgets.QPushButton("Start Hardware Acquisition")
        self._button_start_hardware_acquisition.clicked.connect(self._start_hardware_acquisition)


        self._button_start_recording = QtWidgets.QPushButton("Start Recording")
        self._button_start_recording.clicked.connect(self._start_recording)




        
        self._button_show_gpu_ui = QtWidgets.QPushButton("Show CRISPI")
        self._button_show_gpu_ui.clicked.connect(self.show_gpu_ui)
        self._button_show_gpu_ui.setEnabled(_GPU_AVAILABLE)



        self._dropdown_trigger_line = QtWidgets.QComboBox()
        self._label_trigger_line = QtWidgets.QLabel("Change Hardware Trigger Line:")



        self._dropdown_trigger_line.addItem("Line0")
        self._dropdown_trigger_line.addItem("Line1")   
        self._dropdown_trigger_line.addItem("Line2")
        self._dropdown_trigger_line.addItem("Line3")


        self._dropdown_trigger_line.currentIndexChanged.connect(self.change_hardware_trigger_line)


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


        self._dropdown_pixel_format.setEnabled(True)
        self._dropdown_trigger_line.setEnabled(True)



        

        self._button_software_trigger = QtWidgets.QPushButton("Snapshot")
        self._button_software_trigger.clicked.connect(self._trigger_sw_trigger)
        
        

        self._button_calibrate = QtWidgets.QPushButton("Calibrate")
        self._button_calibrate.clicked.connect(self._calibrate)

        self._button_project_white = QtWidgets.QPushButton("Project White")
        self._button_project_white.clicked.connect(self._project_white)


        self._gain_label = QtWidgets.QLabel("<b>Gain:</b>")
        self._gain_label.setMaximumWidth(70)

        self._gain_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical)
        self._gain_slider.setRange(100, 1000)
        self._gain_slider.setSingleStep(1)
        self._gain_slider.valueChanged.connect(self._update_gain)

        

        self._dgain_label = QtWidgets.QLabel("<b>DGain:</b>")
        self._dgain_label.setMaximumWidth(70)

        self._dgain_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical)
        self._dgain_slider.setRange(100, 1000)
        self._dgain_slider.setSingleStep(1)
        self._dgain_slider.valueChanged.connect(self._update_dgain)


        self._zoom_label = QtWidgets.QLabel("<b>Zoom:</b>")
        self._zoom_label.setMaximumWidth(70)

        self._zoom_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical)
        self._zoom_slider.setRange(100, 1000)
        self._zoom_slider.setSingleStep(1)
        self._zoom_slider.valueChanged.connect(self._update_zoom)



        config_group = QtWidgets.QGroupBox("Config")
        config_layout = QtWidgets.QGridLayout()
        config_group.setLayout(config_layout)
        config_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid gray;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
                font-size: 11px;
            }
            QLabel {
                font-size: 11px;
            }
        """)


        config_layout.addWidget(self._button_start_hardware_acquisition, 0, 0)

        config_layout.addWidget(self._button_calibrate,                  1, 0)
        config_layout.addWidget(self._button_project_white,              1, 1)
        config_layout.addWidget(self._label_trigger_line,                2, 0)
        config_layout.addWidget(self._dropdown_trigger_line,             2, 1)    


        capture_group = QtWidgets.QGroupBox("Capture")
        capture_layout = QtWidgets.QGridLayout()
        capture_group.setLayout(capture_layout)
        capture_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid gray;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
                font-size: 11px;
            }
            QLabel, QPushButton {
                font-size: 11px;
            }
        """)


        capture_layout.addWidget(self._button_start_recording, 0, 0)
        capture_layout.addWidget(self._button_software_trigger, 0, 1)
        capture_layout.addWidget(self._dropdown_pixel_format, 1, 0)


        control_group = QtWidgets.QGroupBox("Adjustments")
        control_group_layout = QtWidgets.QGridLayout()
        control_group.setLayout(control_group_layout)
        control_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid gray;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
                font-size: 11px;
            }
            QLabel {
                font-size: 11px;
            }
        """)


        self._gain_label.setAlignment(Qt.AlignCenter)
        self._gain_slider.setFixedWidth(25)
        control_group_layout.addWidget(self._gain_label, 0, 0)
        control_group_layout.addWidget(self._gain_slider, 1, 0)
        self._gain_value_label = QtWidgets.QLabel("1.00")
        self._gain_value_label.setAlignment(Qt.AlignCenter)
        self._gain_value_label.setStyleSheet("font-size: 10px;")
        control_group_layout.addWidget(self._gain_value_label, 2, 0)


        self._dgain_label.setAlignment(Qt.AlignCenter)
        self._dgain_slider.setFixedWidth(25)
        control_group_layout.addWidget(self._dgain_label, 0, 1)
        control_group_layout.addWidget(self._dgain_slider, 1, 1)
        self._dgain_value_label = QtWidgets.QLabel("1.00")
        self._dgain_value_label.setAlignment(Qt.AlignCenter)
        self._dgain_value_label.setStyleSheet("font-size: 10px;")
        control_group_layout.addWidget(self._dgain_value_label, 2, 1)


        self._zoom_label.setAlignment(Qt.AlignCenter)
        self._zoom_slider.setFixedWidth(25)
        control_group_layout.addWidget(self._zoom_label, 0, 2)
        control_group_layout.addWidget(self._zoom_slider, 1, 2)
        self._zoom_value_label = QtWidgets.QLabel("1.00")
        self._zoom_value_label.setAlignment(Qt.AlignCenter)
        self._zoom_value_label.setStyleSheet("font-size: 10px;")
        control_group_layout.addWidget(self._zoom_value_label, 2, 2)


        control_group.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed,
            QtWidgets.QSizePolicy.Preferred
        )
        for grp in (config_group, capture_group):
            grp.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Preferred
            )


        button_bar_layout.setColumnStretch(4, 1)
        button_bar_layout.setColumnStretch(5, 1)
        button_bar_layout.setColumnStretch(7, 0)


        button_bar_layout.addWidget(control_group, 0, 7, 7, 1)
        button_bar_layout.addWidget(config_group, 0, 4, 4, 2)
        button_bar_layout.addWidget(capture_group, 4, 4, 1, 2)
        button_bar_layout.addWidget(self._button_show_gpu_ui, 5, 4, 1, 2)



        self._button_start_hardware_acquisition.setToolTip("Start/Stop acquiring images using hardware triggering rather than real time(RT) acquisition. Hardware Trigger FPS must stay <45 hz")
        self._button_start_recording.setToolTip("Start/Stop recording video of the live feed.")
        self._button_software_trigger.setToolTip("Save the next processed frame.")


        self._gain_label.setToolTip("Adjust the analog gain level (brightness).")
        self._dgain_label.setToolTip("Adjust the digital gain level.")
        self._zoom_label.setToolTip("Zoom in and out of the displayed image.")


        button_bar.setLayout(button_bar_layout)
        self._layout.addWidget(button_bar)

    def _create_statusbar(self):
       
        status_bar = QtWidgets.QWidget(self.centralWidget())
        status_bar_layout = QtWidgets.QHBoxLayout()
        status_bar_layout.setContentsMargins(0, 0, 0, 0)


        separator = QFrame(self)
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._layout.addWidget(separator)


        self.acq_label = QLabel("Acquisition Mode: RealTime", self)
        self.acq_label.setStyleSheet("font-size: 14px; color: green;")
        self.acq_label.setAlignment(Qt.AlignLeft)
        self.acq_label.setToolTip("Current Acquisition Mode")


        spacer = QtWidgets.QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)


        self.GUIfps_label = QLabel("GUI FPS: 0.00", self)
        self.GUIfps_label.setStyleSheet("font-size: 14px; color: green;")
        self.GUIfps_label.setAlignment(Qt.AlignRight)
        self.GUIfps_label.setToolTip("Calculated FPS over a rolling average of 2 seconds. If set to hardware trigger mode, camera only supports <45 fps.")


        status_bar_layout.addWidget(self.acq_label)
        status_bar_layout.addItem(spacer)  
        status_bar_layout.addWidget(self.GUIfps_label)

        status_bar.setLayout(status_bar_layout)
        self._layout.addWidget(status_bar)

    def _close(self):
        try:
            self._camera.shutdown()
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
            self.image_update_signal.connect(self.display.on_image_received)
            print("Bound image_update_signal → Display.on_image_received")
        except Exception as e1:
            print(f"Primary connect failed ({e1}); falling back to setImage alias")
            try:
                self.image_update_signal.connect(self.display.setImage)
                print("Bound image_update_signal → Display.setImage")
            except Exception as e2:
                print(f"Display signal hookup failed: {e2}")
        screens = QGuiApplication.screens()
        screen = screens[1] if len(screens) > 1 else screens[0]
        from projection import ProjectDisplay

        try:
            self.projection = ProjectDisplay(screen, parent=self)
        except TypeError:
            self.projection = ProjectDisplay(screen)
            self.projection.setParent(self)
        self.projection.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)


    @QtCore.pyqtSlot()
    def _on_recording_started(self):
        self._recording_status = True
        self._button_start_recording.setText("Stop Recording")
        self._button_start_hardware_acquisition.setEnabled(False)
        self._dropdown_trigger_line.setEnabled(False)

    @QtCore.pyqtSlot()
    def _on_recording_stopped(self):
        self._recording_status = False
        self._button_start_recording.setText("Start Recording")
        self._button_start_hardware_acquisition.setEnabled(True)
        if not self._hardware_status:
            self._dropdown_trigger_line.setEnabled(True)
    
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
        else:
            self._camera.stop_hardware_acquisition()
            self._camera.start_realtime_acquisition()
            self.acq_label.setText("Acquisition Mode: RealTime")
            self._button_start_hardware_acquisition.setText("Start Hardware Acquisition")
            if not self._recording_status:
                self._dropdown_trigger_line.setEnabled(True)

        self._hardware_status = not self._hardware_status


    def _start_recording(self):
        try:
            if getattr(self._camera, "is_recording", False):
                self._camera.stop_recording()
            else:
                self._camera.start_recording()
        except Exception as e:
            print(f"Recording toggle failed: {e}")


    def _calibrate(self):
       
        if self.projection is None:
            print("Calibration aborted: projection window unavailable.")
            return
        try:
            img_path = ASSETS / "Generated" / "custom_registration_image.png"
            if not img_path.exists():

                try:
                    from calibration import create_custom_registration_image
                    scr = self.projection.windowHandle().screen() if self.projection.windowHandle() else None
                    geo = scr.geometry() if scr else None
                    w = geo.width() if geo else 1920
                    h = geo.height() if geo else 1080
                    create_custom_registration_image(w, h, (255, 255, 255), (255, 255, 255))
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


            QtCore.QTimer.singleShot(150, lambda: getattr(self._camera, "start_calibration", lambda: None)())
        except Exception as e:
            print(f"Calibration start failed: {e}")

    
    def _project_white(self):
        
        try:

            if self.projection is None:
                print("Projection window unavailable.")
                return



            self.projection.show_solid_fullscreen((255, 255, 255))


            """
            from pathlib import Path
            import cv2
            img_path = (ASSETS / "Generated" / "solid_white_image.png").resolve()
            if not img_path.exists():
                print(f"Solid white asset missing, regenerating via makeWhite(1920,1080)")
                try:
                    from WhiteBackgroundGen import makeWhite
                    makeWhite(1920, 1080)
                except Exception as e:
                    print(f"Failed to regenerate white asset: {e}")
            if img_path.exists():
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)  # BGR
                if img is not None:
                    H = getattr(self._camera, "homography_matrix", None)
                    if not (isinstance(H, np.ndarray) and H.shape == (3, 3)):
                        H = None
                    self.projection.show_image_fullscreen_on_second_monitor(img, H)
                else:
                    print(f"White image unreadable: {img_path}")
            """
        except Exception as e:
            print(f"_project_white failed: {e}")



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
                GUIfps = self._camera.get_actual_fps()
                self.GUIfps_label.setText(f"GUI FPS: {GUIfps:.2f}")
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
        from gpu_ui import GPU

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


    @Slot(float)
    def change_slider_zoom(self, val):
        self._zoom_slider.setValue(int(val * 100))

    @Slot(int)
    def _update_zoom(self, val):
        value = val / 100
        self._zoom_value_label.setText(f"{value:.2f}")
        self.display.set_zoom(value)
