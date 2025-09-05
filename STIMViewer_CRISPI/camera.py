
import os
import sys
import time
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from typing import Optional

import numpy as np
import cv2

from ids_peak import ids_peak
from ids_peak_ipl import ids_peak_ipl
from ids_peak import ids_peak_ipl_extension
from PyQt5 import QtCore
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer





def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default

def _get_env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v else default

TARGET_PIXEL_FORMAT = {
    "BGRA8": ids_peak_ipl.PixelFormatName_BGRa8,
    "BGR8":  ids_peak_ipl.PixelFormatName_BGR8,
    "RGBA8": ids_peak_ipl.PixelFormatName_RGBa8,
    "RGB8":  ids_peak_ipl.PixelFormatName_RGB8,
}.get(_get_env_str("STIM_PIXEL_FORMAT", "BGRA8").upper(), ids_peak_ipl.PixelFormatName_BGRa8)

DEFAULT_FPS       = _get_env_int("STIM_CAMERA_FPS", 60)
DEFAULT_BUFFERS   = max(4, _get_env_int("STIM_PEAK_BUFFERS", 16))
DEFAULT_TRIG_LINE = _get_env_str("STIM_TRIGGER_LINE", "Line0")
DEFAULT_RT_START  = _get_env_int("STIM_RT_DEFAULT", 1) == 1 

ASSETS_DIR  = _get_env_str("STIM_ASSETS_DIR", None)
CRISPI_ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS_FALLBACK = os.path.join(CRISPI_ROOT, "Assets")

def _assets_path(*parts) -> str:
    base = ASSETS_DIR if ASSETS_DIR else ASSETS_FALLBACK
    return os.path.join(base, *parts)




class OptimizedCamera(QObject):
 
    frame_ready = pyqtSignal(object)    
    recordingStarted = pyqtSignal()
    recordingStopped = pyqtSignal()
    performance_metrics = pyqtSignal(dict)
    

    def __init__(self, device_manager, interface):
        super().__init__()
        if interface is None:
            raise ValueError("Interface is None")


        self._interface = interface
        try:
            self.frame_ready.connect(self._interface.on_image_received)
        except Exception:
            pass


        self.device_manager = device_manager
        self._device = None
        self._datastream = None
        self.node_map = None

        self._last_acq_err_ts = 0.0
        self._acq_err_interval = 1.0

        self._snapshot_path: Optional[str] = None



        self.acquisition_mode = 0  # 0: RT, 1: HW
        self.acquisition_running = False
        self._acq_thread: Optional[threading.Thread] = None
        self.acquisition_thread = None   # legacy alias
        self._acq_stop = threading.Event()


        self._buffer_list = []
        self._image_converter = ids_peak_ipl.ImageConverter()


        self.killed = False
        self.is_recording = False
        self.save_image = False
        self.hardware_trigger_line = DEFAULT_TRIG_LINE


        self.target_gain = 1.0
        self.max_gain = 1.0
        self.target_dgain = 1.0


        self.frame_times = deque(maxlen=120)
        self.GUIfps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.performance_stats = {
            "fps": 0.0,
            "frame_processing_time": 0.0,
            "memory_usage": 0.0,
            "thread_pool_usage": 0.0,
        }


        self.translation_matrix = np.eye(3, dtype=np.float64)
        self.calibration_running = False
        self.calibration_lock = threading.Lock()

        self._dest_pf = None


        self.asset_dir = _assets_path("Generated")
        self.save_dir = _get_env_str("STIM_SAVE_DIR",
                                     os.path.join(CRISPI_ROOT, "Saved_Media"))
        os.makedirs(self.asset_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)


        self.thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="CameraWorker")
        self.recording_queue: queue.Queue = queue.Queue(maxsize=30)
        self.save_queue: queue.Queue = queue.Queue(maxsize=10)
        self.recording_worker_running = False
        self.save_worker_running = False


        self._open_device()
        self._apply_defaults()
        self._init_data_stream()
        self._interface.set_camera(self)


        from video_recorder import VideoRecorder

        self.video_recorder = VideoRecorder(interface)


        self._start_background_workers()



    def start(self, start_rt: bool = DEFAULT_RT_START):
       
        if start_rt:
            self.start_realtime_acquisition()
        self._start_acquisition_thread()

    def _pick_dest_pf(self, ipl_src):
        try:
            outs = self._image_converter.SupportedOutputPixelFormatNames(ipl_src.PixelFormat())

            pref = [
                ids_peak_ipl.PixelFormatName_BGRa8,
                ids_peak_ipl.PixelFormatName_BGR8,
                ids_peak_ipl.PixelFormatName_RGBa8,
                ids_peak_ipl.PixelFormatName_RGB8,
            ]
            for p in pref:
                if p in outs:
                    return p
            return outs[0] if outs else TARGET_PIXEL_FORMAT
        except Exception:
            return TARGET_PIXEL_FORMAT


    def _pause_stream_for_change(self):
       
        was_running   = bool(self.acquisition_running)
        was_recording = bool(self.is_recording)
        prev_mode     = self.acquisition_mode  # 0: RT, 1: HW


        critical_change = False
        

        r = getattr(self, "video_recorder", None)
        if was_recording and r is not None and critical_change:
            try: 
                r.stop_recording()
                print("⏸️ Recording paused for critical parameter change")
            except Exception: 
                pass


        if was_running and critical_change:
            try:
                if prev_mode == 0:
                    self.stop_realtime_acquisition()
                else:
                    self.stop_hardware_acquisition()
                print("⏸️ Acquisition paused for critical parameter change")
            except Exception:
                pass
        elif was_running:

            try:

                if self._datastream:
                    self._datastream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
                time.sleep(0.01) 
            except Exception:
                pass

        return was_running, was_recording, prev_mode

    def _resume_stream_after_change(self, was_running, was_recording, prev_mode):
        try:
            if was_running:
                if prev_mode == 0:
                    self.start_realtime_acquisition()
                else:
                    self.start_hardware_acquisition()
                    
            if was_recording and getattr(self, "video_recorder", None):
                self.start_recording() 
        except Exception:
            pass


    def _rebuild_converter_and_buffers(self):
        try:

            try:
                self._payload_size = int(self.node_map.FindNode("PayloadSize").Value())
            except Exception:
                self._payload_size = None


            self.revoke_and_allocate_buffer()


            self.frame_times.clear()
            self.frame_count = 0
            self.start_time = time.time()
            self._dest_pf = None
        except Exception as e:
            print(f"Failed to rebuild buffers after setting: {e}")



    def change_pixel_format(self, symbolic: str) -> bool:
        was_running, was_recording, prev_mode = self._pause_stream_for_change()
        ok = False
        try:
            node = self.node_map.FindNode("PixelFormat")

            setter = getattr(node, "FromString", None)
            if callable(setter):
                setter(symbolic)
            else:

                entries = node.Entries()
                chosen = None
                for e in entries:
                    if e.AccessStatus() in (
                        ids_peak.NodeAccessStatus_NotAvailable,
                        ids_peak.NodeAccessStatus_NotImplemented
                    ):
                        continue
                    if e.SymbolicValue() == symbolic:
                        chosen = e
                        break
                if not chosen:
                    raise RuntimeError(f"PixelFormat '{symbolic}' not available")
                node.SetCurrentEntry(chosen)
            ok = True
        except Exception as e:
            ok = False
        finally:
            self._rebuild_converter_and_buffers()
            self._resume_stream_after_change(was_running, was_recording, prev_mode)
        return ok


    def set_fps(self, fps: int) -> bool:
       
        try:
            was_running, was_recording, prev_mode = self._pause_stream_for_change()
            
            node = self.node_map.FindNode("AcquisitionFrameRate")
            if node is None:
                print("❌ AcquisitionFrameRate node not found")
                return False
            

            try:
                mn, mx = node.Minimum(), node.Maximum()
                fps = max(mn, min(mx, fps))
            except Exception:
                pass
            
            node.SetValue(float(fps))
            print(f"✅ Camera frame rate set to {fps} FPS")
            
            self._resume_stream_after_change(was_running, was_recording, prev_mode)
            return True
            
        except Exception as e:
            print(f"❌ FPS setting error: {e}")
            return False

    def set_gain(self, value: float) -> bool:
        """
        Optimized gain setter that minimizes FPS impact.
        Gain changes usually don't require stopping acquisition.
        """
        try:
            node = self.node_map.FindNode("Gain")
            if node is None:
                print("❌ Gain node not found")
                return False
            

            try:
                access_status = node.AccessStatus()
                if access_status not in (ids_peak.NodeAccessStatus_ReadWrite,):
                    print("⚠️ Gain node not writable during acquisition")

                    return self._set_gain_with_pause(value)
            except Exception:
                pass
            

            try:
                mn, mx = node.Minimum(), node.Maximum()
                value = max(mn, min(mx, value))
            except Exception:
                pass
            

            self.target_gain = value
            

            try:
                node.SetValue(float(value))
                print(f"✅ Gain set to {value:.2f} (live change)")
                return True
            except Exception as e:
                print(f"⚠️ Live gain change failed: {e}, using safe method")
                return self._set_gain_with_pause(value)
                
        except Exception as e:
            print(f"❌ Gain setting error: {e}")
            return False

    def _set_gain_with_pause(self, value: float) -> bool:
       
        was_running, was_recording, prev_mode = self._pause_stream_for_change()
        ok = False
        try:
            node = self.node_map.FindNode("Gain")
            node.SetValue(float(value))
            self.target_gain = value
            ok = True
            print(f"✅ Gain set to {value:.2f} (with pause)")
        except Exception as e:
            print(f"❌ Cannot set gain: {e}")
            ok = False
        finally:

            if not ok:
                self._rebuild_converter_and_buffers()
            self._resume_stream_after_change(was_running, was_recording, prev_mode)
        return ok

    def set_dgain(self, value: float) -> bool:
        """
        Set digital gain with FPS preservation.
        
        Args:
            value: Digital gain value
            
        Returns:
            True if successful, False otherwise
        """
        try:

            node = self.node_map.FindNode("DigitalGain")
            if node is None:
                print("❌ DigitalGain node not found")
                return False
            

            try:
                access_status = node.AccessStatus()
                if access_status in (ids_peak.NodeAccessStatus_ReadWrite,):

                    try:
                        mn, mx = node.Minimum(), node.Maximum()
                        value = max(mn, min(mx, value))
                    except Exception:
                        pass
                    
                    node.SetValue(float(value))
                    self.target_dgain = value
                    print(f"✅ Digital gain set to {value:.2f} (live change)")
                    return True
            except Exception:
                pass
            

            return self._set_dgain_with_pause(value)
            
        except Exception as e:
            print(f"❌ Digital gain setting error: {e}")
            return False

    def _set_dgain_with_pause(self, value: float) -> bool:
       
        was_running, was_recording, prev_mode = self._pause_stream_for_change()
        ok = False
        try:
            node = self.node_map.FindNode("DigitalGain")
            node.SetValue(float(value))
            self.target_dgain = value
            ok = True
            print(f"✅ Digital gain set to {value:.2f} (with pause)")
        except Exception as e:
            print(f"❌ Cannot set digital gain: {e}")
            ok = False
        finally:
            self._resume_stream_after_change(was_running, was_recording, prev_mode)
        return ok


    def snapshot(self, path: str) -> bool:
    
        try:

            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            
            if self.acquisition_running:

                img = self._get_latest_frame_for_snapshot()
                if img is not None:
                    try:
                        ids_peak_ipl.ImageWriter.WriteAsPNG(path, img)
                        print(f"✅ Snapshot saved: {path}")
                        return True
                    except Exception as e:
                        print(f"❌ Snapshot save failed: {e}")
                        return False
                else:
                    print("❌ No frame available for snapshot")
                    return False


            print("Starting temporary acquisition for snapshot...")
            started = self.start_realtime_acquisition()
            if not started:
                print("❌ Snapshot failed: could not start acquisition")
                return False
            
            try:

                time.sleep(0.1)
                

                t0 = time.time()
                while time.time() - t0 < 2.0:
                    img = self.get_data_stream_image()
                    if img is not None:
                        try:
                            ids_peak_ipl.ImageWriter.WriteAsPNG(path, img)
                            print(f"✅ Snapshot saved: {path}")
                            return True
                        except Exception as e:
                            print(f"❌ Snapshot save failed: {e}")
                            return False
                    time.sleep(0.01)  
                
                print("❌ Snapshot failed: no frame captured within timeout")
                return False
                
            finally:

                self.stop_realtime_acquisition()
                print("Temporary acquisition stopped")
                
        except Exception as e:
            print(f"❌ Snapshot error: {e}")
            return False

    def _get_latest_frame_for_snapshot(self):
    
        try:

            for _ in range(3):
                try:
                    self._datastream.KillWait()
                except Exception:
                    pass
            

            for attempt in range(5):
                img = self.get_data_stream_image()
                if img is not None:
                    return img
                time.sleep(0.02) 
            
            return None
            
        except Exception as e:
            print(f"Error getting latest frame: {e}")
            return None


    def shutdown(self):
       
        self.killed = True
        self._acq_stop.set()


        try:
            self.stop_recording()
        except Exception:
            pass


        try:
            self.stop_realtime_acquisition()
        except Exception:
            pass
        try:
            self.stop_hardware_acquisition()
        except Exception:
            pass


        self._stop_background_workers()


        self._teardown_stream_and_device()

    def close(self):
        self.shutdown()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass



    def _open_device(self):
        self.device_manager.Update()
        if self.device_manager.Devices().empty():
            raise RuntimeError("No IDS Peak device found")

        self._device = self.device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
        self.node_map = self._device.RemoteDevice().NodeMaps()[0]


        try:
            self.node_map.FindNode("GainSelector").SetCurrentEntry("AnalogAll")
            self.max_gain = self.node_map.FindNode("Gain").Maximum()
        except Exception:
            self.max_gain = 1.0
        try:
            self.node_map.FindNode("UserSetSelector").SetCurrentEntry("Default")
            self.node_map.FindNode("UserSetLoad").Execute()
            self.node_map.FindNode("UserSetLoad").WaitUntilDone()
        except Exception:
            pass

    def _apply_defaults(self):

        self._find_and_set_enum("GainAuto", "Off")
        self._find_and_set_enum("ExposureAuto", "Off")

        try:
            self.node_map.FindNode("AcquisitionFrameRate").SetValue(DEFAULT_FPS)
            print(f"Acquisition frame rate set to {DEFAULT_FPS} FPS")
        except Exception:
            pass

    def _init_data_stream(self):
        self._datastream = self._device.DataStreams()[0].OpenDataStream()
        self.revoke_and_allocate_buffer()   

    def _teardown_stream_and_device(self):
        t = self._acq_thread
        self._acq_thread = None
        self.acquisition_thread = None 
        if t and t.is_alive():
            try: t.join(timeout=2.0)
            except Exception: pass


        if self._datastream is not None:
            try:
                for b in list(self._datastream.AnnouncedBuffers()):
                    self._datastream.RevokeBuffer(b)
            except Exception:
                pass
            try:
                self._datastream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
            except Exception:
                pass
            try:
                self._datastream.Close()
            except Exception:
                pass
            self._datastream = None


        if self._device is not None:
            try:
                self._device.Close()
            except Exception:
                pass
            self._device = None



    def _start_background_workers(self):
        if not self.recording_worker_running:
            self.recording_worker_running = True
            self.thread_pool.submit(self._recording_worker)
        if not self.save_worker_running:
            self.save_worker_running = True
            self.thread_pool.submit(self._save_worker)

    def _stop_background_workers(self):

        try: self.recording_queue.put_nowait(None)
        except Exception: pass
        try: self.save_queue.put_nowait(None)
        except Exception: pass


        try:
            self.thread_pool.shutdown(wait=True, cancel_futures=True)
        except TypeError:
            self.thread_pool.shutdown(wait=True)
        except Exception:
            pass

        self.recording_worker_running = False
        self.save_worker_running = False


    def _recording_worker(self):
        while True:
            item = self.recording_queue.get()
            try:
                if item is None:
                    self.recording_queue.task_done()
                    break
                self.video_recorder.add_frame(item)
            except Exception as e:
                print(f"Recording worker error: {e}")
            finally:
                if item is not None:
                    self.recording_queue.task_done()

    def _save_worker(self):
        while True:
            item = self.save_queue.get()
            try:
                if item is None:
                    self.save_queue.task_done()
                    break
                save_path, ipl_img = item
                ids_peak_ipl.ImageWriter.WriteAsPNG(save_path, ipl_img)
            except Exception as e:
                print(f"Save worker error: {e}")
            finally:
                if item is not None:
                    self.save_queue.task_done()




    def _queue_all_buffers(self):
        for b in self._buffer_list:
            try:
                self._datastream.QueueBuffer(b)
            except Exception:
                pass

    def start_realtime_acquisition(self) -> bool:
        if self._device is None or self.acquisition_running:
            return False
        if self._datastream is None:
            self._init_data_stream()
        self.acquisition_mode = 0
        self._queue_all_buffers()
        try:
            self._select_trigger("Off", None)
            try:
                self.node_map.FindNode("TLParamsLocked").SetValue(1)
            except Exception:
                pass
            self._datastream.StartAcquisition()
            self.node_map.FindNode("AcquisitionStart").Execute()
            self.acquisition_running = True
            return True
        except Exception as e:
            print(f"start_realtime_acquisition failed: {e}")
            return False

    def stop_realtime_acquisition(self):
        if self._device is None or not self.acquisition_running or self.acquisition_mode != 0:
            return
        self._stop_acquisition_stream("RT")

    def start_hardware_acquisition(self) -> bool:
        if self._device is None or self.acquisition_running:
            return False
        if self._datastream is None:
            self._init_data_stream()
        self.acquisition_mode = 1
        self._queue_all_buffers()
        try:
            self._select_trigger("On", "Software")

            try:
                self.node_map.FindNode("TLParamsLocked").SetValue(1)
            except Exception:
                pass

            self._datastream.StartAcquisition()
            self.node_map.FindNode("AcquisitionStart").Execute()
            self.acquisition_running = True
            print("Hardware Acquisition started! (Software trigger mode)")
            trigger_node = self.node_map.FindNode("TriggerSoftware")
            trigger_node.Execute()
            print("📸 Fired first software trigger")

            return True
        except Exception as e:
            print(f"start_hardware_acquisition failed: {e}")
            return False



    def stop_hardware_acquisition(self):
        if self._device is None or not self.acquisition_running or self.acquisition_mode != 1:
            return
        self._stop_acquisition_stream("HW")

    def _stop_acquisition_stream(self, label: str):
        try: self.node_map.FindNode("AcquisitionStop").Execute()
        except Exception: pass
        try: self._datastream.KillWait()
        except Exception: pass
        try: self._datastream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
        except Exception: pass
        try: self._datastream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
        except Exception: pass

        self.acquisition_running = False
        try:
            self.node_map.FindNode("TLParamsLocked").SetValue(0)
        except Exception:
            pass
        self.revoke_and_allocate_buffer()
        print(f"Closed {label} Acq")

    def _select_trigger(self, mode: str, source: Optional[str]):

        try:
            entries = self.node_map.FindNode("TriggerSelector").Entries()
            symbols = [e.SymbolicValue() for e in entries
                       if e.AccessStatus() not in (ids_peak.NodeAccessStatus_NotAvailable,
                                                   ids_peak.NodeAccessStatus_NotImplemented)]
            sel = "ExposureStart" if "ExposureStart" in symbols else (symbols[0] if symbols else None)
            if sel:
                self.node_map.FindNode("TriggerSelector").SetCurrentEntry(sel)
        except Exception:
            pass


        try:
            self.node_map.FindNode("TriggerMode").SetCurrentEntry(mode)
        except Exception:
            pass


        if mode == "On" and source:
            try:
                self.node_map.FindNode("TriggerSource").SetCurrentEntry(source)
                self.node_map.FindNode("TriggerActivation").SetCurrentEntry("RisingEdge")
            except Exception:
                pass

    def revoke_and_allocate_buffer(self):
        if self._datastream is None:
            return
        try:
            for b in list(self._datastream.AnnouncedBuffers()):
                self._datastream.RevokeBuffer(b)
        except Exception:
            pass

        try:
            payload_size = int(self.node_map.FindNode("PayloadSize").Value())
        except Exception:
            payload_size = 0

        try:
            min_required = self._datastream.NumBuffersAnnouncedMinRequired()
        except Exception:
            min_required = 4

        nbuf = max(min_required, DEFAULT_BUFFERS)
        self._buffer_list = []
        for _ in range(nbuf):
            if payload_size > 0:
                b = self._datastream.AllocAndAnnounceBuffer(payload_size)
            else:

                b = self._datastream.AllocAndAnnounceBuffer()
            self._buffer_list.append(b)


    def conversion_supported(self, source_pixel_format: int) -> bool:
        try:
            outs = self._image_converter.SupportedOutputPixelFormatNames(source_pixel_format)
            return any(TARGET_PIXEL_FORMAT == pf for pf in outs)
        except Exception:
            return False

    def _wait_for_live_fps(self, min_frames: int = 8, timeout: float = 3.0) -> int:
        """Wait until at least `min_frames` frames arrive, then estimate FPS.
        Returns 0 if no valid FPS can be estimated within timeout."""
        start_count = self.frame_count
        t0 = time.time()
        while time.time() - t0 < timeout:
            arrived = self.frame_count - start_count
            if arrived >= min_frames:
                fps = self.get_actual_fps()
                if fps > 0:
                    return fps
            time.sleep(0.05)
        return 0



    @pyqtSlot()
    @pyqtSlot(int)
    def start_recording(self, fps: Optional[int] = None):
        if self.is_recording:
            return
        if self._datastream is None:
            self._init_data_stream()

        # Try reading FPS directly from node (reliable in RT mode only)
        if fps is None and self.acquisition_mode == 0:
            try:
                node = self.node_map.FindNode("AcquisitionFrameRate")
                fps = node.Value() if node is not None else None
            except Exception as e:
                print(f"⚠️ Could not read AcquisitionFrameRate: {e}")
                fps = None

        # In HW mode or fallback: wait for live frames to estimate fps
        if not fps or fps <= 0:
            print("⏳ Waiting for frames to estimate FPS...")
            est = self._wait_for_live_fps(min_frames=8, timeout=3.0)
            if est > 0:
                fps = est
                print(f"🎯 Using measured FPS ≈ {fps}")
            else:
                print("🛑 No frames detected. Recording aborted.")
                return

        try:
            self.video_recorder.start_recording(int(fps))
            self.is_recording = True
            self.recordingStarted.emit()
            print(f"🔴 Recording started at {fps} FPS")
        except Exception as e:
            print(f"❌ Failed to start recording: {e}")



    @pyqtSlot()
    def stop_recording(self):
        if not self.is_recording:
            return
        try:
            self.video_recorder.stop_recording()
        except Exception:
            pass
        self.is_recording = False
        self.recordingStopped.emit()



    def start_calibration(self):
        with self.calibration_lock:
            if self.calibration_running:
                print("⚠️ Calibration already in progress"); return
            self.calibration_running = True

        def delayed_capture():
            try:
                save_path = os.path.join(self.asset_dir, "calibration_capture_image.png")
                latest = None
                for _ in range(20):
                    latest = self.get_data_stream_image()
                    if latest is not None: break
                    time.sleep(0.05)
                if latest is None:
                    print("❌ Failed to capture image for calibration")
                    return
                ids_peak_ipl.ImageWriter.WriteAsPNG(save_path, latest)
                self.thread_pool.submit(compute_h)
            finally:
                pass

        def compute_h():
            try:
                from calibration import find_homography

                H = find_homography()
                if H is not None:
                    self.translation_matrix = H
                    img_path = _assets_path("Generated", "custom_registration_image.png")
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if img is not None:
                        self._safe_project(img, H)
                    print("✅ Homography Computed Successfully!")
            except Exception as e:
                print(f"❌ Homography error: {e}")
            finally:
                with self.calibration_lock:
                    self.calibration_running = False


        try:
            img_path = _assets_path("Generated", "custom_registration_image.png")
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is not None:
                self._safe_project(img, None)
            QTimer.singleShot(80, delayed_capture)
        except Exception as e:
            print(f"❌ Error starting calibration: {e}")
            with self.calibration_lock:
                self.calibration_running = False

    def _safe_project(self, img, H):

        try:
            self._interface.on_projection_received(img, H)
        except Exception:
            pass



    def _find_and_set_enum(self, name: str, value: str):
        try:
            node = self.node_map.FindNode(name)
            entries = node.Entries()
            vals = [e.SymbolicValue() for e in entries
                    if e.AccessStatus() not in (ids_peak.NodeAccessStatus_NotAvailable,
                                                ids_peak.NodeAccessStatus_NotImplemented)]
            if value in vals:
                node.SetCurrentEntry(value)
        except Exception:
            pass

    def set_remote_device_value(self, name: str, value):
        try:
            self.node_map.FindNode(name).SetValue(value)
        except ids_peak.Exception:
            try:
                self._interface.warning(f"Could not set value for {name}!")
            except Exception:
                pass



    def _start_acquisition_thread(self):
        if self._acq_thread and self._acq_thread.is_alive():
            return
        self._acq_stop.clear()
        t = threading.Thread(target=self._acquisition_loop,
                             name="AcquisitionLoop", daemon=True)
        self._acq_thread = t
        t.start()


    def acquisition_thread(self):
        self._acquisition_loop()

    def _ui_alive(self) -> bool:

        try:
            import sip
            return not sip.isdeleted(self._interface)
        except Exception:
            return True  

    def _acquisition_loop(self):
        print("Camera acquisition thread started")
        while not self._acq_stop.is_set() and not self.killed:
            try:
                self.get_data_stream_image()
            except Exception as e:
                now = time.time()
                if now - self._last_acq_err_ts > self._acq_err_interval:
                    try:
                        self._interface.warning(f"Acquisition error: {str(e)}")
                    except Exception:
                        pass
                    self._last_acq_err_ts = now
                self.save_image = False

    def get_actual_fps(self) -> int:
        now = time.time()
        self.frame_times.append(now)
        cutoff = now - 2.0
        while self.frame_times and self.frame_times[0] < cutoff:
            self.frame_times.popleft()
        self.GUIfps = int(round(len(self.frame_times) / 2.0)) if len(self.frame_times) > 1 else 0
        return self.GUIfps

    def _update_performance_metrics(self):
        dur = max(1e-6, time.time() - self.start_time)
        self.performance_stats["fps"] = float(self.frame_count) / dur
        try:
            self.performance_metrics.emit(self.performance_stats)
        except Exception:
            pass

    def get_data_stream_image(self):

        if not self.acquisition_running or self._datastream is None or self.killed:
            time.sleep(0.001)
            return None

        timeout = 500 if self.acquisition_mode == 0 else 2000
        try:
            buffer = self._datastream.WaitForFinishedBuffer(timeout)
        except ids_peak.Exception as e:
            s = str(e)
            if "GC_ERR_TIMEOUT" in s or "GC_ERR_ABORT" in s:
                return None
            return None

        if buffer is None:
            if self.acquisition_mode == 1:
                time.sleep(0.01)
            return None

        try:
            ipl = ids_peak_ipl_extension.BufferToImage(buffer)
            if self._dest_pf is None:
                self._dest_pf = self._pick_dest_pf(ipl)
            converted = self._image_converter.Convert(ipl, self._dest_pf)
            try:
                converted_independent = converted.Clone()
            except Exception:
                converted_independent = converted

        finally:
            try:
                self._datastream.QueueBuffer(buffer)
            except Exception:
                pass


        if self._ui_alive():
            try:
                self.frame_ready.emit(converted_independent)
            except Exception:
                pass


        self.frame_count += 1
        if (self.frame_count % 60) == 0:
            try:
                pf = converted.PixelFormat() if hasattr(converted, "PixelFormat") else "?"
            except Exception:
                print(f"[camera] emitted frame #{self.frame_count}")

        rec_img = converted_independent
        try:
            rec_img = converted_independent.Clone()
        except Exception:
            pass

        if self.is_recording:
            try:
                self.recording_queue.put_nowait(rec_img)
            except queue.Full:
                pass


        if self.save_image:
            save_path = self._snapshot_path or self._valid_name(os.path.join(self.save_dir, "image"), ".png")
            try:
                try:
                    save_img = converted_independent.Clone()
                except Exception:
                    save_img = converted_independent
                self.save_queue.put_nowait((save_path, save_img))
                self.save_image = False
                self._snapshot_path = None
            except queue.Full:
                pass



        if (self.frame_count % 120) == 0:
            self._update_performance_metrics()

        return converted_independent

    def _valid_name(self, base: str, ext: str) -> str:
        num = 0
        while True:
            p = f"{base}_{num}{ext}"
            if not os.path.exists(p):
                return p
            num += 1




    def change_hardware_trigger_line(self, new_line: str):
        self.hardware_trigger_line = new_line
        if self.acquisition_running and self.acquisition_mode == 1:
            self.stop_hardware_acquisition()
            QTimer.singleShot(200, self.start_hardware_acquisition)
        return new_line


    def join_workers(self, timeout: float = 2.0):
        t = self._acq_thread
        if t and t.is_alive():
            try: t.join(timeout=timeout)
            except Exception: pass


Camera = OptimizedCamera
