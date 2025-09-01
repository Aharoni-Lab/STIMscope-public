
import os
import cv2
import datetime
import threading
import queue
import numpy as np
import gc
import time
import logging
from pathlib import Path
from typing import Optional, Callable






WRITER_JOIN_TIMEOUT_S = 30.0    
MAX_FRAME_QUEUE_SIZE   = int(os.environ.get("STIM_REC_QMAX", 120))
BATCH_PROCESSING_SIZE  = int(os.environ.get("STIM_REC_BATCH", 4))

class VideoRecorder:

    def __init__(self, interface=None, on_finalized: Optional[Callable[[str], None]] = None):
        self.interface = interface
        self.on_finalized = on_finalized

        self.recording = False
        self._stopping = False
        self._finalized = threading.Event()
        self._abort = threading.Event()

        self.video_writer: Optional[cv2.VideoWriter] = None
        self.video_filename: str = ""

        self._writer_thread: Optional[threading.Thread] = None
        self._q: queue.Queue = queue.Queue(maxsize=MAX_FRAME_QUEUE_SIZE)

        self._stats_lock = threading.Lock()
        self._frames_written = 0
        self._frames_dropped = 0
        self._start_ts = 0.0
        self._fps = 30
        self._frame_size = (1936, 1096)  # default fallback (W,H)

        out_dir = os.environ.get("STIM_SAVE_DIR", "./Saved_Media")
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        print("🎞️ VideoRecorder ready")



    def start_recording(self, fps: int, frame_size: Optional[tuple]=None) -> bool:
        self._abort.clear()
        if self.recording:
            print("Recording already in progress")
            return True
        if self._stopping and not self._finalized.is_set():
            print("Finalize in progress; cannot start yet")
            return False

        self._fps = int(max(1, fps))
        if frame_size and len(frame_size) == 2:
            self._frame_size = (int(frame_size[0]), int(frame_size[1]))  # (W,H)


        if not self._init_writer():
            return False


        with self._stats_lock:
            self._frames_written = 0
            self._frames_dropped = 0
            self._start_ts = time.time()

        self._finalized.clear()
        self._stopping = False
        self.recording = True


        self._writer_thread = threading.Thread(target=self._writer_loop, name="VR-Writer", daemon=True)
        self._writer_thread.start()

        print(f"🔴 Recording started at {self._fps} FPS → {self.video_filename}")
        return True

    def stop_recording(self) -> None:
       
        if not self.recording and (self._stopping or self._finalized.is_set()):
            return


        self.recording = False
        self._stopping = True

        try:
            remaining = self._q.qsize()
        except Exception:
            remaining = -1
        print(f"🛑 Stop requested. Draining {remaining if remaining >= 0 else 'remaining'} frames...")


    def add_frame(self, frame) -> None:
        
        if not self.recording:
            return

        try:
            self._q.put_nowait(frame)
        except queue.Full:
            with self._stats_lock:
                self._frames_dropped += 1

            try:
                _ = self._q.get_nowait()
                self._q.put_nowait(frame)
            except Exception:
                pass

    def cleanup(self):
        try:
            self.stop_recording()
            if self._writer_thread and self._writer_thread.is_alive():
                self._writer_thread.join(timeout=WRITER_JOIN_TIMEOUT_S)
                if self._writer_thread.is_alive():
                    print("Writer still finalizing; forcing abort")
                    self._abort.set()

                    try:
                        while True:
                            self._q.get_nowait()
                    except Exception:
                        pass
                    self._writer_thread.join(timeout=5.0)
            self._writer_thread = None

            if self.video_writer is not None:
                try: self.video_writer.release()
                except Exception: pass
            self.video_writer = None

            while not self._q.empty():
                try:
                    self._q.get_nowait()
                except Exception:
                    break
            gc.collect()
        except Exception as e:
            print(f"VideoRecorder cleanup error: {e}")




    def _init_writer(self, keep_filename: bool = False) -> bool:
       
        try:
            if self.video_writer is not None:
                try: self.video_writer.release()
                except Exception: pass


            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            if not keep_filename or not self.video_filename:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                out_dir = os.environ.get("STIM_SAVE_DIR", "./Saved_Media")
                os.makedirs(out_dir, exist_ok=True)
                self.video_filename = os.path.join(out_dir, f"recording_{ts}.mp4")

            self.video_writer = cv2.VideoWriter(self.video_filename, fourcc, float(self._fps), self._frame_size)

            if not self.video_writer.isOpened():
                print("Failed to open VideoWriter")
                self.video_writer = None
                return False
            return True
        except Exception as e:
            print(f"VideoWriter init failed: {e}")
            self.video_writer = None
            return False


    @staticmethod
    def _to_bgr_numpy(frame):
       
        try:

            if isinstance(frame, np.ndarray):
                if frame.ndim == 2:
                    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                if frame.ndim == 3 and frame.shape[2] == 3:
                    return frame
                if frame.ndim == 3 and frame.shape[2] == 4:
                    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                return None


            w = h = None
            if hasattr(frame, "Width") or hasattr(frame, "width"):
                try:
                    w = int(frame.Width() if hasattr(frame, "Width") else frame.width())
                    h = int(frame.Height() if hasattr(frame, "Height") else frame.height())
                except Exception:
                    return None


                np_buf = None
                for attr in ("get_numpy_1D", "get_numpy_2D", "get_numpy_view", "get_numpy"):
                    fn = getattr(frame, attr, None)
                    if callable(fn):
                        try:
                            np_buf = fn()
                            break
                        except Exception:
                            pass

                if np_buf is None:
                    return None

                arr = np.array(np_buf, dtype=np.uint8, copy=False)

                if arr.ndim == 1:
                    if arr.size == w * h * 4:
                        return cv2.cvtColor(arr.reshape(h, w, 4), cv2.COLOR_BGRA2BGR)
                    if arr.size == w * h * 3:
                        return arr.reshape(h, w, 3)
                    if arr.size == w * h:
                        return cv2.cvtColor(arr.reshape(h, w), cv2.COLOR_GRAY2BGR)
                elif arr.ndim == 2:
                    if arr.shape == (h, w):
                        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                elif arr.ndim == 3:
                    if arr.shape == (h, w, 4):
                        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
                    if arr.shape == (h, w, 3):
                        return arr
            return None
        except Exception:
            return None


    def _writer_loop(self):
       
        batch = []
        last_flush = time.time()

        try:
            while True:

                if not self.recording and self._q.empty():
                    break


                try:
                    item = self._q.get(timeout=0.05)
                    batch.append(item)
                except queue.Empty:
                    pass

                now = time.time()
                if (len(batch) >= BATCH_PROCESSING_SIZE) or (batch and (now - last_flush) > 0.1):

                    bgr_frames = []
                    for f in batch:
                        arr = self._to_bgr_numpy(f)
                        if arr is not None:
                            bgr_frames.append(arr)
                    batch.clear()
                    last_flush = now


                    if bgr_frames and self.video_writer is not None:
                        h, w = bgr_frames[0].shape[:2]
                        if (w, h) != self._frame_size:
                            self._frame_size = (w, h)
                            if not self._init_writer(keep_filename=True):
                                print("Failed to re-init writer with actual frame size")
                                bgr_frames = []


                    if self.video_writer is not None and self.video_writer.isOpened():
                        for fr in bgr_frames:
                            try:
                                self.video_writer.write(fr)
                                with self._stats_lock:
                                    self._frames_written += 1
                            except Exception:
                                pass

                time.sleep(0.001) 

        except Exception as e:
            print(f"Writer loop error: {e}")

        finally:

            try:
                if self.video_writer is not None:
                    self.video_writer.release()
                    self.video_writer = None
            except Exception:
                pass


            with self._stats_lock:
                written = self._frames_written
                dropped = self._frames_dropped
                dur = max(0.001, time.time() - (self._start_ts or time.time()))
                fps_eff = written / dur

            self._finalized.set()
            self._stopping = False
            print(
                f"✅ Recording finalized: {self.video_filename} | frames={written}, "
                f"dropped={dropped}, avg_fps≈{fps_eff:.1f}"
            )


            if self.on_finalized:
                try:
                    self.on_finalized(self.video_filename)
                except Exception as cb_err:
                    print(f"on_finalized callback raised: {cb_err}")
