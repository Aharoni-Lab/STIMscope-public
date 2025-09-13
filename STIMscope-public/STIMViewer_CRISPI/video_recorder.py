import os
import cv2
import datetime
import threading
import queue
import numpy as np
import gc
import time
from pathlib import Path
from typing import Optional, Callable
import csv
from collections import deque

try:
    from tifffile import TiffWriter, TiffFile
except Exception as _e:
    TiffWriter = None
    TiffFile = None

WRITER_JOIN_TIMEOUT_S = 30.0
MAX_FRAME_QUEUE_SIZE   = int(os.environ.get("STIM_REC_QMAX", 240))
BATCH_PROCESSING_SIZE  = int(os.environ.get("STIM_REC_BATCH", 8))

# TIFF behavior, configurable via environment variables
TIFF_COMPRESSION = os.environ.get("STIM_TIFF_COMPRESSION", "deflate")  # none, deflate, lzma, zstd, jpeg
TIFF_BIGTIFF     = os.environ.get("STIM_TIFF_BIGTIFF", "").strip()     # "", "0", "1", empty means let tifffile decide
TIFF_GRAYSCALE   = bool(int(os.environ.get("STIM_TIFF_GRAYSCALE", "1")))
TIFF_IMAGEJ_MODE = bool(int(os.environ.get("STIM_TIFF_IMAGEJ", "0")))  # default off, critical for real multipage


# RealTimeSync removed - using projector's mask_map.csv for accurate GPIO-based synchronization


class VideoRecorder:
    """
    Records incoming frames into a single multi page TIFF file.
    API matches the previous mp4 based recorder.
    """

    def __init__(self, interface=None, on_finalized: Optional[Callable[[str], None]] = None):
        self.interface = interface
        self.on_finalized = on_finalized

        self.recording = False
        self._stopping = False
        self._finalized = threading.Event()
        self._abort = threading.Event()

        self.video_writer = None      # TiffWriter
        self.video_filename: str = "" # path to .tiff

        self._writer_thread: Optional[threading.Thread] = None
        self._q: queue.Queue = queue.Queue(maxsize=MAX_FRAME_QUEUE_SIZE)

        self._frames_written = 0
        self._frames_dropped = 0
        self._start_ts = 0.0
        self._fps = 30
        self._frame_size = (1936, 1096)  # (W, H) default fallback
        self._locked_shape = None        # only used if ImageJ mode is enabled
        self._locked_dtype = None

        out_dir = os.environ.get("STIM_SAVE_DIR", "./Saved_Media")
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        print("VideoRecorder ready - using projector's mask_map.csv for synchronization")

    def start_recording(self, fps: int, frame_size: Optional[tuple]=None) -> bool:
        if TiffWriter is None:
            print("tifffile is required, install with: pip install tifffile")
            return False

        self._abort.clear()
        if self.recording:
            print("Recording already in progress")
            return True
        if self._stopping and not self._finalized.is_set():
            print("Finalize in progress, cannot start yet")
            return False

        self._fps = int(max(1, fps))
        if frame_size and len(frame_size) == 2:
            self._frame_size = (int(frame_size[0]), int(frame_size[1]))

        if not self._init_writer():
            return False

        self._frames_written = 0
        self._frames_dropped = 0
        self._start_ts = time.time()
        self._finalized.clear()
        self._stopping = False
        self.recording = True

        self._writer_thread = threading.Thread(target=self._writer_loop, name="VR-Writer", daemon=True)
        self._writer_thread.start()

        print(f"Recording started at {self._fps} FPS, writing to {self.video_filename}")
        if TIFF_IMAGEJ_MODE:
            print("Note, ImageJ mode is ON, frames must keep the same shape and dtype")
        else:
            print("ImageJ mode is OFF, generic multi page TIFF will be written")
        print("Note: Mask-to-frame mapping handled by projector's mask_map.csv")
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
        print(f"Stop requested, draining {remaining if remaining >= 0 else 'remaining'} frames")
# Synchronization handled by projector system

    def add_frame(self, frame) -> None:
        if not self.recording:
            return
        try:
            self._q.put_nowait(frame)
        except queue.Full:
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
                    print("Writer still finalizing, forcing abort")
                    self._abort.set()
                    try:
                        while True:
                            self._q.get_nowait()
                    except Exception:
                        pass
                    self._writer_thread.join(timeout=5.0)
            self._writer_thread = None

            if self.video_writer is not None:
                try: self.video_writer.close()
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

    def _init_writer(self) -> bool:
        try:
            if self.video_writer is not None:
                try: self.video_writer.close()
                except Exception: pass
                self.video_writer = None

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.environ.get("STIM_SAVE_DIR", "./Saved_Media")
            os.makedirs(out_dir, exist_ok=True)
            self.video_filename = os.path.join(out_dir, f"recording_{ts}.tiff")

            if TIFF_BIGTIFF.strip() in ("0", "1"):
                bigtiff = bool(int(TIFF_BIGTIFF))
            else:
                bigtiff = None  # let tifffile decide

            # Important, ImageJ off by default to avoid single image header
            self.video_writer = TiffWriter(self.video_filename, bigtiff=bigtiff, imagej=TIFF_IMAGEJ_MODE)
            self._locked_shape = None
            self._locked_dtype = None
            return True
        except Exception as e:
            print(f"TIFF writer init failed: {e}")
            self.video_writer = None
            return False

    @staticmethod
    def _to_numpy(frame) -> Optional[np.ndarray]:
        try:
            if isinstance(frame, np.ndarray):
                return frame

            # Vendor frame path
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

                arr = np.array(np_buf, copy=False)
                if arr.ndim == 1:
                    if arr.size == w * h:
                        return arr.reshape(h, w)
                    if arr.size == w * h * 3:
                        return arr.reshape(h, w, 3)
                    if arr.size == w * h * 4:
                        return arr.reshape(h, w, 4)
                return arr
            return None
        except Exception:
            return None

    def _prep_frame(self, arr: np.ndarray) -> Optional[np.ndarray]:
        if arr is None:
            return None

        if not isinstance(arr, np.ndarray):
            try:
                arr = np.array(arr)
            except Exception:
                return None

        # Normalize dtype
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            a = np.clip(arr, 0, 1) if arr.max() <= 1.5 else np.clip(arr, 0, 255) / 255.0
            arr = (a * 255.0).astype(np.uint8)
        elif arr.dtype not in (np.uint8, np.uint16, np.int16, np.uint32):
            arr = arr.astype(np.uint8, copy=False)

        # Channels
        if arr.ndim == 3:
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            if arr.shape[2] == 3 and TIFF_GRAYSCALE:
                try:
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
                except Exception:
                    r = arr[:, :, 2].astype(np.float32)
                    g = arr[:, :, 1].astype(np.float32)
                    b = arr[:, :, 0].astype(np.float32)
                    arr = (0.299 * r + 0.587 * g + 0.114 * b).astype(arr.dtype)

        return np.ascontiguousarray(arr)

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
                    frames_np = []
                    for f in batch:
                        arr = self._to_numpy(f)
                        arr = self._prep_frame(arr)
                        if arr is not None:
                            frames_np.append(arr)
                        else:
                            self._frames_dropped += 1
                    batch.clear()
                    last_flush = now

                    if frames_np and self.video_writer is not None:
                        if TIFF_IMAGEJ_MODE and self._locked_shape is None:
                            self._locked_shape = frames_np[0].shape
                            self._locked_dtype = frames_np[0].dtype

                        for fr in frames_np:
                            try:
                                # In ImageJ mode enforce constant shape and dtype
                                if TIFF_IMAGEJ_MODE:
                                    if fr.shape != self._locked_shape or fr.dtype != self._locked_dtype:
                                        self._frames_dropped += 1
                                        continue

                                photometric = "minisblack" if fr.ndim == 2 else "rgb"
                                self.video_writer.write(
                                    fr,
                                    photometric=photometric,
                                    compression=None if TIFF_COMPRESSION.lower() == "none" else TIFF_COMPRESSION,
                                    metadata=None  # keep simple to avoid single image IJ header issues
                                )
                                self._frames_written += 1
                            except Exception:
                                self._frames_dropped += 1

                time.sleep(0.001)

        except Exception as e:
            print(f"Writer loop error: {e}")

        finally:
            # Close writer
            try:
                if self.video_writer is not None:
                    self.video_writer.close()
            except Exception:
                pass
            self.video_writer = None

            dur = max(0.001, time.time() - (self._start_ts or time.time()))
            fps_eff = self._frames_written / dur
            print(
                f"Recording finalized, file {self.video_filename}, "
                f"frames={self._frames_written}, dropped={self._frames_dropped}, avg_fps={fps_eff:.1f}"
            )

            # Quick verification of page count
            try:
                if TiffFile is not None:
                    with TiffFile(self.video_filename) as tf:
                        n_pages = len(tf.pages)
                    print(f"Verify, TIFF pages detected: {n_pages}")
            except Exception as ver_e:
                print(f"Verify failed: {ver_e}")

            self._finalized.set()
            self._stopping = False

            if self.on_finalized:
                try:
                    self.on_finalized(self.video_filename)
                except Exception as cb_err:
                    print(f"on_finalized callback raised: {cb_err}")
