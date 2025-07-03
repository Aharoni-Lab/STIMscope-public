# video_recorder.py
import os
import cv2
import datetime
import threading
import queue
import numpy as np
from logbook import Logbook
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMessageBox

class VideoRecorder:
    """Handles video recording for the IDS camera."""

    def __init__(self, interface):
        self.interface = interface
        self.recording = False
        self.video_writer = None
        self.video_filename = ""
        self.video_writer_thread = None
        self.frame_queue = queue.Queue(maxsize=0)
        self.processing_remaining_frames = False

        self.save_dir = "./Saved_Media"
        os.makedirs(self.save_dir, exist_ok=True)

    def init_video_writer(self, fps=30, frame_size=(1936, 1096)):
        """Initialize the video writer."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_filename = os.path.join(self.save_dir, f"recording_{timestamp}.mp4")
        self.video_writer = cv2.VideoWriter(self.video_filename, fourcc, fps, frame_size)

    def start_recording(self, fps):
        """Start video recording in a separate thread."""
        if not self.recording:
            self.recording = True
            self.init_video_writer(fps)
            Logbook.log_INFO(f"🔴 Recording started at {fps} FPS...") 
            self.video_writer_thread = threading.Thread(target=self._video_writer_loop, daemon=True)
            self.video_writer_thread.start()
            

    def stop_recording(self):
        """Stop recording and finalize the video file."""
        if not self.recording:
            return

        self.recording = False
        Logbook.log_INFO(f"🛑 Recording stopped. Finalizing {self.video_filename}...")
        remaining_frames = self.frame_queue.qsize()
        estimated_time = round(remaining_frames / 30, 2)

        self.processing_remaining_frames = True
        QTimer.singleShot(100, self._check_video_writer_status)

        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle("Processing Video")
        msg_box.setText(
            f"Recording stopped.\n"
            f"{remaining_frames} frames are remaining to be processed.\n"
            f"Estimated processing time: {estimated_time} seconds."
        )
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

        self.interface._button_start_recording.setEnabled(False)

    def _check_video_writer_status(self):
        """Check if all frames have been written before finalizing the file."""
        if self.processing_remaining_frames and not self.frame_queue.empty():
            QTimer.singleShot(100, self._check_video_writer_status)
        else:
            self.video_writer_thread.join()
            self.video_writer.release()
            self.video_writer = None
            self.processing_remaining_frames = False
            self.interface._button_start_recording.setEnabled(True)

            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle("Video Processing Complete")
            msg_box.setText("Your video has finished processing and is ready for use!")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec_()

    def _video_writer_loop(self):
        """Continuously write frames to the video file."""
        while self.recording or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=1)
                image_np = np.array(frame.get_numpy_1D(), dtype=np.uint8).reshape((frame.Height(), frame.Width(), 4))
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_BGRA2BGR)
                if self.video_writer is not None:
                    self.video_writer.write(image_bgr)
            except queue.Empty:
                continue
            except Exception as e:
                Logbook.log_ERRO(f"Error processing video frame: {e}")

    def add_frame(self, frame):
        """Add a frame to the queue for recording."""
        if self.recording:
            self.frame_queue.put(frame)