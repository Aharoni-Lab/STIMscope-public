import json
from typing import Optional

import numpy as np


class ProjectorClient:
    """
    Minimal client to send 8-bit grayscale frames to the projector engine over ZMQ.
    The projector engine listens on tcp://*:5558 and expects multipart messages:
      part1: JSON string with optional {"id": int}
      part2: raw bytes of shape (HEIGHT, WIDTH) = (1080, 1920), dtype=uint8
    """

    def __init__(self, endpoint: str = "tcp://127.0.0.1:5558", width: int = 1920, height: int = 1080):
        import zmq  # local import to avoid hard dependency if not used
        self._zmq = zmq
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUSH)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(endpoint)
        self.width = int(width)
        self.height = int(height)
        # Optional SUB for projector status (pidx/vis_id) to pace patterns precisely
        try:
            self._sub = self._ctx.socket(zmq.SUB)
            self._sub.setsockopt(zmq.LINGER, 0)
            self._sub.setsockopt(zmq.RCVTIMEO, 50)
            self._sub.setsockopt_string(zmq.SUBSCRIBE, "")
            self._sub.connect("tcp://127.0.0.1:5562")
        except Exception:
            self._sub = None

    def close(self):
        try:
            self._sock.close(0)
        except Exception:
            pass
        try:
            if getattr(self, '_sub', None) is not None:
                self._sub.close(0)
        except Exception:
            pass

    def send_gray(self, img_gray: np.ndarray, frame_id: Optional[int] = None, immediate: bool = True, visible_overlay: Optional[bool] = None) -> None:
        if not isinstance(img_gray, np.ndarray):
            raise TypeError("img_gray must be np.ndarray")
        if img_gray.ndim == 3:
            import cv2
            img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
        if img_gray.shape[::-1] != (self.width, self.height):
            import cv2
            img_gray = cv2.resize(img_gray, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        if img_gray.dtype != np.uint8:
            img_gray = img_gray.astype(np.uint8, copy=False)
        meta = {"id": int(frame_id) if frame_id is not None else 0, "immediate": bool(immediate)}
        if visible_overlay is not None:
            meta["visible_id"] = bool(visible_overlay)
        try:
            self._sock.send_multipart([
                json.dumps(meta).encode("utf-8"),
                memoryview(img_gray)
            ], copy=False)
            # If pacing to projector triggers, wait to observe the vis_id change once
            if immediate and self._sub is not None:
                try:
                    # Drain one status frame opportunistically (non-blocking)
                    _ = self._sub.recv(flags=self._zmq.NOBLOCK)
                except Exception:
                    pass
            # Note: GPIO pulse is handled by callers after confirming visibility via PUB/trigger
        except Exception:
            # Best-effort send; drop if engine not present
            self.close()

    def wait_visible(self, expected_vis_id: int, timeout_ms: int = 500) -> Optional[int]:
        """Block until a PUB status reports vis_id == expected_vis_id. Return pidx if matched."""
        if getattr(self, '_sub', None) is None:
            return None
        import time, json
        t_end = time.time() + timeout_ms / 1000.0
        while time.time() < t_end:
            try:
                msg = self._sub.recv(flags=0)
                s = msg.decode('utf-8', errors='ignore')
                data = json.loads(s)
                if int(data.get('vis_id', -1)) == int(expected_vis_id):
                    # Emit GPIO pulse here if enabled (engine confirmed visibility)
                    if getattr(self, '_gpio_enabled', False):
                        try:
                            import Jetson.GPIO as GPIO, time as _t
                            GPIO.output(self._gpio_pin, GPIO.HIGH)
                            _t.sleep(0.001)
                            GPIO.output(self._gpio_pin, GPIO.LOW)
                        except Exception:
                            pass
                    return int(data.get('pidx', 0))
            except Exception:
                pass
        return None

    # --- GPIO trigger out control ---
    def enable_gpio_trigger(self, pin_board: int = 22):
        try:
            import Jetson.GPIO as GPIO
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(pin_board, GPIO.OUT, initial=GPIO.LOW)
            self._gpio_enabled = True
            self._gpio_pin = int(pin_board)
        except Exception:
            self._gpio_enabled = False
            self._gpio_pin = int(pin_board)

    def disable_gpio_trigger(self):
        try:
            import Jetson.GPIO as GPIO
            GPIO.output(getattr(self, '_gpio_pin', 22), GPIO.LOW)
        except Exception:
            pass
        self._gpio_enabled = False

    def wait_next_trigger(self, last_pidx: Optional[int], timeout_ms: int = 500) -> Optional[int]:
        """Block until projector pidx advances beyond last_pidx. Return new pidx."""
        if getattr(self, '_sub', None) is None:
            return None
        if last_pidx is None:
            return None
        import time, json
        t_end = time.time() + timeout_ms / 1000.0
        while time.time() < t_end:
            try:
                msg = self._sub.recv(flags=0)
                s = msg.decode('utf-8', errors='ignore')
                data = json.loads(s)
                pidx = int(data.get('pidx', 0))
                if pidx > int(last_pidx):
                    return pidx
            except Exception:
                pass
        return None


