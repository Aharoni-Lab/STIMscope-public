# send_mask_400us_then_black.py
import json, time, zmq, numpy as np
import Jetson.GPIO as GPIO

WIDTH, HEIGHT = 1920, 1080
TRIG_PIN_BOARD = 22           # J30 pin 22 (BOARD numbering) -> GPIO17
DURATION_US = 8000000             # keep bright for 400 microseconds (sender-side)

# --- GPIO setup (optional gate/marker) ---
GPIO.setmode(GPIO.BOARD)
GPIO.setup(TRIG_PIN_BOARD, GPIO.OUT, initial=GPIO.LOW)

# --- Prepare frames ---
bright = np.zeros((HEIGHT, WIDTH), np.uint8)
bright[190:690, 170:1150] = 255      # bright ROI
black  = np.zeros((HEIGHT, WIDTH), np.uint8)

# --- ZMQ PUSH socket ---
ctx = zmq.Context.instance()
s = ctx.socket(zmq.PUSH)
s.setsockopt(zmq.LINGER, 0)
s.connect("tcp://127.0.0.1:5556")

def send_frame(arr, meta_extra=None):
    meta = {
        "width": WIDTH, "height": HEIGHT, "channels": 1, "dtype": "uint8",
        "sent_unix_ns": time.time_ns()
    }
    if meta_extra:
        meta.update(meta_extra)
    s.send_multipart([json.dumps(meta).encode("utf-8"), memoryview(arr)], copy=False)

def busy_wait_us(us):
    t_end = time.perf_counter_ns() + us * 1_000
    while time.perf_counter_ns() < t_end:
        pass

# --- Sequence: BRIGHT for 400 µs, then BLACK ---
GPIO.output(TRIG_PIN_BOARD, GPIO.HIGH)  # optional: gate/marker HIGH during 'bright'
send_frame(bright, {"event": "bright", "duration_us": DURATION_US})
busy_wait_us(DURATION_US)               # ~400 µs in user space
send_frame(black,  {"event": "black", "after_us": DURATION_US})
GPIO.output(TRIG_PIN_BOARD, GPIO.LOW)

print(f"bright -> {DURATION_US}us -> black (queued)")

# Clean up
GPIO.cleanup()
