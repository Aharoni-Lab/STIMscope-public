import time, zmq, json, numpy as np
W, H = 1920, 1080
ENDPOINT = "tcp://127.0.0.1:5558"
FPS = 25.0
INTERVAL = 1.0 / FPS

ctx = zmq.Context.instance()
s = ctx.socket(zmq.PUSH)
# ---- non-blocking sender options ----
s.setsockopt(zmq.SNDHWM, 4)      # small queue
s.setsockopt(zmq.IMMEDIATE, 1)   # fail fast if no peer yet
s.setsockopt(zmq.SNDTIMEO, 0)    # don't block on send
s.connect(ENDPOINT)

def send_mask(mid, img):
    meta = json.dumps({"id": int(mid)}).encode()
    try:
        s.send_multipart([meta, img.tobytes()], flags=zmq.DONTWAIT)
        return True
    except zmq.Again:
        # receiver busy: drop this frame
        return False

def blank():
    return np.zeros((H, W), np.uint8)

def moving_bar(t, speed=400, w=40, val=255):
    img = blank()
    x = int((t * speed) % (W + w)) - w
    x0, x1 = max(0, x), min(W, x + w)
    if x1 > x0: img[:, x0:x1] = val
    return img

def main():
    print("Streaming; Ctrl-C to stop")
    t0 = time.perf_counter()
    next_t = t0
    mid = 0
    
    # Create CSV file for mask sending log
    import csv
    import os
    csv_path = os.path.join(os.getcwd(), "sent_masks.csv")
    with open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["mask_id", "timestamp", "status"])
        
        try:
            while True:
                t = time.perf_counter() - t0
                img = moving_bar(t)
                mid += 1
                timestamp = time.perf_counter()
                ok = send_mask(mid, img)
                
                # Log each mask sent
                status = "sent" if ok else "dropped"
                csv_writer.writerow([mid, timestamp, status])
                csv_file.flush()
                
                if not ok:
                    print(f"drop {mid}")

                next_t += INTERVAL
                current_t = time.perf_counter()
                sleep_s = next_t - current_t
                
                if sleep_s > 0:
                    time.sleep(sleep_s)
                elif sleep_s < -INTERVAL:
                    # More than one frame behind - log but maintain mask sequence
                    drift_frames = int(-sleep_s / INTERVAL)
                    print(f"⚠️  Timing drift: {drift_frames} frames behind at mask {mid}")
                    next_t = current_t  # Resync but don't skip masks
                # else: Slightly behind but within tolerance - continue
        except KeyboardInterrupt:
            print(f"\nStopped by user. Sent masks log saved to: {csv_path}")
        finally:
            s.close()
            ctx.term()

if __name__ == "__main__":
    main()
