import os, time, zmq, json, numpy as np, argparse, glob
from PIL import Image

W, H = 1920, 1080

def _to_gray_wh(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 3:
        # simple luminance
        img = (0.114*img[:,:,0] + 0.587*img[:,:,1] + 0.299*img[:,:,2]).astype(np.uint8)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = (0.114*img[:,:,0] + 0.587*img[:,:,1] + 0.299*img[:,:,2]).astype(np.uint8)
    elif img.ndim == 2:
        pass
    else:
        img = np.zeros((h, w), np.uint8)
    if img.shape[0] != h or img.shape[1] != w:
        img = np.array(Image.fromarray(img).resize((w, h), resample=Image.BILINEAR))
    return img.astype(np.uint8, copy=False)

def build_patterns(args):
    def blank(val=0):
        return np.full((H, W), val, np.uint8) if val else np.zeros((H, W), np.uint8)

    def moving_bar(t):
        img = blank()
        speed = args.speed
        w = max(1, args.bar_width)
        val = args.value
        x = int((t * speed) % (W + w)) - w
        x0, x1 = max(0, x), min(W, x + w)
        if x1 > x0: img[:, x0:x1] = val
        return img

    def checkerboard(_):
        sz = max(2, args.checker_size)
        img = blank()
        for y in range(0, H, sz):
            for x in range(0, W, sz):
                c = ((x//sz) + (y//sz)) & 1
                if c:
                    img[y:y+sz, x:x+sz] = args.value
        return img

    def solid(_):
        return blank(args.value)

    def circle(_):
        r = max(1, args.radius)
        img = blank()
        cy, cx = H//2, W//2
        y = np.arange(H)[:, None]
        x = np.arange(W)[None, :]
        mask = (x - cx)**2 + (y - cy)**2 <= r*r
        img[mask] = args.value
        return img

    def gradient_sequence():
        # Steps from black to white with optional gamma and hold per step
        n = max(2, int(getattr(args, 'gradient_steps', 6)))
        g = float(getattr(args, 'gradient_gamma', 1.0))
        hold = max(1, int(getattr(args, 'gradient_hold', 10)))
        vals = []
        for i in range(n):
            x = i / float(n - 1)
            if g != 1.0:
                x = x ** g
            v = int(round(x * 255.0))
            vals.append(v)
        seq = []
        for v in vals:
            frame = blank(v)
            for _ in range(hold):
                seq.append(frame.copy())
        return seq

    seq = []
    if args.pattern == "folder":
        files = sorted(glob.glob(os.path.join(args.folder, "*.png")) +
                       glob.glob(os.path.join(args.folder, "*.jpg")) +
                       glob.glob(os.path.join(args.folder, "*.jpeg")) +
                       glob.glob(os.path.join(args.folder, "*.bmp")))
        for fp in files:
            try:
                arr = np.array(Image.open(fp).convert("RGB"))
                seq.append(_to_gray_wh(arr, W, H))
            except Exception:
                pass
        if not seq:
            seq.append(blank())
        return None, seq
    elif args.pattern == "image":
        if os.path.isfile(args.image):
            try:
                arr = np.array(Image.open(args.image).convert("RGB"))
                seq.append(_to_gray_wh(arr, W, H))
            except Exception:
                seq.append(blank())
        else:
            seq.append(blank())
        return None, seq
    elif args.pattern == "segmask":
        # Load labels or masks from NPZ and create a single grayscale frame
        fp = getattr(args, 'roi_npz', '') or os.path.join(os.getcwd(), "rois.npz")
        try:
            data = np.load(fp, allow_pickle=True)
            if 'labels' in data:
                labels = data['labels'].astype(np.int32)
                img = (labels > 0).astype(np.uint8) * 255
            elif 'masks' in data:
                masks = data['masks']
                if isinstance(masks, np.ndarray) and masks.ndim == 3 and masks.shape[0] > 0:
                    img = (masks[0].astype(bool)).astype(np.uint8) * 255
                elif isinstance(masks, list) and len(masks) > 0:
                    img = (np.array(masks[0]).astype(bool)).astype(np.uint8) * 255
                else:
                    img = blank()
            else:
                img = blank()
            # Pad to projector size without scaling if smaller
            ih, iw = img.shape[:2]
            if ih <= H and iw <= W:
                pad_top = (H - ih) // 2
                pad_bottom = H - ih - pad_top
                pad_left = (W - iw) // 2
                pad_right = W - iw - pad_left
                img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
            else:
                img = np.array(Image.fromarray(img).resize((W, H), resample=Image.NEAREST))
            return None, [img]
        except Exception as _e:
            print(f"⚠️  segmask load failed: {_e}")
            return None, [blank()]
    elif args.pattern == "checkerboard":
        return checkerboard, None
    elif args.pattern == "solid":
        return solid, None
    elif args.pattern == "circle":
        return circle, None
    elif args.pattern == "gradient":
        return None, gradient_sequence()
    else:
        return moving_bar, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="tcp://127.0.0.1:5558")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--pattern", default="moving_bar",
                    choices=["moving_bar", "checkerboard", "solid", "circle", "gradient", "image", "folder", "segmask"]) 
    ap.add_argument("--speed", type=float, default=400.0)
    ap.add_argument("--bar-width", dest="bar_width", type=int, default=40)
    ap.add_argument("--value", type=int, default=255)
    ap.add_argument("--checker-size", dest="checker_size", type=int, default=64)
    ap.add_argument("--radius", type=int, default=200)
    ap.add_argument("--image", type=str, default="")
    ap.add_argument("--folder", type=str, default="")
    ap.add_argument("--gradient-steps", dest="gradient_steps", type=int, default=6)
    ap.add_argument("--gradient-hold", dest="gradient_hold", type=int, default=20)
    ap.add_argument("--gradient-gamma", dest="gradient_gamma", type=float, default=2.2)
    ap.add_argument("--prewarp-lut-dir", type=str, default="",
                    help="If set, load cam_from_proj_{x,y}.npy from this dir and prewarp frames")
    ap.add_argument("--roi-npz", type=str, default="",
                    help="Path to rois.npz containing 'labels' or 'masks'")
    args = ap.parse_args()

    global W, H
    # Allow W/H override via env if needed
    try:
        W = int(os.getenv("MASK_W", W))
        H = int(os.getenv("MASK_H", H))
    except Exception:
        pass

    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.PUSH)
    s.setsockopt(zmq.SNDHWM, 4)
    s.setsockopt(zmq.IMMEDIATE, 1)
    s.setsockopt(zmq.SNDTIMEO, 0)
    s.connect(args.endpoint)

    # Optional LUT prewarp (projector expects prewarped content when H is cleared)
    inv_x = inv_y = None
    if args.prewarp_lut_dir:
        try:
            import numpy as _np
            import os as _os
            inv_x = _np.load(_os.path.join(args.prewarp_lut_dir, "cam_from_proj_x.npy")).astype(np.float32)
            inv_y = _np.load(_os.path.join(args.prewarp_lut_dir, "cam_from_proj_y.npy")).astype(np.float32)
        except Exception as _e:
            print(f"⚠️  Failed to load LUTs from {args.prewarp_lut_dir}: {_e}")
            inv_x = inv_y = None

    def _prewarp(img_gray: np.ndarray) -> np.ndarray:
        if inv_x is None or inv_y is None:
            return img_gray
        try:
            import cv2 as _cv2
            h, w = img_gray.shape[:2]
            # Resize LUT if projector size differs
            if inv_x.shape != (H, W):
                _ix = _cv2.resize(inv_x, (W, H), interpolation=_cv2.INTER_LINEAR)
                _iy = _cv2.resize(inv_y, (W, H), interpolation=_cv2.INTER_LINEAR)
            else:
                _ix, _iy = inv_x, inv_y
            # Build camera image (expand gray to BGR for remap, then collapse)
            cam_bgr = _cv2.cvtColor(img_gray, _cv2.COLOR_GRAY2BGR)
            warped = _cv2.remap(cam_bgr, _ix, _iy, interpolation=_cv2.INTER_LINEAR,
                                borderMode=_cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            return _cv2.cvtColor(warped, _cv2.COLOR_BGR2GRAY)
        except Exception as _e:
            print(f"⚠️  LUT prewarp failed: {_e}")
            return img_gray

    def send_mask(mid, img):
        meta = json.dumps({"id": int(mid)}).encode()
        try:
            frame = _prewarp(img)
            s.send_multipart([meta, frame.tobytes()], flags=zmq.DONTWAIT)
            return True
        except zmq.Again:
            return False

    gen_fn, seq = build_patterns(args)

    print("Streaming; Ctrl-C to stop")
    t0 = time.perf_counter()
    next_t = t0
    mid = 0
    INTERVAL = 1.0 / max(1e-6, float(args.fps))

    import csv
    csv_path = os.path.join(os.getcwd(), "sent_masks.csv")
    with open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["mask_id", "timestamp", "status"])
        try:
            idx = 0
            while True:
                if gen_fn is not None:
                    t = time.perf_counter() - t0
                    img = gen_fn(t)
                else:
                    if not seq:
                        img = np.zeros((H, W), np.uint8)
                    else:
                        img = seq[idx % len(seq)]
                        idx += 1

                mid += 1
                timestamp = time.perf_counter()
                ok = send_mask(mid, img)
                csv_writer.writerow([mid, timestamp, ("sent" if ok else "dropped")])
                csv_file.flush()

                next_t += INTERVAL
                current_t = time.perf_counter()
                sleep_s = next_t - current_t
                if sleep_s > 0:
                    time.sleep(sleep_s)
                elif sleep_s < -INTERVAL:
                    drift_frames = int(-sleep_s / INTERVAL)
                    print(f"⚠️  Timing drift: {drift_frames} frames behind at mask {mid}")
                    next_t = current_t
        except KeyboardInterrupt:
            print(f"\nStopped by user. Sent masks log saved to: {csv_path}")
        finally:
            s.close()
            ctx.term()

if __name__ == "__main__":
    main()
