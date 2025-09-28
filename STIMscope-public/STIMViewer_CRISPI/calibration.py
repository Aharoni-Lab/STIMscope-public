
from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw
try:
    # OpenCV ArUco (contrib) for robust marker detection
    from cv2 import aruco as cv2_aruco
except Exception:
    cv2_aruco = None


ASSETS = (Path(__file__).resolve().parent / "Assets").resolve()
GEN_DIR = (ASSETS / "Generated").resolve()
GEN_DIR.mkdir(parents=True, exist_ok=True)

REF_REG_IMG = GEN_DIR / "custom_registration_image.png"
CALIB_CAPTURE_IMG = GEN_DIR / "calibration_capture_image.png"
CALIB_OUTPUT_IMG = GEN_DIR / "CalibOutput.jpg"
HOMOGRAPHY_NPY = GEN_DIR / "homography_cam2proj.npy"
SL_DIR = (GEN_DIR / "GrayCode").resolve()
SL_DIR.mkdir(parents=True, exist_ok=True)

PROJ_FROM_CAM_X_NPY = GEN_DIR / "proj_from_cam_x.npy"
PROJ_FROM_CAM_Y_NPY = GEN_DIR / "proj_from_cam_y.npy"
CAM_FROM_PROJ_X_NPY = GEN_DIR / "cam_from_proj_x.npy"
CAM_FROM_PROJ_Y_NPY = GEN_DIR / "cam_from_proj_y.npy"





def create_custom_registration_image(
    width: int,
    height: int,
    line_color: Tuple[int, int, int] | str = "white",
    fill_color: Tuple[int, int, int] | str = "white",
    save_path: Path = REF_REG_IMG,
) -> Path:
   
    img = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(img)
    
    print(f"🎨 Creating enhanced calibration pattern ({width}x{height})")

    large_font_size = max(200, min(width, height) // 2)  
    number_font_size = max(80, min(width, height) // 5)
    chessboard_size = 8
    chessboard_cell_size = max(20, min(width, height) // 40)
    circle_radius = min(width, height) // 4
    cross_size = max(120, min(width, height) // 4)
    gradient_bar_width = max(100, width // 10)
    circle_thickness = max(4, width // 500)
    cross_thickness = max(12, width // 160)
    f_thickness = max(8, width // 40)


    x = width // 2 - large_font_size // 2
    y = height // 2 - large_font_size // 2
    lw = f_thickness
    draw.line([(x, y), (x + int(large_font_size * 0.8), y)], fill=line_color, width=lw)             # Top
    draw.line([(x, y), (x, y + int(large_font_size * 0.6))], fill=line_color, width=lw)             # Vertical
    draw.line([(x, y + int(large_font_size * 0.4)),
               (x + int(large_font_size * 0.6), y + int(large_font_size * 0.4))],
              fill=line_color, width=lw)                                                             # Middle


    number_positions = [
        (width // 4 - number_font_size // 2, height // 4 - number_font_size // 2),
        (3 * width // 4 - number_font_size // 2, height // 4 - number_font_size // 2),
        (width // 4 - number_font_size // 2, 3 * height // 4 - number_font_size // 2),
        (3 * width // 4 - number_font_size // 2, 3 * height // 4 - number_font_size // 2),
        (width // 4 - number_font_size // 2, height // 2 - number_font_size // 2),
        (3 * width // 4 - number_font_size // 2, height // 2 - number_font_size // 2),
    ]
    for number, pos in zip(range(1, 7), number_positions):
        draw_number(draw, pos, number, number_font_size, line_color)


    for i in range(gradient_bar_width):
        g = int(i * 255 / max(1, gradient_bar_width - 1))
        draw.line([(i, 0), (i, height)], fill=(g, g, g), width=1)


    for i in range(5):
        inset = i * max(10, width // 200)
        draw.ellipse(
            [(width - circle_radius - inset, inset),
             (width - inset, circle_radius + inset)],
            outline=line_color, width=circle_thickness
        )


    cb_w = chessboard_size * chessboard_cell_size
    cb_h = cb_w
    chessboard_start_x = (width - cb_w) // 2
    chessboard_start_y = height - cb_h - 20  # Add margin
    
    for i in range(chessboard_size):
        for j in range(chessboard_size):
            tl = (chessboard_start_x + i * chessboard_cell_size,
                  chessboard_start_y + j * chessboard_cell_size)
            br = (tl[0] + chessboard_cell_size, tl[1] + chessboard_cell_size)
            fill = fill_color if ((i + j) % 2 == 0) else "black"
            draw.rectangle([tl, br], fill=fill)
    

    corner_size = max(30, min(width, height) // 60)
    corner_offset = 20
    corners = [
        (corner_offset, corner_offset),  # Top-left
        (width - corner_offset - corner_size, corner_offset),  # Top-right
        (corner_offset, height - corner_offset - corner_size),  # Bottom-left
        (width - corner_offset - corner_size, height - corner_offset - corner_size)  # Bottom-right
    ]
    
    for corner in corners:

        draw.rectangle([corner, (corner[0] + corner_size, corner[1] + corner_size)], 
                      fill=line_color, outline="black", width=2)

        inner_size = corner_size // 3
        inner_corner = (corner[0] + inner_size, corner[1] + inner_size)
        draw.rectangle([inner_corner, (inner_corner[0] + inner_size, inner_corner[1] + inner_size)], 
                      fill="black")


    cx, cy = (cross_size, cross_size)
    draw.line([(cx - cross_size, cy), (cx + cross_size, cy)], fill=line_color, width=cross_thickness)
    draw.line([(cx, cy - cross_size), (cx, cy + cross_size)], fill=line_color, width=cross_thickness)


    draw_smiley_face(draw, (width - 900, height - 700), 50, line_color)
    draw_smiley_face(draw, (width - 1000, height - 950), 100, line_color)

    img.save(save_path.as_posix())

    # Overlay ArUco markers in the four corners to provide robust feature anchors
    try:
        if cv2_aruco is not None:
            bgr = cv2.imread(save_path.as_posix(), cv2.IMREAD_COLOR)
            h, w = bgr.shape[:2]
            marker_size = max(48, min(w, h) // 14)
            margin = max(16, marker_size // 6)
            ar_dict = cv2_aruco.getPredefinedDictionary(cv2_aruco.DICT_4X4_50)
            ids = [11, 22, 33, 44]
            # Positions: TL, TR, BR, BL
            positions = [
                (margin, margin),
                (w - margin - marker_size, margin),
                (w - margin - marker_size, h - margin - marker_size),
                (margin, h - margin - marker_size),
            ]
            for mid, (px, py) in zip(ids, positions):
                marker = cv2_aruco.generateImageMarker(ar_dict, mid, marker_size)
                if marker.ndim == 2:
                    marker = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
                bgr[py:py+marker_size, px:px+marker_size] = marker
            cv2.imwrite(save_path.as_posix(), bgr)
    except Exception:
        pass

    print(f"✅ Custom registration image saved: {save_path}")
    return save_path


def create_charuco_registration_image(width: int, height: int, save_path: Path = REF_REG_IMG) -> Path:
    """Create a Charuco (chessboard + ArUco) registration image sized to projector."""
    if cv2_aruco is None:
        # Fallback: use existing generator if aruco unavailable
        return create_custom_registration_image(width, height, (255, 255, 255), (255, 255, 255), save_path)

    # Choose a board that fits screen well
    squares_x = 12
    squares_y = 8
    # Make square size ~ 1/10 of min dimension
    square_size = max(20, min(width, height) // 10)
    marker_size = int(square_size * 0.6)

    ar_dict = cv2_aruco.getPredefinedDictionary(cv2_aruco.DICT_4X4_100)
    board = cv2_aruco.CharucoBoard((squares_x, squares_y), square_size, marker_size, ar_dict)

    # Render at native scale
    img = board.generateImage((width, height), marginSize=20, borderBits=1)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(save_path.as_posix(), img)
    print(f"✅ Charuco registration image saved: {save_path}")
    return save_path


# ------------------------
# Structured-Light (Gray Code)
# ------------------------

def _gray_code(n: int) -> int:
    return n ^ (n >> 1)

def _num_bits(n: int) -> int:
    b = 0
    v = max(1, n - 1)
    while v:
        b += 1
        v >>= 1
    return b

def generate_gray_code_patterns(width: int, height: int) -> list[dict]:
    """Generate Gray-code patterns for projector resolution.
    Returns list of dicts: {"axis": "x"|"y", "bit": int, "inv": bool, "image": np.ndarray BGR}
    """
    patterns: list[dict] = []
    bx = _num_bits(width)
    by = _num_bits(height)
    # X-axis patterns
    cols = np.arange(width, dtype=np.int32)
    gray_cols = _gray_code(cols)
    for bit in range(bx - 1, -1, -1):
        bitplane = ((gray_cols >> bit) & 1).astype(np.uint8)
        img = np.repeat(bitplane[None, :, None], height, axis=0) * 255
        img = np.repeat(img, 3, axis=2)
        patterns.append({"axis": "x", "bit": bit, "inv": False, "image": img})
        inv = 255 - img
        patterns.append({"axis": "x", "bit": bit, "inv": True, "image": inv})
    # Y-axis patterns
    rows = np.arange(height, dtype=np.int32)
    gray_rows = _gray_code(rows)
    for bit in range(by - 1, -1, -1):
        bitplane = ((gray_rows >> bit) & 1).astype(np.uint8)
        img = np.repeat(bitplane[:, None, None], width, axis=1) * 255
        img = np.repeat(img, 3, axis=2)
        patterns.append({"axis": "y", "bit": bit, "inv": False, "image": img})
        inv = 255 - img
        patterns.append({"axis": "y", "bit": bit, "inv": True, "image": inv})
    return patterns

def save_gray_code_patterns(patterns: list[dict], out_dir: Path = SL_DIR) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx, p in enumerate(patterns):
        axis = p["axis"]
        bit = p["bit"]
        inv = "inv" if p["inv"] else "pos"
        path = out_dir / f"pat_{idx:03d}_{axis}_b{bit}_{inv}.png"
        cv2.imwrite(path.as_posix(), p["image"])
        paths.append(path.as_posix())
    return paths

def save_structured_light_patterns(patterns: list[dict], out_dir: Path = SL_DIR) -> list[str]:
    """Save a heterogeneous list of structured-light patterns (Gray + Phase).
    Supports entries from generate_gray_code_patterns(...) and
    generate_phase_shift_patterns(...).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for idx, p in enumerate(patterns):
        try:
            img = p.get("image")
            if img is None:
                continue
            axis = p.get("axis", "x")
            if p.get("type") == "phase":
                k = int(p.get("phase_idx", idx))
                path = out_dir / f"pat_{idx:03d}_{axis}_phase_{k}.png"
            else:
                # Backward-compatible Gray-code naming
                bit = int(p.get("bit", -1))
                inv = "inv" if bool(p.get("inv", False)) else "pos"
                path = out_dir / f"pat_{idx:03d}_{axis}_b{bit}_{inv}.png"
            cv2.imwrite(path.as_posix(), img)
            paths.append(path.as_posix())
        except Exception:
            continue
    return paths

def decode_gray_code_from_files(capture_files: list[str], pattern_meta: list[dict], cam_h: int, cam_w: int, proj_w: int, proj_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode Gray code projection captures to per-camera-pixel projector coords.
    Returns (proj_x_of_cam [H,W], proj_y_of_cam [H,W]) as float32 with -1 for invalid.
    capture_files must align 1:1 with pattern_meta order.
    """
    # Load all captures in grayscale
    captures = []
    for fp in capture_files:
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((cam_h, cam_w), dtype=np.uint8)
        captures.append(img)
    bx = _num_bits(proj_w)
    by = _num_bits(proj_h)
    # Initialize bit stacks
    bits_x = [None] * bx
    bits_y = [None] * by
    # For each bit we have pair (pos, inv) to decide bit by comparing
    for cap, meta in zip(captures, pattern_meta):
        axis = meta["axis"]
        bit = meta["bit"]
        inv = meta["inv"]
        if axis == "x":
            if bits_x[bit] is None:
                bits_x[bit] = {"pos": None, "inv": None}
            key = "inv" if inv else "pos"
            bits_x[bit][key] = cap
        else:
            if bits_y[bit] is None:
                bits_y[bit] = {"pos": None, "inv": None}
            key = "inv" if inv else "pos"
            bits_y[bit][key] = cap
    # Decide bits via pos>inv - simple threshold but with diagnostics
    def _decide(bits_list, num_bits):
        vals = np.zeros((cam_h, cam_w), dtype=np.int32)
        valid = np.ones((cam_h, cam_w), dtype=bool)
        contrast_sum = 0
        for b in range(num_bits - 1, -1, -1):
            pair = bits_list[b]
            if pair is None or pair["pos"] is None or pair["inv"] is None:
                valid[:] = False
                continue
            # Simple difference threshold
            diff = pair["pos"].astype(np.int16) - pair["inv"].astype(np.int16)
            
            # Log contrast for diagnostics
            contrast = np.mean(np.abs(diff))
            contrast_sum += contrast
            
            # Simple threshold at 0
            bitmask = diff > 0
            vals = (vals << 1) | bitmask.astype(np.int32)
        
        # Print average contrast
        if num_bits > 0:
            avg_contrast = contrast_sum / num_bits
            print(f"Gray code average contrast: {avg_contrast:.1f} gray levels")
        
        return vals, valid
    gray_x, valid_x = _decide(bits_x, bx)
    gray_y, valid_y = _decide(bits_y, by)
    # Gray to binary
    def _gray_to_bin(arr: np.ndarray) -> np.ndarray:
        g = arr.copy()
        b = np.zeros_like(g)
        while True:
            b ^= g
            g >>= 1
            if not g.any():
                break
        return b
    bin_x = _gray_to_bin(gray_x)
    bin_y = _gray_to_bin(gray_y)
    # Map directly to projector pixel coordinates.
    # bin_x, bin_y already represent 0..(width-1)/(height-1) because we encoded rows/cols in that range.
    proj_x = bin_x.astype(np.float32)
    proj_y = bin_y.astype(np.float32)
    # Clamp to projector bounds in case of sparse/invalid decodes near edges
    proj_x = np.clip(proj_x, 0.0, float(proj_w - 1))
    proj_y = np.clip(proj_y, 0.0, float(proj_h - 1))
    valid = valid_x & valid_y
    proj_x[~valid] = -1.0
    proj_y[~valid] = -1.0
    return proj_x.astype(np.float32), proj_y.astype(np.float32)

def invert_cam_to_proj_lut(proj_x_of_cam: np.ndarray, proj_y_of_cam: np.ndarray, proj_w: int, proj_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute inverse LUT (projector→camera) by bilinear splatting.
    proj_x_of_cam, proj_y_of_cam are per-camera-pixel projector coords in pixels.
    Returns inv_x, inv_y of shape (proj_h, proj_w) with -1 for holes.
    """
    cam_h, cam_w = proj_x_of_cam.shape

    sum_x = np.zeros((proj_h, proj_w), dtype=np.float32)
    sum_y = np.zeros((proj_h, proj_w), dtype=np.float32)
    wts   = np.zeros((proj_h, proj_w), dtype=np.float32)

    valid = (proj_x_of_cam >= 0) & (proj_y_of_cam >= 0)
    ys, xs = np.where(valid)

    for cy, cx in zip(ys.tolist(), xs.tolist()):
        px = float(proj_x_of_cam[cy, cx])
        py = float(proj_y_of_cam[cy, cx])
        if not (0.0 <= px < proj_w and 0.0 <= py < proj_h):
            continue
        x0 = int(np.floor(px)); y0 = int(np.floor(py))
        dx = px - x0;           dy = py - y0
        for j in (0, 1):
            for i in (0, 1):
                xi = x0 + i; yi = y0 + j
                if 0 <= xi < proj_w and 0 <= yi < proj_h:
                    w = (1 - dx if i == 0 else dx) * (1 - dy if j == 0 else dy)
                    sum_x[yi, xi] += w * float(cx)
                    sum_y[yi, xi] += w * float(cy)
                    wts[yi, xi]   += w

    inv_x = np.full((proj_h, proj_w), -1.0, dtype=np.float32)
    inv_y = np.full((proj_h, proj_w), -1.0, dtype=np.float32)
    nonzero = wts > 1e-6
    inv_x[nonzero] = sum_x[nonzero] / wts[nonzero]
    inv_y[nonzero] = sum_y[nonzero] / wts[nonzero]

    # Hole fill by nearest neighbor propagation
    holes = ~nonzero
    if holes.any():
        _, labels = cv2.distanceTransformWithLabels(holes.astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        for y in range(proj_h):
            for x in range(proj_w):
                if holes[y, x]:
                    lab = int(labels[y, x])
                    if lab > 0:
                        vy = (lab - 1) // proj_w
                        vx = (lab - 1) %  proj_w
                        inv_x[y, x] = inv_x[vy, vx]
                        inv_y[y, x] = inv_y[vy, vx]

    return inv_x, inv_y

def prewarp_with_inverse_lut(desired_camera_img_bgr: np.ndarray, inv_x: np.ndarray, inv_y: np.ndarray, proj_w: int, proj_h: int) -> np.ndarray:
    """Create projector image that will appear as desired_camera_img when viewed by camera.
    inv_x, inv_y: for each projector pixel (y,x), the corresponding camera (x,y) coordinate.
    """
    map_x = inv_x.astype(np.float32)
    map_y = inv_y.astype(np.float32)
    cam_h, cam_w = desired_camera_img_bgr.shape[:2]
    # Clamp to camera bounds
    np.clip(map_x, 0.0, float(cam_w - 1), out=map_x)
    np.clip(map_y, 0.0, float(cam_h - 1), out=map_y)
    warped = cv2.remap(desired_camera_img_bgr, map_x, map_y,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT,
                       borderValue=(0, 0, 0))
    return warped

def visualize_lut_quality(inv_x: np.ndarray, inv_y: np.ndarray, save_path: Optional[str] = None) -> np.ndarray:
    """Generate a diagnostic image showing LUT quality and potential issues."""
    proj_h, proj_w = inv_x.shape
    
    # Create diagnostic image with multiple panels
    diag_img = np.zeros((proj_h * 2, proj_w * 2, 3), dtype=np.uint8)
    
    # Panel 1: X coordinate map (top-left)
    x_norm = np.clip((inv_x - inv_x[inv_x >= 0].min()) / 
                     (inv_x[inv_x >= 0].max() - inv_x[inv_x >= 0].min()), 0, 1)
    x_vis = (x_norm * 255).astype(np.uint8)
    diag_img[:proj_h, :proj_w, 0] = x_vis  # Red channel for X
    
    # Panel 2: Y coordinate map (top-right)
    y_norm = np.clip((inv_y - inv_y[inv_y >= 0].min()) / 
                     (inv_y[inv_y >= 0].max() - inv_y[inv_y >= 0].min()), 0, 1)
    y_vis = (y_norm * 255).astype(np.uint8)
    diag_img[:proj_h, proj_w:, 1] = y_vis  # Green channel for Y
    
    # Panel 3: Invalid regions (bottom-left)
    invalid = ((inv_x < 0) | (inv_y < 0)).astype(np.uint8) * 255
    diag_img[proj_h:, :proj_w] = np.stack([invalid, invalid, invalid], axis=-1)
    
    # Panel 4: Gradient magnitude showing discontinuities (bottom-right)
    dx = np.gradient(inv_x, axis=1)
    dy = np.gradient(inv_y, axis=0)
    grad_mag = np.sqrt(dx**2 + dy**2)
    grad_norm = np.clip(grad_mag / np.percentile(grad_mag[grad_mag > 0], 95), 0, 1)
    grad_vis = (grad_norm * 255).astype(np.uint8)
    diag_img[proj_h:, proj_w:, 2] = grad_vis  # Blue channel for gradients
    
    # Add text labels
    cv2.putText(diag_img, "X Coords", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(diag_img, "Y Coords", (proj_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(diag_img, "Invalid", (10, proj_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(diag_img, "Discontinuities", (proj_w + 10, proj_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    if save_path:
        cv2.imwrite(save_path, diag_img)
        print(f"LUT diagnostic image saved to {save_path}")
    
    return diag_img

# ------------------------
# Structured-Light (Phase-Shift Sinusoidal) for Subpixel
# ------------------------

def generate_phase_shift_patterns(
    width: int,
    height: int,
    num_phases: int = 3,
    cycles_x: int = 1,
    cycles_y: int = 1,
    gamma: float = 1.0,
) -> list[dict]:
    """Generate three-phase (or N-phase) sinusoidal patterns for X and Y axes.
    Each entry: {"type":"phase", "axis":"x"|"y", "phase_idx":k, "image": np.ndarray BGR}
    cycles_* specifies how many sinusoidal periods across that dimension (default 1).
    """
    assert num_phases >= 3, "num_phases must be >= 3"
    patterns: list[dict] = []

    def _mk(axis: str, cycles: int):
        if axis == "x":
            xs = (np.arange(width, dtype=np.float32)[None, :] / float(width))
            base = 2.0 * np.pi * cycles * xs  # shape (1, W)
            base = np.repeat(base, height, axis=0)  # (H, W)
        else:
            ys = (np.arange(height, dtype=np.float32)[:, None] / float(height))
            base = 2.0 * np.pi * cycles * ys  # shape (H, 1)
            base = np.repeat(base, width, axis=1)  # (H, W)

        for k in range(num_phases):
            phase_shift = 2.0 * np.pi * (k / float(num_phases))
            # Raw sinusoid in [0,1]
            s = 0.5 + 0.5 * np.cos(base + phase_shift)
            if gamma and abs(gamma - 1.0) > 1e-6:
                s = np.clip(s, 0.0, 1.0) ** (1.0 / float(gamma))
            img = (np.clip(s * 255.0, 0.0, 255.0)).astype(np.uint8)
            img3 = np.repeat(img[:, :, None], 3, axis=2)
            patterns.append({"type": "phase", "axis": axis, "phase_idx": k, "image": img3})

    _mk("x", cycles_x)
    _mk("y", cycles_y)
    return patterns

def decode_phase_shift_from_files(
    capture_files: list[str],
    pattern_meta: list[dict],
    cam_h: int,
    cam_w: int,
    proj_w: int,
    proj_h: int,
    num_phases: int = 3,
    amp_thresh: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode N-phase shifted sinusoidal captures to continuous projector coords.
    Returns (proj_x_of_cam [H,W], proj_y_of_cam [H,W], amp_x [H,W], amp_y [H,W])
    Values are float32; invalid where amplitude below threshold will be set to -1.
    Assumes cycles_x=cycles_y=1 in generation, i.e., phase spans one full 2π over width/height.
    """
    # Load all captures grayscale
    caps: list[np.ndarray] = []
    for fp in capture_files:
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((cam_h, cam_w), dtype=np.uint8)
        caps.append(img.astype(np.float32))

    # Split by axis and sort by phase_idx
    xs = [(cap, meta) for cap, meta in zip(caps, pattern_meta) if meta.get("type") == "phase" and meta.get("axis") == "x"]
    ys = [(cap, meta) for cap, meta in zip(caps, pattern_meta) if meta.get("type") == "phase" and meta.get("axis") == "y"]

    def _phase_decode(seq: list[tuple[np.ndarray, dict]], length: int) -> tuple[np.ndarray, np.ndarray]:
        if not seq:
            return np.full((cam_h, cam_w), -1.0, np.float32), np.zeros((cam_h, cam_w), np.float32)
        try:
            seq_sorted = sorted(seq, key=lambda t: int(t[1].get("phase_idx", 0)))
        except Exception:
            seq_sorted = seq
        # Use first N frames (N-step) for better accuracy
        frames = [seq_sorted[i][0] for i in range(min(len(seq_sorted), num_phases))]
        if len(frames) < 3:
            # Not enough frames; cannot decode
            return np.full((cam_h, cam_w), -1.0, np.float32), np.zeros((cam_h, cam_w), np.float32)
        
        # Apply median filter to reduce noise in captured images
        frames = [cv2.medianBlur(f.astype(np.uint8), 3).astype(np.float32) for f in frames]
        
        if num_phases == 3:
            I1, I2, I3 = frames[0], frames[1], frames[2]
            # Three-step phase-shift decoding with improved formula
            # phi in [-pi, pi]
            num = np.sqrt(3.0) * (I1 - I3)
            den = (2.0 * I2 - I1 - I3) + 1e-6  # Add small epsilon to avoid division by zero
            phi = np.arctan2(num, den)
            # Better amplitude calculation
            amp = np.sqrt(num**2 + den**2) / 2.0
        elif num_phases == 4:
            # Four-step phase-shift for better accuracy
            I1, I2, I3, I4 = frames[0], frames[1], frames[2], frames[3]
            num = I4 - I2
            den = I1 - I3 + 1e-6
            phi = np.arctan2(num, den)
            # Better amplitude for 4-step
            amp = 0.5 * np.sqrt(num**2 + den**2)
        else:
            # General N-step formula
            I1, I2, I3 = frames[0], frames[1], frames[2]
            num = np.sqrt(3.0) * (I1 - I3)
            den = (2.0 * I2 - I1 - I3) + 1e-6
            phi = np.arctan2(num, den)
            amp = np.sqrt(num**2 + den**2) / 2.0
        
        # Apply Gaussian smoothing to phase for better continuity
        phi_smooth = cv2.GaussianBlur(phi, (5, 5), 1.0)
        
        # Map to [0, 2pi)
        phi_mod = (phi_smooth + 2.0 * np.pi) % (2.0 * np.pi)
        # Convert to absolute projector coordinate (cycles=1)
        coords = (length * (phi_mod / (2.0 * np.pi))).astype(np.float32)
        # Invalidate low-amplitude pixels
        coords[amp < float(amp_thresh)] = -1.0
        return coords.astype(np.float32), amp.astype(np.float32)

    proj_x_phase, amp_x = _phase_decode(xs, proj_w)
    proj_y_phase, amp_y = _phase_decode(ys, proj_h)
    return proj_x_phase, proj_y_phase, amp_x, amp_y





def decompose_homography(H: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Decompose 3x3 homography into translation (tx, ty), scale (sx, sy), rotation (deg).
    Returns (tx, ty, sx, sy, angle_deg).
    """
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError("Homography must be 3x3.")

    if abs(H[2, 2]) < 1e-12:
        print("Homography H[2,2] ~ 0; normalizing skipped.")
    else:
        H = H / H[2, 2]

    tx = float(H[0, 2])
    ty = float(H[1, 2])

    A = H[:2, :2]

    sx = float(np.linalg.norm(A[:, 0]))
    sy = float(np.linalg.norm(A[:, 1])) if np.linalg.norm(A[:, 1]) > 1e-12 else 1.0


    R = np.zeros_like(A)
    if sx > 1e-12:
        R[:, 0] = A[:, 0] / sx
    if sy > 1e-12:
        R[:, 1] = A[:, 1] / sy



    angle = math.degrees(math.atan2(R[1, 0], R[0, 0]))

    return tx, ty, sx, sy, angle


def find_homography(
    registration_path: Path = REF_REG_IMG,
    capture_path: Path = CALIB_CAPTURE_IMG,
    save_outputs: bool = True,
) -> np.ndarray:
    """
    Compute homography mapping 'capture' onto 'registration'.
    Saves transformed preview and homography .npy in Assets/Generated.
    Returns H (3x3, float64). Identity if failed.
    """
    reg_p = Path(registration_path)
    cap_p = Path(capture_path)

    if not reg_p.exists():
        print(f"Registration image not found: {reg_p}")
        return np.eye(3, dtype=np.float64)
    if not cap_p.exists():
        print(f"Calibration capture image not found: {cap_p}")
        return np.eye(3, dtype=np.float64)

    img_ref = cv2.imread(reg_p.as_posix(), cv2.IMREAD_COLOR)
    img_cap = cv2.imread(cap_p.as_posix(), cv2.IMREAD_COLOR)
    if img_ref is None or img_cap is None:
        print("Failed to load one or both images for homography.")
        return np.eye(3, dtype=np.float64)

    g_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    g_cap = cv2.cvtColor(img_cap, cv2.COLOR_BGR2GRAY)

    # Use capture as-is; horizontal flip is handled at projection time.
    img_cap_for_match = img_cap
    g_cap_for_match = g_cap

    # Ensure deterministic behavior for RANSAC/USAC inside OpenCV
    try:
        cv2.setRNGSeed(123456)
    except Exception:
        pass


    print("🔍 Preprocessing images for better feature detection...")
    

    g_cap_enhanced = cv2.equalizeHist(g_cap_for_match)
    g_ref_enhanced = cv2.equalizeHist(g_ref)
    

    sift = getattr(cv2, "SIFT_create", None)
    detector = None
    norm = None
    
    if callable(sift):
        try:

            detector = sift(nfeatures=5000, contrastThreshold=0.03, edgeThreshold=20)
            norm = cv2.NORM_L2
            print("🎯 Using enhanced SIFT detector")
        except Exception as e:
            print(f"⚠️ Enhanced SIFT failed: {e}")
    

    if detector is None:
        print("🔄 Using enhanced ORB detector")
        detector = cv2.ORB_create(nfeatures=8000, scaleFactor=1.1, nlevels=12)
        norm = cv2.NORM_HAMMING


    # First try Charuco/ArUco markers (if available) for a very stable homography
    def _homography_from_aruco(img_cap_bgr, img_ref_bgr):
        if cv2_aruco is None:
            return None
        try:
            dict4 = cv2_aruco.getPredefinedDictionary(cv2_aruco.DICT_4X4_100)
            params = cv2_aruco.DetectorParameters()
            detector = cv2_aruco.ArucoDetector(dict4, params)

            def _detect(img_bgr):
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = detector.detectMarkers(gray)
                if ids is None or len(ids) == 0:
                    return {}
                pts = {}
                for i, mid in enumerate(ids.flatten().tolist()):
                    c = corners[i].reshape(-1, 2)
                    # Use all four ordered corners for robustness
                    for k in range(4):
                        pts[(mid, k)] = c[k]
                return pts

            # Prefer Charuco corner interpolation if markers form a Charuco board
            def _detect_charuco(img_bgr):
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = detector.detectMarkers(gray)
                if ids is None or len(ids) == 0:
                    return None
                # Build a CharucoBoard that matches generator defaults
                h, w = gray.shape
                squares_x = 12
                squares_y = 8
                square_size = max(20, min(w, h) // 10)
                marker_size = int(square_size * 0.6)
                board = cv2_aruco.CharucoBoard((squares_x, squares_y), square_size, marker_size, dict4)
                # Interpolate Charuco corners
                ok, ch_corners, ch_ids, _ = cv2_aruco.interpolateCornersCharuco(corners, ids, gray, board)
                if ch_ids is None or ch_corners is None or len(ch_ids) < 8:
                    return None
                return ch_corners.reshape(-1, 2), ch_ids.flatten()

            cap_char = _detect_charuco(img_cap_bgr)
            ref_char = _detect_charuco(img_ref_bgr)
            if cap_char is not None and ref_char is not None:
                cap_pts_all, cap_ids = cap_char
                ref_pts_all, ref_ids = ref_char
                # Match by Charuco IDs
                id_to_idx_cap = {int(i): idx for idx, i in enumerate(cap_ids.tolist())}
                pts_cap = []
                pts_ref = []
                for idx_ref, iid in enumerate(ref_ids.tolist()):
                    if iid in id_to_idx_cap:
                        pts_cap.append(cap_pts_all[id_to_idx_cap[iid]])
                        pts_ref.append(ref_pts_all[idx_ref])
                if len(pts_cap) >= 8:
                    pts_cap = np.float32(pts_cap).reshape(-1, 2)
                    pts_ref = np.float32(pts_ref).reshape(-1, 2)
                    Hc, inl = cv2.findHomography(pts_cap, pts_ref, cv2.RANSAC, ransacReprojThreshold=1.5, confidence=0.999)
                    return Hc

            # Fallback to raw ArUco corners mapping
            cap_pts = _detect(img_cap_bgr)
            ref_pts = _detect(img_ref_bgr)
            keys = sorted(set(cap_pts.keys()) & set(ref_pts.keys()))
            if len(keys) < 4:
                return None
            pts_cap = np.float32([cap_pts[k] for k in keys]).reshape(-1, 2)
            pts_ref = np.float32([ref_pts[k] for k in keys]).reshape(-1, 2)
            Hc, inl = cv2.findHomography(pts_cap, pts_ref, cv2.RANSAC, ransacReprojThreshold=2.0, confidence=0.999)
            return Hc
        except Exception:
            return None

    H_aruco = _homography_from_aruco(img_cap_for_match, img_ref)
    if isinstance(H_aruco, np.ndarray) and H_aruco.shape == (3, 3):
        H = H_aruco.astype(np.float64)
        inlier_count = 16  # Minimum from 4 markers * 4 corners
        print("✅ ArUco-based Homography successful")
    else:
        # Fall back to SIFT/ORB features
        kp1, d1 = detector.detectAndCompute(g_cap_enhanced, None)
        kp2, d2 = detector.detectAndCompute(g_ref_enhanced, None)

    print(f"🔍 Enhanced keypoints: capture={len(kp1 or [])}, reference={len(kp2 or [])}")

    if H_aruco is None and (d1 is None or d2 is None or len(kp1) < 8 or len(kp2) < 8):
        print("❌ Insufficient features detected. Try different lighting or pattern.")
        print(f"   Capture keypoints: {len(kp1 or [])}")
        print(f"   Reference keypoints: {len(kp2 or [])}")
        return np.eye(3, dtype=np.float64)


    matches = []
    

    if H_aruco is None:
        try:
            # Prefer KNN + ratio test for stability
            print("📍 FLANN/KNN matching...")
            if norm == cv2.NORM_L2:
                index_params = dict(algorithm=1, trees=5)  # KDTree
                search_params = dict(checks=64)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
            else:
                flann = cv2.BFMatcher(norm)
            knn = flann.knnMatch(d1, d2, k=2)
            for m_n in knn:
                if len(m_n) != 2:
                    continue
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    matches.append(m)
            print(f"📍 KNN+ratio matches: {len(matches)}")
        except Exception as e:
            print(f"⚠️ KNN matching failed: {e}")


    if len(matches) < 20:  # Need more matches for robust calibration
        try:
            print("🔄 Applying KNN+ratio test for more matches...")
            bf = cv2.BFMatcher(norm, crossCheck=False)
            knn = bf.knnMatch(d1, d2, k=2)
            knn_matches = []
            for pair in knn:
                if len(pair) < 2:
                    continue
                m, n = pair

                if m.distance < 0.65 * n.distance:
                    knn_matches.append(m)
            

            existing_pairs = {(m.queryIdx, m.trainIdx) for m in matches}
            for m in knn_matches:
                if (m.queryIdx, m.trainIdx) not in existing_pairs:
                    matches.append(m)
            
            matches = sorted(matches, key=lambda m: m.distance)
            print(f"📍 Combined matches: {len(matches)}")
            
        except Exception as e:
            print(f"❌ KNN matching failed: {e}")
            if not matches:
                return np.eye(3, dtype=np.float64)


    if H_aruco is None and matches:

        distances = [m.distance for m in matches]
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        threshold = mean_dist + 1.5 * std_dist  # Remove matches beyond 1.5 std devs
        
        good_matches = [m for m in matches if m.distance <= threshold]
        

        if len(good_matches) >= 12:
            matches = good_matches
            print(f"📊 Quality filtered matches: {len(matches)} (removed outliers)")
        else:

            keep = max(12, int(len(matches) * 0.85))
            matches = matches[:keep]
            print(f"📊 Top matches: {len(matches)} (kept best 85%)")

    if H_aruco is None and len(matches) < 8:
        print(f"❌ Insufficient quality matches: {len(matches)}/8 minimum")
        print("   💡 Try improving lighting, focus, or pattern visibility")
        return np.eye(3, dtype=np.float64)



    if H_aruco is None:
        pts_cap = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        pts_ref = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)



    H = None
    inlier_count = 0
    

    if H_aruco is None:
        try:
            H, inliers = cv2.findHomography(pts_cap, pts_ref, cv2.RANSAC, ransacReprojThreshold=2.0, confidence=0.999)
            if H is not None:
                inlier_count = int(inliers.sum()) if inliers is not None else len(matches)
                print(f"✅ Homography successful: {inlier_count}/{len(matches)} inliers")
        except Exception as e:
            print(f"⚠️ Homography failed: {e}")
    

    # Skip LMEDS fallback to avoid unstable jumps
    

    if H is None:
        try:
            print("🔄 Trying least squares method...")
            H, _ = cv2.findHomography(pts_cap, pts_ref, 0)  # Regular method
            if H is not None:
                inlier_count = len(matches)
                print(f"✅ Least squares Homography successful")
        except Exception as e:
            print(f"❌ All homography methods failed: {e}")
    
    if H is None:
        print("❌ Homography estimation failed completely. Returning identity.")
        return np.eye(3, dtype=np.float64)

    print(f"📊 Final homography inliers: {inlier_count}/{len(matches)} ({100*inlier_count/len(matches):.1f}%)")

    # Optional: intensity-based refinement using ECC to stabilize result further
    try:
        h, w = g_ref.shape
        # ECC expects float32 images normalized [0,1]
        ref32 = (g_ref.astype(np.float32) / 255.0)
        cap_warp_init = cv2.warpPerspective(g_cap_for_match, H, (w, h))
        cap32 = (cap_warp_init.astype(np.float32) / 255.0)

        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
        warp_init = H.astype(np.float32)
        cc, warp_refined = cv2.findTransformECC(ref32, cap32, warp_init, cv2.MOTION_HOMOGRAPHY, criteria)
        if isinstance(warp_refined, np.ndarray) and warp_refined.shape == (3, 3):
            H = warp_refined.astype(np.float64)
            print(f"🔧 ECC refinement applied (cc={cc:.6f})")
    except Exception as e:
        print(f"ℹ️ ECC refinement skipped: {e}")


    try:
        tx, ty, sx, sy, ang = decompose_homography(H)
        print(f"📐 Decomposed H → tx={tx:.2f}, ty={ty:.2f}, sx={sx:.3f}, sy={sy:.3f}, angle={ang:.2f}°")
        

        validation_failed = False
        

        inlier_percent = 100 * inlier_count / len(matches)
        if inlier_percent < 40:
            print(f"❌ Poor inlier ratio: {inlier_percent:.1f}% (need >40%)")
            validation_failed = True
        

        if abs(sx - 1.0) > 0.7 or abs(sy - 1.0) > 0.7:
            print(f"❌ Extreme scale change: sx={sx:.3f}, sy={sy:.3f} (max deviation: ±0.7)")
            validation_failed = True
        elif abs(sx - 1.0) > 0.3 or abs(sy - 1.0) > 0.3:
            print(f"⚠️ Warning: Large scale change detected (sx={sx:.3f}, sy={sy:.3f})")
        


        normalized_ang = ang
        if abs(ang) > 90:

            if ang > 90:
                normalized_ang = ang - 180
            elif ang < -90:
                normalized_ang = ang + 180
            print(f"📐 Normalized rotation from {ang:.1f}° to {normalized_ang:.1f}° (pattern orientation)")
        
        if abs(normalized_ang) > 60:
            print(f"❌ Extreme rotation: {normalized_ang:.1f}° (max: ±60°)")
            print("   💡 Try ensuring the calibration pattern is right-side up in both camera and projector")
            validation_failed = True
        elif abs(normalized_ang) > 30:
            print(f"⚠️ Warning: Large rotation detected ({normalized_ang:.1f}°)")
        

        img_diagonal = np.sqrt(g_ref.shape[0]**2 + g_ref.shape[1]**2)
        max_translation = img_diagonal * 0.8  # 80% of diagonal
        if abs(tx) > max_translation or abs(ty) > max_translation:
            print(f"❌ Extreme translation: tx={tx:.1f}, ty={ty:.1f} (max: ±{max_translation:.1f})")
            validation_failed = True
        elif abs(tx) > max_translation * 0.5 or abs(ty) > max_translation * 0.5:
            print(f"⚠️ Warning: Large translation detected (tx={tx:.1f}, ty={ty:.1f})")
        

        det = np.linalg.det(H[:2, :2])
        if abs(det) < 0.01:
            print(f"❌ Degenerate homography: determinant={det:.6f}")
            validation_failed = True
        
        if validation_failed:
            print("❌ Homography failed validation - proceeding with raw H for confirmation projection")
            print("   📊 Calibration diagnostics:")
            print(f"      - Inlier ratio: {inlier_percent:.1f}% (need >40%)")
            print(f"      - Scale factors: sx={sx:.3f}, sy={sy:.3f} (need ±0.7 from 1.0)")
            print(f"      - Rotation: {normalized_ang:.1f}° (need ±60°)")
            print(f"      - Translation: tx={tx:.1f}, ty={ty:.1f} (max ±{max_translation:.1f})")
            print("   💡 Specific suggestions based on your setup:")
            
            if inlier_percent < 20:
                print("      🔍 Very low feature matching - check lighting and focus")
            if abs(sx - 1.0) > 0.5 or abs(sy - 1.0) > 0.5:
                print("      📏 Major scale distortion - check camera distance and projector size")
            if abs(normalized_ang) > 45:
                print("      🔄 Large rotation - align calibration pattern orientation")
            if abs(tx) > max_translation * 0.6 or abs(ty) > max_translation * 0.6:
                print("      📍 Large offset - center the pattern in both camera and projector view")
                
            print("   🛠️ General troubleshooting:")
            print("      - Ensure calibration pattern is fully visible in camera")
            print("      - Improve lighting conditions (avoid glare and shadows)")
            print("      - Check camera focus")
            print("      - Verify projector is displaying pattern correctly")
            print("      - Try moving camera closer or adjusting projector size")
            # Note: we intentionally continue and return the raw H so the UI can project
            # a confirmation image. Visual confirmation helps verify calibration even if
            # automated validation flags issues.
        else:
            print("✅ Homography passed validation checks")
            
    except Exception as e:
        print(f"⚠️ Could not validate homography: {e}")


    if save_outputs:
        h, w = g_ref.shape
        warped = cv2.warpPerspective(img_cap_for_match, H, (w, h))
        try:
            cv2.imwrite(CALIB_OUTPUT_IMG.as_posix(), warped)
            np.save(HOMOGRAPHY_NPY.as_posix(), H.astype(np.float64))
            print(f"💾 Saved warped preview: {CALIB_OUTPUT_IMG}")
            print(f"💾 Saved homography: {HOMOGRAPHY_NPY}")
            

            _generate_alignment_verification(img_ref, warped, H)
            
        except Exception as e:
            print(f"❌ Output save failed: {e}")

    print(f"✅ Calibration completed successfully!")
    return H.astype(np.float64)


def _generate_alignment_verification(reference, warped, homography):
   
    try:

        h, w = reference.shape[:2]
        comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
        

        if len(reference.shape) == 3:
            comparison[:, :w] = reference
        else:
            comparison[:, :w] = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)
        

        if len(warped.shape) == 3:
            comparison[:, w:] = warped
        else:
            comparison[:, w:] = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        

        cv2.line(comparison, (w, 0), (w, h), (0, 255, 0), 2)
        

        cv2.putText(comparison, "REFERENCE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "ALIGNED CAPTURE", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        

        verification_path = CALIB_OUTPUT_IMG.parent / "calibration_verification.png"
        cv2.imwrite(str(verification_path), comparison)
        print(f"📸 Alignment verification saved: {verification_path}")
        

        if len(reference.shape) == 3:
            ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = reference
            
        if len(warped.shape) == 3:
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        else:
            warped_gray = warped
        

        mse = np.mean((ref_gray.astype(float) - warped_gray.astype(float)) ** 2)
        print(f"📊 Alignment quality MSE: {mse:.2f} (lower is better)")
        
        if mse < 1000:
            print(f"✅ Excellent alignment quality!")
        elif mse < 3000:
            print(f"✅ Good alignment quality")
        elif mse < 8000:
            print(f"⚠️ Fair alignment quality - consider recalibrating")
        else:
            print(f"❌ Poor alignment quality - recalibration recommended")
            
    except Exception as e:
        print(f"⚠️ Verification image generation failed: {e}")





def draw_number(draw: ImageDraw.ImageDraw, position: Tuple[int, int], number: int, size: int, color):
   
    x, y = position
    lw = max(1, size // 10)
    if number == 1:
        draw.line([(x + size // 2, y), (x + size // 2, y + size)], fill=color, width=lw)
    elif number == 2:
        draw.line([(x, y), (x + size, y)], fill=color, width=lw)
        draw.line([(x + size, y), (x + size, y + size // 2)], fill=color, width=lw)
        draw.line([(x, y + size // 2), (x + size, y + size // 2)], fill=color, width=lw)
        draw.line([(x, y + size // 2), (x, y + size)], fill=color, width=lw)
        draw.line([(x, y + size), (x + size, y + size)], fill=color, width=lw)
    elif number == 3:
        draw.line([(x, y), (x + size, y)], fill=color, width=lw)
        draw.line([(x, y + size // 2), (x + size, y + size // 2)], fill=color, width=lw)
        draw.line([(x, y + size), (x + size, y + size)], fill=color, width=lw)
    elif number == 4:
        draw.line([(x + size, y), (x + size, y + size)], fill=color, width=lw)
        draw.line([(x, y + size // 2), (x + size, y + size // 2)], fill=color, width=lw)
        draw.line([(x, y), (x, y + size // 2)], fill=color, width=lw)
    elif number == 5:
        draw.line([(x, y), (x + size, y)], fill=color, width=lw)
        draw.line([(x, y), (x, y + size // 2)], fill=color, width=lw)
        draw.line([(x, y + size // 2), (x + size, y + size // 2)], fill=color, width=lw)
        draw.line([(x, y + size), (x + size, y + size)], fill=color, width=lw)
    elif number == 6:
        draw.line([(x, y + size // 2), (x + size, y + size // 2)], fill=color, width=lw)
        draw.line([(x, y), (x, y + size)], fill=color, width=lw)
        draw.line([(x, y + size), (x + size, y + size)], fill=color, width=lw)
        draw.line([(x, y + size // 2), (x + size, y + size // 2)], fill=color, width=lw)


def draw_smiley_face(draw: ImageDraw.ImageDraw, center: Tuple[int, int], radius: int, color):
   
    x, y = center
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=color, width=max(2, radius // 20))
    eye_r = max(2, radius // 6)
    left_eye = (x - radius // 3, y - radius // 3)
    right_eye = (x + radius // 3, y - radius // 3)
    draw.ellipse([left_eye[0] - eye_r, left_eye[1] - eye_r, left_eye[0] + eye_r, left_eye[1] + eye_r], fill=color)
    draw.ellipse([right_eye[0] - eye_r, right_eye[1] - eye_r, right_eye[0] + eye_r, right_eye[1] + eye_r], fill=color)

    mouth_h = max(2, radius // 15)
    draw.arc([x - radius // 2, y + radius // 4 - mouth_h, x + radius // 2, y + radius // 4 + mouth_h],
             start=0, end=180, fill=color, width=max(2, radius // 25))
