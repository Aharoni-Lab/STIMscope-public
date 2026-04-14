# CalibrationPattern.py
# Python 3.8+ compatible (no 3.10 "X | None" syntax)
# Generates a grayscale calibration pattern and PUSHes it over ZMQ.

import argparse
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw
import zmq


def draw_number(draw, position, number, size, color=255):
    """Draw numbers 1..6 using simple strokes (grayscale)."""
    x, y = position
    w = size
    lw = max(1, size // 10)  # line width

    if number == 1:
        draw.line([(x + w // 2, y), (x + w // 2, y + w)], fill=color, width=lw)
    elif number == 2:
        draw.line([(x, y), (x + w, y)], fill=color, width=lw)                  # top
        draw.line([(x + w, y), (x + w, y + w // 2)], fill=color, width=lw)     # right upper
        draw.line([(x, y + w // 2), (x + w, y + w // 2)], fill=color, width=lw)# middle
        draw.line([(x, y + w // 2), (x, y + w)], fill=color, width=lw)         # left lower
        draw.line([(x, y + w), (x + w, y + w)], fill=color, width=lw)          # bottom
    elif number == 3:
        draw.line([(x, y), (x + w, y)], fill=color, width=lw)                  # top
        draw.line([(x, y + w // 2), (x + w, y + w // 2)], fill=color, width=lw)# middle
        draw.line([(x, y + w), (x + w, y + w)], fill=color, width=lw)          # bottom
    elif number == 4:
        draw.line([(x, y + w // 2), (x + w, y + w // 2)], fill=color, width=lw)# middle
        draw.line([(x, y), (x, y + w // 2)], fill=color, width=lw)             # left upper
        draw.line([(x + w, y), (x + w, y + w)], fill=color, width=lw)          # right full
    elif number == 5:
        draw.line([(x, y), (x + w, y)], fill=color, width=lw)                  # top
        draw.line([(x, y), (x, y + w // 2)], fill=color, width=lw)             # left upper
        draw.line([(x, y + w // 2), (x + w, y + w // 2)], fill=color, width=lw)# middle
        draw.line([(x, y + w), (x + w, y + w)], fill=color, width=lw)          # bottom
    elif number == 6:
        draw.line([(x, y), (x, y + w)], fill=color, width=lw)                  # left full
        draw.line([(x, y + w // 2), (x + w, y + w // 2)], fill=color, width=lw)# middle
        draw.line([(x, y + w), (x + w, y + w)], fill=color, width=lw)          # bottom


def draw_smiley_face(draw, center, radius, color=255):
    """Simple smiley in grayscale."""
    x, y = center
    lw = max(1, radius // 8)
    # face outline
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=color, width=lw)
    # eyes
    er = max(1, radius // 8)
    draw.ellipse([x - radius // 3 - er, y - radius // 3 - er,
                  x - radius // 3 + er, y - radius // 3 + er], fill=color)
    draw.ellipse([x + radius // 3 - er, y - radius // 3 - er,
                  x + radius // 3 + er, y - radius // 3 + er], fill=color)
    # mouth (arc)
    mouth_w = radius
    mouth_h = radius // 2
    draw.arc([x - mouth_w // 2, y, x + mouth_w // 2, y + mouth_h], start=0, end=180, fill=color, width=lw)


def create_custom_registration_image(width, height, line_color=255, fill_color=255,
                                     save_png: Optional[str] = None) -> np.ndarray:
    """
    Build a grayscale (uint8) calibration image.
    Returns a 2D numpy array (H, W) suitable to send directly to your ZMQ/GL app.
    """
    # 'L' = grayscale
    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)

    # Properties
    large_font_size = int(min(width, height) * 0.5)  # scales with size
    number_font_size = int(min(width, height) * 0.18)
    chessboard_size = 8
    chess_cell = max(10, min(width, height) // 30)
    circle_radius = min(width, height) // 4
    cross_size = int(min(width, height) * 0.23)
    gradient_bar_width = max(100, width // 10)
    circle_thickness = max(3, min(width, height) // 200)
    cross_thickness = max(8, min(width, height) // 30)
    f_thickness = max(6, min(width, height) // 36)

    # Big block "F" at center (stroke-based)
    x0 = width // 2 - large_font_size // 2
    y0 = height // 2 - large_font_size // 2
    lw = f_thickness
    # Top horizontal
    draw.line([(x0, y0), (x0 + int(large_font_size * 0.8), y0)], fill=line_color, width=lw)
    # Vertical
    draw.line([(x0, y0), (x0, y0 + int(large_font_size * 0.6))], fill=line_color, width=lw)
    # Middle horizontal
    draw.line([(x0, y0 + int(large_font_size * 0.4)),
               (x0 + int(large_font_size * 0.6), y0 + int(large_font_size * 0.4))],
              fill=line_color, width=lw)

    # Numbers 1?6 near the quadrants/center sides
    num_pos = [
        (width // 4 - number_font_size // 2, height // 4 - number_font_size // 2),
        (3 * width // 4 - number_font_size // 2, height // 4 - number_font_size // 2),
        (width // 4 - number_font_size // 2, 3 * height // 4 - number_font_size // 2),
        (3 * width // 4 - number_font_size // 2, 3 * height // 4 - number_font_size // 2),
        (width // 4 - number_font_size // 2, height // 2 - number_font_size // 2),
        (3 * width // 4 - number_font_size // 2, height // 2 - number_font_size // 2),
    ]
    for n, pos in zip(range(1, 7), num_pos):
        draw_number(draw, pos, n, number_font_size, line_color)

    # Left grayscale gradient
    for i in range(gradient_bar_width):
        g = int(i * 255 / max(1, gradient_bar_width - 1))
        draw.line([(i, 0), (i, height)], fill=g, width=1)

    # Concentric circles in top-right
    for i in range(5):
        pad = i * 20
        draw.ellipse([(width - circle_radius - pad, 0 + pad),
                      (width - 0 - pad, circle_radius + pad)],
                     outline=line_color, width=circle_thickness)

    # Chessboard at bottom center
    start_x = (width - chessboard_size * chess_cell) // 2
    start_y = height - chessboard_size * chess_cell
    for i in range(chessboard_size):
        for j in range(chessboard_size):
            tl = (start_x + i * chess_cell, start_y + j * chess_cell)
            br = (start_x + (i + 1) * chess_cell, start_y + (j + 1) * chess_cell)
            fill = fill_color if (i + j) % 2 == 0 else 0
            draw.rectangle([tl, br], fill=fill)

    # Thick cross in top-left
    cx = cy = cross_size
    draw.line([(cx - cross_size, cy), (cx + cross_size, cy)], fill=line_color, width=cross_thickness)
    draw.line([(cx, cy - cross_size), (cx, cy + cross_size)], fill=line_color, width=cross_thickness)

    # Two smileys bottom-right-ish
    draw_smiley_face(draw, (width - 900, height - 700), 50, line_color)
    draw_smiley_face(draw, (width - 1000, height - 950), 100, line_color)

    if save_png:
        img.save(save_png)

    # Return as numpy uint8 (H, W)
    return np.array(img, dtype=np.uint8)


def send_over_zmq(frame_gray: np.ndarray,
                  endpoint: str = "tcp://127.0.0.1:5556") -> None:
    """Send a grayscale uint8 image over ZMQ PUSH."""
    if frame_gray.ndim != 2 or frame_gray.dtype != np.uint8:
        raise ValueError("frame_gray must be 2D uint8")

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUSH)
    sock.connect(endpoint)
    sock.send(frame_gray.tobytes())
    print("? Sent calibration pattern:", frame_gray.shape, frame_gray.dtype)


def main():
    parser = argparse.ArgumentParser(description="Generate and send a calibration pattern over ZMQ.")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--save", type=str, default=None, help="Optional path to save PNG")
    parser.add_argument("--endpoint", type=str, default="tcp://127.0.0.1:5556")
    args = parser.parse_args()

    img = create_custom_registration_image(
        width=args.width,
        height=args.height,
        line_color=255,
        fill_color=255,
        save_png=args.save
    )
    send_over_zmq(img, endpoint=args.endpoint)


if __name__ == "__main__":
    main()
