

import time
import gc
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


try:
    import psutil  
except Exception:
    psutil = None


_THIS = Path(__file__).resolve()

_candidates = [
    _THIS.parents[1] / "Assets" / "Generated",
    _THIS.parent / "Assets" / "Generated",
    Path.cwd() / "Assets" / "Generated",
]
for _cand in _candidates:
    if (_cand.parent).exists():
        ASSET_DIR = _cand
        break
else:
    ASSET_DIR = _THIS.parents[1] / "Assets" / "Generated"
ASSET_DIR.mkdir(parents=True, exist_ok=True)



def _safe_filename(pattern: str, size: Tuple[int, int], color: Tuple[int, int, int], fmt: str) -> Path:
    w, h = size
    r, g, b = color
    name = f"{pattern}_{w}x{h}_{r}-{g}-{b}.{fmt.lower()}"
    return ASSET_DIR / name

def _clamp_color(c: Tuple[int, int, int]) -> Tuple[int, int, int]:
    r, g, b = c
    return (max(0, min(255, int(r))),
            max(0, min(255, int(g))),
            max(0, min(255, int(b))))



class WhiteBackgroundGenerator:

    def __init__(self):
        self._cache: Dict[str, Path] = {}
        self._start_ts = time.time()
        self._images_generated = 0
        self._images_failed = 0
        self._peak_rss_mb = 0.0
        print("🚀 WhiteBackgroundGenerator ready")


    def make_white(
        self,
        width: int,
        height: int,
        pattern: str = "solid",
        color: Tuple[int, int, int] = (255, 255, 255),
        save_format: str = "png",
        optimize: bool = True,
    ) -> bool:
        try:
            w = int(width)
            h = int(height)
            if w <= 0 or h <= 0:
                print("make_white: width/height must be positive")
                return False

            color = _clamp_color(color)
            key = f"{pattern}:{w}x{h}:{color[0]}-{color[1]}-{color[2]}:{save_format.lower()}"
            cached = self._cache.get(key)
            if cached and cached.exists():
                print(f"✅ Using cached background: {cached}")
                return True

            out_path = _safe_filename(pattern, (w, h), color, save_format)
            ok = self._generate(pattern, (w, h), color, out_path, optimize)
            if ok:
                self._cache[key] = out_path
                self._images_generated += 1

                if pattern == "solid" and color == (255, 255, 255):
                    (ASSET_DIR / "solid_white_image.png").write_bytes(out_path.read_bytes())
                print(f"✅ {pattern.capitalize()} background generated: {w}x{h} → {out_path}")
            else:
                self._images_failed += 1
            self._update_peak_mem()
            return ok
        except Exception as e:
            self._images_failed += 1
            print(f"make_white failed: {e}")
            return False


    def _generate(self, pattern: str, size: Tuple[int, int], color: Tuple[int, int, int], out_path: Path, optimize: bool) -> bool:
        try:
            if pattern == "solid":
                img = Image.new("RGB", size, color)
            elif pattern == "gradient":
                img = self._gradient(size, color)
            elif pattern == "checkerboard":
                img = self._checker(size, color)
            elif pattern == "noise":
                img = self._noise(size, color)
            else:
                print(f"Unknown pattern '{pattern}', falling back to solid")
                img = Image.new("RGB", size, color)

            save_kwargs = {}
            fmt = out_path.suffix.lower().lstrip(".")
            if fmt == "png" and optimize:
                save_kwargs["optimize"] = True
            elif fmt in ("jpg", "jpeg"):
                save_kwargs["quality"] = 95
                save_kwargs["optimize"] = True

            img.save(out_path, **save_kwargs)
            return True
        except Exception as e:
            print(f"_generate('{pattern}') failed: {e}")
            return False
        finally:
            try:
                del img
            except Exception:
                pass
            gc.collect()

    def _gradient(self, size: Tuple[int, int], color: Tuple[int, int, int]) -> Image.Image:
        w, h = size
        r, g, b = color
        img = Image.new("RGB", (w, h))
        draw = ImageDraw.Draw(img)
        for y in range(h):
            t = y / max(1, h - 1)
            draw.line([(0, y), (w, y)], fill=(int(r * t), int(g * t), int(b * t)))
        return img

    def _checker(self, size: Tuple[int, int], color: Tuple[int, int, int]) -> Image.Image:
        w, h = size
        cell = max(4, min(w, h) // 20)
        img = Image.new("RGB", (w, h))
        draw = ImageDraw.Draw(img)
        for y in range(0, h, cell):
            for x in range(0, w, cell):
                c = color if ((x // cell + y // cell) % 2 == 0) else (0, 0, 0)
                draw.rectangle([x, y, x + cell, y + cell], fill=c)
        return img

    def _noise(self, size: Tuple[int, int], color: Tuple[int, int, int]) -> Image.Image:
        w, h = size
        noise = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        r, g, b = color

        noise[:, :, 0] = (0.3 * noise[:, :, 0] + 0.7 * r).astype(np.uint8)
        noise[:, :, 1] = (0.3 * noise[:, :, 1] + 0.7 * g).astype(np.uint8)
        noise[:, :, 2] = (0.3 * noise[:, :, 2] + 0.7 * b).astype(np.uint8)
        img = Image.fromarray(noise)
        return img.filter(ImageFilter.GaussianBlur(radius=0.5))

    def _update_peak_mem(self):
        if psutil is None:
            return
        try:
            rss = psutil.Process().memory_info().rss / (1024 * 1024)
            self._peak_rss_mb = max(self._peak_rss_mb, rss)
        except Exception:
            pass


    def get_stats(self) -> Dict[str, Any]:
        uptime = time.time() - self._start_ts
        return {
            "images_generated": self._images_generated,
            "images_failed": self._images_failed,
            "peak_rss_mb": round(self._peak_rss_mb, 1),
            "uptime_s": round(uptime, 1),
            "cache_size": len(self._cache),
            "asset_dir": str(ASSET_DIR),
        }


_gen = WhiteBackgroundGenerator()


def makeWhite(width: int, height: int) -> bool:
    return _gen.make_white(width, height, pattern="solid", color=(255, 255, 255))

def makeGradientWhite(width: int, height: int) -> bool:
    return _gen.make_white(width, height, pattern="gradient", color=(255, 255, 255))

def makeCheckerboardWhite(width: int, height: int) -> bool:
    return _gen.make_white(width, height, pattern="checkerboard", color=(255, 255, 255))

def makeNoiseWhite(width: int, height: int) -> bool:
    return _gen.make_white(width, height, pattern="noise", color=(255, 255, 255))
