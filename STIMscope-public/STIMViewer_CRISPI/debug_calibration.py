#!/usr/bin/env python3
"""
Debug script to analyze captured calibration patterns
"""

import numpy as np
from pathlib import Path
from PIL import Image

def analyze_captures():
    """Analyze the captured structured light patterns"""
    
    save_dir = Path('/media/aharonilabjetson2/NVMe/projects/STIMViewerV2/STIMscope-public/STIMViewer_CRISPI/Saved_Media')
    
    # Find all captured patterns
    captures = sorted(save_dir.glob('sl_cap_*.png'))
    print(f"Found {len(captures)} captured patterns")
    
    if not captures:
        print("No captures found!")
        return
    
    # Analyze first few and last few captures
    samples = captures[:5] + captures[-5:] if len(captures) > 10 else captures
    
    for cap_path in samples:
        img_pil = Image.open(cap_path).convert('L')
        img = np.array(img_pil)
        if img is not None:
            # Calculate statistics
            mean_val = img.mean()
            std_val = img.std()
            min_val = img.min()
            max_val = img.max()
            
            # Check if image is useful (has variation)
            is_uniform = std_val < 5  # Very low variation
            is_saturated = max_val == 255 and (img == 255).sum() > img.size * 0.1
            is_dark = mean_val < 20
            
            status = "OK"
            if is_uniform:
                status = "UNIFORM (no pattern)"
            elif is_saturated:
                status = "SATURATED"
            elif is_dark:
                status = "TOO DARK"
            
            print(f"{cap_path.name}: mean={mean_val:.1f}, std={std_val:.1f}, "
                  f"range=[{min_val}, {max_val}] - {status}")
            
            # Save a sample crop for visual inspection
            if cap_path.name in ['sl_cap_000.png', 'sl_cap_010.png', 'sl_cap_020.png']:
                h, w = img.shape
                crop = img[h//2-50:h//2+50, w//2-50:w//2+50]
                Image.fromarray(crop).save(str(save_dir / f'debug_{cap_path.stem}_crop.png'))
    
    # Check Gray code decoding
    print("\n=== Checking LUT values ===")
    asset_dir = Path('/media/aharonilabjetson2/NVMe/projects/STIMViewerV2/STIMscope-public/STIMViewer_CRISPI/Assets/Generated')
    
    try:
        proj_x = np.load(asset_dir / 'proj_from_cam_x.npy')
        proj_y = np.load(asset_dir / 'proj_from_cam_y.npy')
        
        valid_x = proj_x[proj_x >= 0]
        valid_y = proj_y[proj_y >= 0]
        
        print(f"Forward LUT shape: {proj_x.shape}")
        print(f"Valid pixels: {len(valid_x)} / {proj_x.size} ({100*len(valid_x)/proj_x.size:.1f}%)")
        
        if len(valid_x) > 0:
            print(f"X range: [{valid_x.min():.1f}, {valid_x.max():.1f}]")
            print(f"Y range: [{valid_y.min():.1f}, {valid_y.max():.1f}]")
            
            # Check if values make sense
            if valid_x.max() < 100:
                print("WARNING: X values suspiciously small - might be normalized or incorrect")
            if valid_y.max() < 100:
                print("WARNING: Y values suspiciously small - might be normalized or incorrect")
        else:
            print("ERROR: No valid pixels in LUT!")
            
    except Exception as e:
        print(f"Error loading LUTs: {e}")

    print("\n=== Recommendations ===")
    print("1. If captures are UNIFORM: Increase delay between projection and capture")
    print("2. If captures are SATURATED: Reduce projector brightness or camera exposure")
    print("3. If captures are TOO DARK: Increase projector brightness or camera exposure")
    print("4. If LUT has no valid pixels: Check pattern generation and Gray code decoding")


if __name__ == "__main__":
    analyze_captures()