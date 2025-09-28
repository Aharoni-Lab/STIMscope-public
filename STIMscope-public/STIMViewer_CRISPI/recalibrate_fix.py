#!/usr/bin/env python3
"""
Recalibration script to fix LUT distortions
Run this after structured-light calibration to improve quality
"""

import numpy as np
from pathlib import Path
import sys

def fix_lut_distortions():
    """Post-process LUTs to reduce distortions"""
    
    asset_dir = Path('/media/aharonilabjetson2/NVMe/projects/STIMViewerV2/STIMscope-public/STIMViewer_CRISPI/Assets/Generated')
    
    print("Loading LUTs...")
    try:
        # Load the inverse LUTs
        inv_x = np.load(asset_dir / 'cam_from_proj_x.npy')
        inv_y = np.load(asset_dir / 'cam_from_proj_y.npy')
        
        print(f"Original LUT shape: {inv_x.shape}")
        
        # Apply 90-degree rotation fix
        print("Applying rotation correction...")
        proj_h, proj_w = inv_x.shape
        inv_x_rot = inv_y.T.copy()
        inv_y_rot = (proj_h - 1) - inv_x.T
        inv_x = inv_x_rot
        inv_y = inv_y_rot
        
        # Analyze and fix distortions
        valid = (inv_x >= 0) & (inv_y >= 0)
        print(f"Valid pixels: {valid.sum()} / {valid.size} ({100*valid.sum()/valid.size:.1f}%)")
        
        if valid.sum() > 1000:
            # 1. Fill holes using nearest neighbor
            if (~valid).sum() > 0:
                print(f"Filling {(~valid).sum()} holes...")
                
                # Simple nearest neighbor filling
                from PIL import Image
                
                # Convert to image for processing
                inv_x_img = Image.fromarray(inv_x.astype(np.float32))
                inv_y_img = Image.fromarray(inv_y.astype(np.float32))
                
                # Fill by expanding valid regions
                for iteration in range(5):
                    inv_x_old = inv_x.copy()
                    inv_y_old = inv_y.copy()
                    
                    for y in range(1, inv_x.shape[0] - 1):
                        for x in range(1, inv_x.shape[1] - 1):
                            if inv_x[y, x] < 0:  # Invalid pixel
                                # Check neighbors
                                neighbors_x = []
                                neighbors_y = []
                                for dy in [-1, 0, 1]:
                                    for dx in [-1, 0, 1]:
                                        if dy == 0 and dx == 0:
                                            continue
                                        ny, nx = y + dy, x + dx
                                        if inv_x_old[ny, nx] >= 0:
                                            neighbors_x.append(inv_x_old[ny, nx])
                                            neighbors_y.append(inv_y_old[ny, nx])
                                
                                if neighbors_x:
                                    inv_x[y, x] = np.mean(neighbors_x)
                                    inv_y[y, x] = np.mean(neighbors_y)
            
            # 2. Simple smoothing using convolution
            print("Smoothing LUT...")
            
            # Create a simple Gaussian-like kernel
            kernel = np.array([[1, 2, 1],
                              [2, 4, 2],
                              [1, 2, 1]]) / 16.0
            
            # Manual convolution for smoothing
            inv_x_smooth = inv_x.copy()
            inv_y_smooth = inv_y.copy()
            
            for y in range(1, inv_x.shape[0] - 1):
                for x in range(1, inv_x.shape[1] - 1):
                    if inv_x[y, x] >= 0:  # Only smooth valid pixels
                        # Apply kernel
                        patch_x = inv_x[y-1:y+2, x-1:x+2]
                        patch_y = inv_y[y-1:y+2, x-1:x+2]
                        valid_patch = patch_x >= 0
                        
                        if valid_patch.sum() > 4:  # Need enough valid neighbors
                            inv_x_smooth[y, x] = np.sum(patch_x[valid_patch] * kernel[valid_patch]) / kernel[valid_patch].sum()
                            inv_y_smooth[y, x] = np.sum(patch_y[valid_patch] * kernel[valid_patch]) / kernel[valid_patch].sum()
            
            # Apply smoothing
            blend = 0.5  # Blend factor
            mask = (inv_x >= 0)
            inv_x[mask] = blend * inv_x_smooth[mask] + (1 - blend) * inv_x[mask]
            inv_y[mask] = blend * inv_y_smooth[mask] + (1 - blend) * inv_y[mask]
            
            # 3. Remove outliers using simple median filter
            print("Removing outliers...")
            
            def median_filter_simple(arr, size=3):
                result = arr.copy()
                offset = size // 2
                for y in range(offset, arr.shape[0] - offset):
                    for x in range(offset, arr.shape[1] - offset):
                        if arr[y, x] >= 0:
                            patch = arr[y-offset:y+offset+1, x-offset:x+offset+1]
                            valid_patch = patch[patch >= 0]
                            if len(valid_patch) > 0:
                                result[y, x] = np.median(valid_patch)
                return result
            
            inv_x_median = median_filter_simple(inv_x, 3)
            inv_y_median = median_filter_simple(inv_y, 3)
            
            # Detect and fix outliers
            diff_x = np.abs(inv_x - inv_x_median)
            diff_y = np.abs(inv_y - inv_y_median)
            
            outliers = ((diff_x > 30) | (diff_y > 30)) & (inv_x >= 0)
            if outliers.sum() > 0:
                print(f"Correcting {outliers.sum()} outliers...")
                inv_x[outliers] = inv_x_median[outliers]
                inv_y[outliers] = inv_y_median[outliers]
        
        # Save the corrected LUTs
        np.save(asset_dir / 'cam_from_proj_x_fixed.npy', inv_x)
        np.save(asset_dir / 'cam_from_proj_y_fixed.npy', inv_y)
        
        # Also save as the main LUTs (backup originals first)
        import shutil
        shutil.copy(asset_dir / 'cam_from_proj_x.npy', 
                   asset_dir / 'cam_from_proj_x_backup.npy')
        shutil.copy(asset_dir / 'cam_from_proj_y.npy', 
                   asset_dir / 'cam_from_proj_y_backup.npy')
        
        np.save(asset_dir / 'cam_from_proj_x.npy', inv_x)
        np.save(asset_dir / 'cam_from_proj_y.npy', inv_y)
        
        print("✅ LUTs fixed and saved!")
        print("Now try 'Project LUT-Warped Registration' again")
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    return True


if __name__ == "__main__":
    fix_lut_distortions()