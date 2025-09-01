
from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw


ASSETS = (Path(__file__).resolve().parent / "Assets").resolve()
GEN_DIR = (ASSETS / "Generated").resolve()
GEN_DIR.mkdir(parents=True, exist_ok=True)

REF_REG_IMG = GEN_DIR / "custom_registration_image.png"
CALIB_CAPTURE_IMG = GEN_DIR / "calibration_capture_image.png"
CALIB_OUTPUT_IMG = GEN_DIR / "CalibOutput.jpg"
HOMOGRAPHY_NPY = GEN_DIR / "homography_cam2proj.npy"





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
    print(f"✅ Custom registration image saved: {save_path}")
    return save_path





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


    print("🔍 Preprocessing images for better feature detection...")
    

    g_cap_enhanced = cv2.equalizeHist(g_cap)
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


    kp1, d1 = detector.detectAndCompute(g_cap_enhanced, None)
    kp2, d2 = detector.detectAndCompute(g_ref_enhanced, None)

    print(f"🔍 Enhanced keypoints: capture={len(kp1 or [])}, reference={len(kp2 or [])}")

    if d1 is None or d2 is None or len(kp1) < 8 or len(kp2) < 8:
        print("❌ Insufficient features detected. Try different lighting or pattern.")
        print(f"   Capture keypoints: {len(kp1 or [])}")
        print(f"   Reference keypoints: {len(kp2 or [])}")
        return np.eye(3, dtype=np.float64)


    matches = []
    

    try:
        bf = cv2.BFMatcher(norm, crossCheck=True)
        raw_matches = bf.match(d1, d2)
        matches = sorted(list(raw_matches), key=lambda m: m.distance)
        print(f"📍 Cross-check matches: {len(matches)}")
    except Exception as e:
        print(f"⚠️ Cross-check matching failed: {e}")


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


    if matches:

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

    if len(matches) < 8:
        print(f"❌ Insufficient quality matches: {len(matches)}/8 minimum")
        print("   💡 Try improving lighting, focus, or pattern visibility")
        return np.eye(3, dtype=np.float64)



    pts_cap = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts_ref = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)



    H = None
    inlier_count = 0
    

    try:
        H, inliers = cv2.findHomography(pts_cap, pts_ref, cv2.RANSAC, ransacReprojThreshold=3.0, confidence=0.995)
        if H is not None:
            inlier_count = int(inliers.sum()) if inliers is not None else len(matches)
            print(f"✅ Homography successful: {inlier_count}/{len(matches)} inliers")
    except Exception as e:
        print(f"⚠️ Homography failed: {e}")
    

    if H is None or inlier_count < len(matches) * 0.3:
        try:
            print("🔄 Trying LMEDS method...")
            H_lmeds, _ = cv2.findHomography(pts_cap, pts_ref, cv2.LMEDS)
            if H_lmeds is not None:
                H = H_lmeds
                inlier_count = len(matches)  # LMEDS doesn't provide inlier mask
                print(f"✅ LMEDS Homography successful")
        except Exception as e:
            print(f"⚠️ LMEDS homography failed: {e}")
    

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
            print("❌ Homography failed validation - using identity matrix")
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
            return np.eye(3, dtype=np.float64)
        else:
            print("✅ Homography passed validation checks")
            
    except Exception as e:
        print(f"⚠️ Could not validate homography: {e}")


    if save_outputs:
        h, w = g_ref.shape
        warped = cv2.warpPerspective(img_cap, H, (w, h))
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
