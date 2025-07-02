# calibration.py
import cv2
import numpy as np
from PIL import Image, ImageDraw


"""
    image_path = 'custom_registration_image.png'  # Your image file path
    img = Image.open(image_path)
    width, height = img.size
    line_color = 'white'  # Color for lines and outlines
    fill_color = 'white'   # Color for chessboard fill
    custom_image = create_custom_registration_image(width, height, line_color, fill_color)
    custom_image.save(image_path)  # Save the image
    show_image_fullscreen_on_second_monitor(image_path)  # Open image in full-screen on second monitor
"""
def create_custom_registration_image(width, height, line_color, fill_color):
    # Create a new image with a black background
    img = Image.new('RGB', (width, height), 'black')
    draw = ImageDraw.Draw(img)

    # Define properties
    large_font_size = 550  # Adjusted "F" size
    number_font_size = 200
    chessboard_size = 8
    chessboard_cell_size = 50
    circle_center = (width // 2, height // 2)
    circle_radius = min(width, height) // 4
    cross_size = 250
    gradient_bar_width = 200
    circle_thickness = 9  # Increased thickness for concentric circles
    cross_thickness = 60  # Increased thickness for cross
    f_thickness = 50      # Increased thickness for "F"

    # Draw the big "F" letter in the center
    x, y = width // 2 - large_font_size // 2, height // 2 - large_font_size // 2
    line_width = f_thickness
    draw.line([(x, y), (x + large_font_size * 0.8, y)], fill=line_color, width=line_width)  # Top horizontal
    draw.line([(x, y), (x, y + large_font_size * 0.6)], fill=line_color, width=line_width)  # Vertical part
    draw.line([(x, y + large_font_size * 0.4), (x + large_font_size * 0.6, y + large_font_size * 0.4)], fill=line_color, width=line_width)  # Middle horizontal

    # Draw numbers 1-6 with lines, slightly closer to the center
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

    # Draw a black and white gradient bar pattern on the left side
    for i in range(gradient_bar_width):
        color = (i * 255 // gradient_bar_width, i * 255 // gradient_bar_width, 
                 i * 255 // gradient_bar_width)  # Grayscale gradient
        draw.line([(i, 0), (i, height)], fill=color, width=1)

    # Draw concentric circles in the top-right corner
    for i in range(5):
        draw.ellipse([(width - circle_radius - i * 20, i * 20),
                      (width - i * 20, circle_radius + i * 20)],
                     outline=line_color, width=circle_thickness)

    # Draw a small chessboard pattern in the bottom center
    chessboard_start_x = (width - chessboard_size * chessboard_cell_size) // 2
    chessboard_start_y = height - chessboard_size * chessboard_cell_size
    for i in range(chessboard_size):
        for j in range(chessboard_size):
            top_left = (chessboard_start_x + i * chessboard_cell_size, 
                        chessboard_start_y + j * chessboard_cell_size)
            bottom_right = (chessboard_start_x + (i + 1) * chessboard_cell_size, 
                            chessboard_start_y + (j + 1) * chessboard_cell_size)
            fill = fill_color if (i + j) % 2 == 0 else 'black'
            draw.rectangle([top_left, bottom_right], fill=fill)

    # Draw a thick cross shape in the top-left corner
    cross_center = (cross_size, cross_size)
    draw.line([(cross_center[0] - cross_size, cross_center[1]), 
               (cross_center[0] + cross_size, cross_center[1])], 
              fill=line_color, width=cross_thickness)
    draw.line([(cross_center[0], cross_center[1] - cross_size), 
               (cross_center[0], cross_center[1] + cross_size)], 
              fill=line_color, width=cross_thickness)

    # Draw a smiley face in the bottom-right corner
    smiley_center = (width - 900, height - 700)
    smiley_radius = 50
    draw_smiley_face(draw, smiley_center, smiley_radius, line_color)
    
      # Draw a smiley face in the bottom-right corner
    smiley_center = (width - 1000, height - 950)
    smiley_radius = 100
    draw_smiley_face(draw, smiley_center, smiley_radius, line_color)
    print("Saving Custom Registration Image")
    img.save("./Assets/Generated/custom_registration_image.png")

    
"""
    Breaks down a homography matrix into its key geometric transformations: Translation (tx, ty), 
    Scaling (sx, sy) and Rotation (angle in degrees)
"""
def decompose_homography(H):
    # Ensure the matrix is normalized
    H = H / H[2, 2] #h_33

    # Extract translation
    tx = H[0, 2]
    ty = H[1, 2]

    # Extract rotation and scaling from the upper-left 2x2 part
    r1 = H[0, 0:2]
    r2 = H[1, 0:2]

    # Compute scaling factors
    sx = np.linalg.norm(r1)
    sy = np.linalg.norm(r2)

    # Normalize the rotation components
    r1 /= sx
    r2 /= sy

    # Construct the rotation matrix vertically
    R = np.vstack([r1, r2])

    # Compute the rotation angle
    angle = np.arctan2(r2[0], r2[1]) * (180 / np.pi)

    return tx, ty, sx, sy, angle

def find_homography():
    # Read images
    img2 = cv2.imread("./Assets/Generated/custom_registration_image.png")
    img1 = cv2.imread("./Assets/Generated/calibration_capture_image.png")

    # Convert images to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Get the dimensions of the second image
    height, width = img2_gray.shape

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors
    kp1, d1_image = sift.detectAndCompute(img1_gray, None)
    kp2, d2_image = sift.detectAndCompute(img2_gray, None)


    print(f"Keypoints detected: {len(kp1)} in image1, {len(kp2)} in image2")

    if d1_image is None or d2_image is None:
        raise RuntimeError("❌ Feature detection failed: No keypoints found in one or both images.")


    # Match descriptors
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(d1_image, d2_image)

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:int(len(matches) * 0.9)]

    # Check if there are enough matches
    if len(matches) < 4:
        print("Not enough matches found - at least 4 required. Returning Identity Matrix")
        return np.eye(3)

    no_of_matches = len(matches)

    # Allocate space for matched points
    p1_image = np.zeros((no_of_matches, 2))
    p2_image = np.zeros((no_of_matches, 2))

    # Store coordinates of the matched points
    for i in range(len(matches)):
        p1_image[i, :] = kp1[matches[i].queryIdx].pt
        p2_image[i, :] = kp2[matches[i].trainIdx].pt

    # Find the homography matrix
    homography, mask = cv2.findHomography(p1_image, p2_image, cv2.RANSAC)

    if homography is None:
        print("❌ Homography calculation failed. Returning identity matrix.")
        return np.eye(3)

    # Print the homography matrix
    print("Homography matrix:")
    print(homography)

    # Decompose the homography matrix
    tx, ty, sx, sy, angle = decompose_homography(homography)

    print(f"Translation: tx = {tx}, ty = {ty}")
    print(f"Scaling: sx = {sx}, sy = {sy}")
    print(f"Rotation angle: {angle} degrees")

    # Compute the inverse homography matrix
    # Alex's note: not too sure what the point of the inverse is here
    """
        inverse_homography = np.linalg.inv(homography)

        # Decompose the inv. homography matrix
        print("Inverse Homography matrix:")
        print(inverse_homography)

        tx, ty, sx, sy, angle = decompose_homography(inverse_homography)
        
        print(f"Translation: tx = {tx}, ty = {ty}")
        print(f"Scaling: sx = {sx}, sy = {sy}")
        print(f"Rotation angle: {angle} degrees")
    """
    # Warp the first image to align with the second image
    transformed_img = cv2.warpPerspective(img1, homography, (width, height))
    # Save the transformed image
    cv2.imwrite('./Assets/Generated/CalibOutput.jpg', transformed_img)

    """
        # Create a 2x2 subplot layout
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        # Plot the first original image
        axs[0, 0].imshow(cv2.cvtColor(cv2.flip(img1, 0), cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title('Calibration Pattern')
        axs[0, 0].axis('off')

        # Plot the second original image
        axs[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        axs[0, 1].set_title('Recorded Image')
        axs[0, 1].axis('off')

        # Compute the difference image
        diff_img = cv2.absdiff(img2, img1)
        # Plot the difference image between img2 and img1
        axs[1, 0].imshow(cv2.cvtColor(diff_img, cv2.COLOR_BGR2RGB))
        axs[1, 0].set_title('Difference Image (img2 - img1)')
        axs[1, 0].axis('off')


        # Plot the transformed image
        axs[1, 1].imshow(cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB))
        axs[1, 1].set_title('Transformed Image (mapping 1 onto 2)')
        axs[1, 1].axis('off')

        # Adjust layout
        plt.tight_layout()

        # Show plot
        plt.show()
    """
    np.save("homography_cam2proj.npy", homography)
    return homography




################################################################################################

## Helper Functions to create custom Registration Images
def draw_number(draw, position, number, size, color):
    """Draw numbers with lines."""
    x, y = position
    line_width = size // 10
    
    if number == 1:
        draw.line([(x + size // 2, y), (x + size // 2, y + size)], fill=color, width=line_width)
    elif number == 2:
        draw.line([(x, y), (x + size, y)], fill=color, width=line_width)  # Top
        draw.line([(x + size, y), (x + size, y + size // 2)], fill=color, width=line_width)  # Right
        draw.line([(x, y + size // 2), (x + size, y + size // 2)], fill=color, width=line_width)  # Middle
        draw.line([(x, y + size // 2), (x, y + size)], fill=color, width=line_width)  # Left
        draw.line([(x, y + size), (x + size, y + size)], fill=color, width=line_width)  # Bottom
    elif number == 3:
        draw.line([(x, y), (x + size, y)], fill=color, width=line_width)  # Top
        draw.line([(x, y + size // 2), (x + size, y + size // 2)], fill=color, width=line_width)  # Middle
        draw.line([(x, y + size), (x + size, y + size)], fill=color, width=line_width)  # Bottom
    elif number == 4:
        draw.line([(x + size, y), (x + size, y + size)], fill=color, width=line_width)  # Right
        draw.line([(x, y + size // 2), (x + size, y + size // 2)], fill=color, width=line_width)  # Middle
        draw.line([(x, y), (x, y + size // 2)], fill=color, width=line_width)  # Left
    elif number == 5:
        draw.line([(x, y), (x + size, y)], fill=color, width=line_width)  # Top
        draw.line([(x, y), (x, y + size // 2)], fill=color, width=line_width)  # Left
        draw.line([(x, y + size // 2), (x + size, y + size // 2)], fill=color, width=line_width)  # Middle
        draw.line([(x, y + size), (x + size, y + size)], fill=color, width=line_width)  # Bottom
    elif number == 6:
        draw.line([(x, y + size // 2), (x + size, y + size // 2)], fill=color, width=line_width)  # Middle
        draw.line([(x, y), (x, y + size)], fill=color, width=line_width)  # Left
        draw.line([(x, y + size), (x + size, y + size)], fill=color, width=line_width)  # Bottom
        draw.line([(x, y + size // 2), (x + size, y + size // 2)], fill=color, width=line_width)  # Diagonal

def draw_smiley_face(draw, center, radius, color):
    """Draw a smiley face at the specified center with the given radius."""
    x, y = center
    # Draw the face outline
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=color, width=3)
    # Draw the eyes
    eye_radius = radius // 6
    left_eye_center = (x - radius // 3, y - radius // 3)
    right_eye_center = (x + radius // 3, y - radius // 3)
    draw.ellipse([left_eye_center[0] - eye_radius, left_eye_center[1] - eye_radius,
                   left_eye_center[0] + eye_radius, left_eye_center[1] + eye_radius], 
                  fill=color)
    draw.ellipse([right_eye_center[0] - eye_radius, right_eye_center[1] - eye_radius,
                   right_eye_center[0] + eye_radius, right_eye_center[1] + eye_radius], 
                  fill=color)
    # Draw the mouth
    mouth_start = (x - radius // 2, y + radius // 4)
    mouth_end = (x + radius // 2, y + radius // 4)
    draw.arc([mouth_start[0], mouth_start[1] - 10, mouth_end[0], mouth_end[1] + 10],
              start=0, end=180, fill=color, width=3)