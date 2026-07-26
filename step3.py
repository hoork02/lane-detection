"""
Step 3: Apply Canny Edge Detection

This step:
1. Converts the input image I to grayscale (Ig)
2. Uses the custom Canny edge detector to detect edges
3. Uses the binary image B to remove all edge pixels that are set to zero in B

Input: RGB images from test/ and filtered binary images from step2
Output: Canny edge images (both before and after filtering with B)
"""

import cv2
import numpy as np
import glob
import os

# Import custom Canny modules
from masks import calculate_filter_size, calculate_gradient, convolve2d
from grad import normalize_to_uint8, compute_gradient_magnitude, compute_gradient_direction
from non import non_max_suppression
from hyst import hysteresis_thresholding

input_folder = "test/"
step2_folder = "output_step2_filter_colors/"
output_folder = "output_step3_canny/"
os.makedirs(output_folder, exist_ok=True)

# Canny parameters
sigma = 1.4
T = 0.01
Th = 40  # High threshold for hysteresis
Tl = 10  # Low threshold

# Pre-compute Canny filters
filter_size, half = calculate_filter_size(sigma, T)
Gx, Gy, scale = calculate_gradient(filter_size, sigma)

# Process all images
image_paths = glob.glob(input_folder + "*.jpg") + glob.glob(input_folder + "*.png")

print("Step 3: Apply Canny Edge Detection")
print("=" * 60)
print(f"Found {len(image_paths)} images\n")

for path in image_paths:
    filename = os.path.basename(path)
    basename = os.path.splitext(filename)[0]
    print(f"Processing: {filename}")
    
    # Load original RGB image
    bgr = cv2.imread(path)
    if bgr is None:
        print(f"  Could not load image: {path}")
        continue
    
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    # Load filtered binary image B from step 2
    binary_path = os.path.join(step2_folder, f"{basename}_binary_filtered.jpg")
    if not os.path.exists(binary_path):
        print(f"  Missing binary image from step 2: {binary_path}")
        continue
    
    B_img = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)
    # Convert to 0/1 mask (B is 0 or 255)
    B = (B_img > 0).astype(np.uint8)
    
    # Convert to grayscale
    Igray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian Derivative
    fx = convolve2d(Igray, Gx)
    fy = convolve2d(Igray, Gy)
    
    # Compute gradient magnitude + direction
    M, M_uint8 = compute_gradient_magnitude(fx, fy, scale)
    theta = compute_gradient_direction(fx, fy)
    
    # Non-maximum suppression
    NMS = non_max_suppression(M, theta)
    
    # Hysteresis thresholding
    edges = hysteresis_thresholding(NMS, Tl, Th)
    
    # Save Canny edges before filtering
    cv2.imwrite(os.path.join(output_folder, f"{basename}_canny_edges.jpg"), edges)
    
    # Remove edges where B == 0
    edges_filtered = edges * B
    
    # Save filtered Canny edges
    cv2.imwrite(os.path.join(output_folder, f"{basename}_canny_filtered.jpg"), edges_filtered)
    
    print(f"  Saved: {basename}_canny_edges.jpg, {basename}_canny_filtered.jpg")

print(f"\nStep 3 complete! Outputs saved to: {output_folder}")

