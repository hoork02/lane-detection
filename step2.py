"""
Step 2: Apply Gaussian Filter and Filter Lane Colors

This step:
1. Applies Gaussian filter to remove noise and small artifacts
2. Filters out (sets to zero) any pixel in binary image B that is not 
   the color of lanes (white or yellow) in HSV space
3. Updates binary image B accordingly

Input: HSV images and binary images from step1
Output: Filtered binary images and filtered RGB images
"""

import cv2
import numpy as np
import glob
import os

input_folder = "test/"
step1_folder = "output_step1_hsv_binary/"
output_folder = "output_step2_filter_colors/"
os.makedirs(output_folder, exist_ok=True)

# HSV thresholds for lane colors
yellow_lower = np.array([18, 80, 80])
yellow_upper = np.array([35, 255, 255])
white_lower = np.array([0, 0, 200])
white_upper = np.array([180, 40, 255])

# Process all images
image_paths = glob.glob(input_folder + "*.jpg") + glob.glob(input_folder + "*.png")

print("Step 2: Apply Gaussian Filter and Filter Lane Colors")
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
    
    # Load HSV image from step 1
    hsv_path = os.path.join(step1_folder, f"{basename}_hsv.jpg")
    if not os.path.exists(hsv_path):
        print(f"  Missing HSV image from step 1: {hsv_path}")
        continue
    hsv_bgr = cv2.imread(hsv_path)
    hsv = cv2.cvtColor(hsv_bgr, cv2.COLOR_BGR2HSV)
    
    # Load binary image B from step 1
    binary_path = os.path.join(step1_folder, f"{basename}_binary_initial.jpg")
    if not os.path.exists(binary_path):
        print(f"  Missing binary image from step 1: {binary_path}")
        continue
    B = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Gaussian filter for noise reduction
    hsv_blur = cv2.GaussianBlur(hsv, (5, 5), 0)
    
    # Create masks for yellow and white lanes
    yellow_mask = cv2.inRange(hsv_blur, yellow_lower, yellow_upper)
    white_mask = cv2.inRange(hsv_blur, white_lower, white_upper)
    
    # Combine masks (binary: 0 or 255)
    lane_mask = cv2.bitwise_or(yellow_mask, white_mask)
    
    # Ensure B_filtered is binary 0/255
    B_filtered = cv2.bitwise_and(B, lane_mask)
    
    # Create filtered RGB image (for visualization)
    # Convert lane_mask to 3-channel
    lane_mask_3c = cv2.merge([lane_mask, lane_mask, lane_mask])
    filtered_rgb = cv2.bitwise_and(rgb, lane_mask_3c)
    
    # Save filtered binary image B
    cv2.imwrite(os.path.join(output_folder, f"{basename}_binary_filtered.jpg"), B_filtered)
    
    # Save filtered RGB image (BGR format for saving)
    cv2.imwrite(os.path.join(output_folder, f"{basename}_filtered_rgb.jpg"),
                cv2.cvtColor(filtered_rgb, cv2.COLOR_RGB2BGR))
    
    print(f"  Saved: {basename}_binary_filtered.jpg, {basename}_filtered_rgb.jpg")

print(f"\nStep 2 complete! Outputs saved to: {output_folder}")
