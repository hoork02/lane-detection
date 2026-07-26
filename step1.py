"""
Step 1: Convert RGB to HSV and Create Binary Image B

This step:
1. Converts the input RGB image (I) to HSV color space
2. Creates a binary image B of the same size as the RGB image
3. Sets all pixels in B to 1 (white/255)

Input: RGB images from test/ folder
Output: HSV images and initial binary images (all 1s)
"""

import cv2
import numpy as np
import glob
import os

input_folder = "test/"
output_folder = "output_step1_hsv_binary/"
os.makedirs(output_folder, exist_ok=True)

# Process all images
image_paths = glob.glob(input_folder + "*.jpg") + glob.glob(input_folder + "*.png")

print("Step 1: Convert RGB to HSV and Create Binary Image B")
print("=" * 60)
print(f"Found {len(image_paths)} images\n")

for path in image_paths:
    filename = os.path.basename(path)
    basename = os.path.splitext(filename)[0]
    print(f"Processing: {filename}")
    
    # Load image (OpenCV loads BGR)
    bgr = cv2.imread(path)
    if bgr is None:
        print(f"  Could not load image: {path}")
        continue
    
    # Convert to RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    # Convert RGB to HSV
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    
    # Create binary image B of the same size, set all pixels to 1 (255)
    height, width = hsv.shape[:2]
    B = np.ones((height, width), dtype=np.uint8) * 255
    
    # Save HSV image
    cv2.imwrite(os.path.join(output_folder, f"{basename}-hsv.jpg"), hsv)
    hsv_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(os.path.join(output_folder, f"{basename}_hsv.jpg"), hsv_bgr)
    
    # Save binary image B (all 1s)
    cv2.imwrite(os.path.join(output_folder, f"{basename}_binary_initial.jpg"), B)
    
    print(f"  Saved: {basename}_hsv.jpg, {basename}_binary_initial.jpg")

print(f"\nStep 1 complete! Outputs saved to: {output_folder}")

