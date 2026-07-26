"""
Step 4: Define Region of Interest (ROI)

This step:
1. Defines a trapezoidal region of interest covering the lower half of the image
2. The trapezoid covers the bottom corners and center
3. Assumption: Camera remains in constant place and lanes are flat

Input: Filtered Canny edge images from step3
Output: ROI mask and edges within ROI
"""

import cv2
import numpy as np
import glob
import os

step3_folder = "output_step3_canny/"
output_folder = "output_step4_roi/"
os.makedirs(output_folder, exist_ok=True)


def create_roi_mask(image_shape):
    """
    Create a trapezoidal region of interest mask covering the lower half of the image.
    The trapezoid covers the bottom corners and center, suitable for lane detection.
    
    Assumption: Camera remains in constant place and lanes are flat.
    """
    height, width = image_shape[:2]
    
    # Define trapezoid vertices (bottom-left, top-left, top-right, bottom-right)
    # Using lower half of image
    bottom_left = (0, height)
    top_left = (int(width * 0.1), int(height * 0.5))      # 10% from left, middle of image
    top_right = (int(width * 0.9), int(height * 0.5))     # 90% from left, middle of image
    bottom_right = (width, height)
    
    # Create mask
    mask = np.zeros((height, width), dtype=np.uint8)
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    
    return mask


# Process all filtered Canny edge images
edge_paths = glob.glob(step3_folder + "*_canny_filtered.jpg")

print("Step 4: Define Region of Interest (ROI)")
print("=" * 60)
print(f"Found {len(edge_paths)} edge images\n")

for edge_path in edge_paths:
    filename = os.path.basename(edge_path)
    # Extract basename (remove "_canny_filtered.jpg")
    basename = filename.replace("_canny_filtered.jpg", "")
    print(f"Processing: {basename}")
    
    # Load filtered Canny edges
    edges = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
    if edges is None:
        print(f"  Could not load edge image: {edge_path}")
        continue
    
    # Create ROI mask
    roi_mask = create_roi_mask(edges.shape)
    
    # Apply ROI mask to edges
    edges_roi = cv2.bitwise_and(edges, roi_mask)
    
    # Save ROI mask
    cv2.imwrite(os.path.join(output_folder, f"{basename}_roi_mask.jpg"), roi_mask)
    
    # Save edges within ROI
    cv2.imwrite(os.path.join(output_folder, f"{basename}_edges_roi.jpg"), edges_roi)
    
    print(f"  Saved: {basename}_roi_mask.jpg, {basename}_edges_roi.jpg")

print(f"\nStep 4 complete! Outputs saved to: {output_folder}")

