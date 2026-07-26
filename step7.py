"""
Step 7: Linear Regression and Draw Final Results

This step:
1. Applies linear regression to fit a single line through each group of 
   line segments (left and right)
2. This fills gaps in dashed lane lines
3. Draws the final lane lines on the original image

Input: Original images from test/ and filtered lines from step6
Output: Final results with fitted lane lines
"""

import cv2
import numpy as np
import glob
import os

input_folder = "test/"
step4_folder = "output_step4_roi/"
step5_folder = "output_step5_hough/"
output_folder = "output_step7_final/"
os.makedirs(output_folder, exist_ok=True)

# Slope thresholds (same as step6)
min_slope = 0.3
max_slope = 2.0


def create_roi_mask(image_shape):
    """Create ROI mask (same as step4)"""
    height, width = image_shape[:2]
    bottom_left = (0, height)
    top_left = (int(width * 0.1), int(height * 0.5))
    top_right = (int(width * 0.9), int(height * 0.5))
    bottom_right = (width, height)
    mask = np.zeros((height, width), dtype=np.uint8)
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    return mask


def filter_lines_by_slope(line_segments):
    """
    Filter lines by slope (same as step6).
    
    Args:
        line_segments: List of (x1, y1, x2, y2) tuples
    
    Returns:
        left_lines, right_lines: Lists of (x1, y1, x2, y2, slope) tuples
    """
    left_lines = []
    right_lines = []
    
    if line_segments is None or len(line_segments) == 0:
        return left_lines, right_lines
    
    for x1, y1, x2, y2 in line_segments:
        if abs(x2 - x1) < 1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < min_slope or abs(slope) > max_slope:
            continue
        if slope < 0:
            left_lines.append((x1, y1, x2, y2, slope))
        else:
            right_lines.append((x1, y1, x2, y2, slope))
    
    return left_lines, right_lines


def fit_line_linear_regression(lines):
    """
    Use linear regression to fit a line through a group of line segments.
    This helps fill gaps in dashed lane lines.
    
    Uses least squares method: y = mx + b
    Returns: (slope, intercept) or None if not enough points
    """
    if len(lines) == 0:
        return None
    
    # Collect all points from line segments
    x_points = []
    y_points = []
    
    for x1, y1, x2, y2, _ in lines:
        x_points.extend([x1, x2])
        y_points.extend([y1, y2])
    
    x_points = np.array(x_points)
    y_points = np.array(y_points)
    
    if len(x_points) < 2:
        return None
    
    # Calculate linear regression using least squares method
    n = len(x_points)
    sum_x = np.sum(x_points)
    sum_y = np.sum(y_points)
    sum_xy = np.sum(x_points * y_points)
    sum_x2 = np.sum(x_points ** 2)
    
    # Calculate slope and intercept
    denominator = n * sum_x2 - sum_x ** 2
    if abs(denominator) < 1e-10:  # Avoid division by zero
        return None
    
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    
    return (slope, intercept)


def draw_lane_lines(image, left_line, right_line, roi_mask):
    """
    Draw the detected lane lines on the image within the ROI.
    Also visualizes the ROI region.
    """
    result = image.copy()
    height, width = image.shape[:2]
    
    # Draw ROI outline for visualization
    roi_contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, roi_contours, -1, (255, 0, 255), 2)  # Magenta outline
    
    # Draw left lane line
    if left_line is not None:
        slope, intercept = left_line
        # Calculate line endpoints within image bounds
        y1 = int(height * 0.5)  # Top of ROI
        y2 = height              # Bottom of image
        x1 = int((y1 - intercept) / slope) if slope != 0 else 0
        x2 = int((y2 - intercept) / slope) if slope != 0 else 0
        
        # Clip to image bounds
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        
        cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    # Draw right lane line
    if right_line is not None:
        slope, intercept = right_line
        # Calculate line endpoints within image bounds
        y1 = int(height * 0.5)  # Top of ROI
        y2 = height              # Bottom of image
        x1 = int((y1 - intercept) / slope) if slope != 0 else width
        x2 = int((y2 - intercept) / slope) if slope != 0 else width
        
        # Clip to image bounds
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        
        cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    return result


# Process all images
image_paths = glob.glob(input_folder + "*.jpg") + glob.glob(input_folder + "*.png")

print("Step 7: Linear Regression and Draw Final Results")
print("=" * 60)
print(f"Found {len(image_paths)} images\n")

for path in image_paths:
    filename = os.path.basename(path)
    basename = os.path.splitext(filename)[0]
    print(f"Processing: {filename}")
    
    # Load original image
    bgr = cv2.imread(path)
    if bgr is None:
        print(f"  Could not load image: {path}")
        continue
    
    # Load edges within ROI from step4
    edges_roi_path = os.path.join(step4_folder, f"{basename}_edges_roi.jpg")
    if not os.path.exists(edges_roi_path):
        print(f"  Missing ROI edges from step 4: {edges_roi_path}")
        continue
    
    edges_roi = cv2.imread(edges_roi_path, cv2.IMREAD_GRAYSCALE)
    
    # Load ROI mask
    roi_mask_path = os.path.join(step4_folder, f"{basename}_roi_mask.jpg")
    if not os.path.exists(roi_mask_path):
        print(f"  Missing ROI mask from step 4: {roi_mask_path}")
        continue
    
    roi_mask = cv2.imread(roi_mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Load line segments from step5 (custom Hough transform output)
    lines_file = os.path.join(step5_folder, f"{basename}_hough_lines.txt")
    line_segments = []
    
    if os.path.exists(lines_file):
        with open(lines_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 4:
                    x1, y1, x2, y2 = map(int, parts)
                    line_segments.append((x1, y1, x2, y2))
    else:
        print(f"  Missing Hough lines file from step 5: {lines_file}")
        continue
    
    # Filter lines by slope
    left_lines, right_lines = filter_lines_by_slope(line_segments)
    
    # Apply linear regression
    left_line = fit_line_linear_regression(left_lines)
    right_line = fit_line_linear_regression(right_lines)
    
    # Draw final results
    result = draw_lane_lines(bgr, left_line, right_line, roi_mask)
    
    # Save final result
    cv2.imwrite(os.path.join(output_folder, f"{basename}_final_result.jpg"), result)
    
    print(f"  Left line: {'Detected' if left_line else 'Not detected'}, "
          f"Right line: {'Detected' if right_line else 'Not detected'}")
    print(f"  Saved: {basename}_final_result.jpg")

print(f"\nStep 7 complete! Outputs saved to: {output_folder}")
print("\nAll steps complete! Lane detection pipeline finished.")

