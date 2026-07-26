"""
Step 6: Filter Lines by Slope

This step:
1. Filters lines that have appropriate slopes (not too horizontal or vertical)
2. Separates lines into left and right lanes based on slope sign:
   - Left Lane: negative slope (y decreases as x increases)
   - Right Lane: positive slope (y increases as x increases)

Input: Hough line detections from step5
Output: Visualizations of filtered and separated lines
"""

import cv2
import numpy as np
import glob
import os

step4_folder = "output_step4_roi/"
step5_folder = "output_step5_hough/"
output_folder = "output_step6_filter_lines/"
os.makedirs(output_folder, exist_ok=True)

# Slope thresholds to filter horizontal lines
min_slope = 0.3      # Minimum absolute slope (filter out near-horizontal lines)
max_slope = 2.0      # Maximum absolute slope (filter out near-vertical lines)


def filter_lines_by_slope(line_segments):
    """
    Filter lines by slope and separate into left and right lanes.
    
    Left Lane: As column (x) increases, row (y) decreases -> negative slope
    Right Lane: As column (x) increases, row (y) increases -> positive slope
    
    Also filters out lines that are too horizontal or too vertical.
    
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
        # Skip if line is too short
        if abs(x2 - x1) < 1:
            continue
            
        # Calculate slope
        slope = (y2 - y1) / (x2 - x1)
        
        # Filter by slope magnitude (remove near-horizontal and near-vertical lines)
        if abs(slope) < min_slope or abs(slope) > max_slope:
            continue
        
        # Separate into left (negative slope) and right (positive slope)
        if slope < 0:
            left_lines.append((x1, y1, x2, y2, slope))
        else:
            right_lines.append((x1, y1, x2, y2, slope))
    
    return left_lines, right_lines


# Process all ROI edge images
edge_paths = glob.glob(step4_folder + "*_edges_roi.jpg")

print("Step 6: Filter Lines by Slope")
print("=" * 60)
print(f"Found {len(edge_paths)} ROI edge images\n")

for edge_path in edge_paths:
    filename = os.path.basename(edge_path)
    basename = filename.replace("_edges_roi.jpg", "")
    print(f"Processing: {basename}")
    
    # Load edges within ROI
    edges_roi = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
    if edges_roi is None:
        print(f"  Could not load edge image: {edge_path}")
        continue
    
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
    
    # Visualize filtered lines
    filtered_vis = cv2.cvtColor(edges_roi, cv2.COLOR_GRAY2BGR)
    
    # Draw left lines in blue
    for x1, y1, x2, y2, _ in left_lines:
        cv2.line(filtered_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Draw right lines in red
    for x1, y1, x2, y2, _ in right_lines:
        cv2.line(filtered_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Save visualization
    cv2.imwrite(os.path.join(output_folder, f"{basename}_filtered_lines.jpg"), filtered_vis)
    
    print(f"  Found {len(left_lines)} left lines, {len(right_lines)} right lines")
    print(f"  Saved: {basename}_filtered_lines.jpg")

print(f"\nStep 6 complete! Outputs saved to: {output_folder}")

