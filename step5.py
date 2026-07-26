"""
Step 5: Apply Hough Transform (Custom Implementation - No built-in functions)

This step:
1. Applies custom Hough transform to detect lines in the edge image
2. Hough transform groups edge points that lie on the same line by finding
   intersections in parameter space (rho, theta)
3. Converts detected lines to line segments

Input: Edges within ROI from step4
Output: Visualizations of detected lines
"""

import cv2
import numpy as np
import glob
import os

step4_folder = "output_step4_roi/"
output_folder = "output_step5_hough/"
os.makedirs(output_folder, exist_ok=True)

# Hough transform parameters
rho_res = 1          # Distance resolution in pixels
theta_res = np.pi / 180  # Angular resolution in radians (1 degree)
threshold = 50       # Minimum number of intersections to detect a line
min_line_length = 50 # Minimum number of pixels making up a line
max_line_gap = 100   # Maximum gap in pixels between connectable line segments


def hough_transform(edge_img, rho_res=1, theta_res=np.pi/180, threshold=50):
    """
    Custom Hough Transform implementation (no built-in functions).
    
    The Hough transform converts points from image space (x, y) to parameter space (rho, theta).
    For each edge point, it votes for all possible lines passing through that point.
    Lines are represented as: rho = x*cos(theta) + y*sin(theta)
    
    Args:
        edge_img: Binary edge image (0 or 255)
        rho_res: Distance resolution in pixels
        theta_res: Angular resolution in radians
        threshold: Minimum votes to detect a line
    
    Returns:
        List of (rho, theta) tuples representing detected lines
    """
    H, W = edge_img.shape
    max_rho = int(np.hypot(H, W))  # Maximum possible rho (diagonal length)
    
    # Create arrays for rho and theta values
    rhos = np.arange(-max_rho, max_rho + 1, rho_res)
    thetas = np.arange(0, np.pi, theta_res)
    
    # Initialize accumulator (voting space)
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    
    # Find all edge points
    y_idxs, x_idxs = np.nonzero(edge_img == 255)
    
    # For each edge point, vote for all possible lines passing through it
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        
        # For each possible theta, calculate corresponding rho
        for t_idx in range(len(thetas)):
            theta = thetas[t_idx]
            # Line equation: rho = x*cos(theta) + y*sin(theta)
            rho = x * np.cos(theta) + y * np.sin(theta)
            
            # Find the closest rho index
            r_idx = int(np.round((rho + max_rho) / rho_res))
            
            # Make sure index is within bounds
            if 0 <= r_idx < len(rhos):
                accumulator[r_idx, t_idx] += 1
    
    # Find peaks in accumulator (lines with votes >= threshold)
    lines = []
    for r_idx in range(accumulator.shape[0]):
        for t_idx in range(accumulator.shape[1]):
            if accumulator[r_idx, t_idx] >= threshold:
                rho = rhos[r_idx]
                theta = thetas[t_idx]
                lines.append((rho, theta, accumulator[r_idx, t_idx]))
    
    return lines


def hough_line_to_segments(rho, theta, img_shape, min_line_length=50, max_line_gap=100):
    """
    Convert a Hough line (rho, theta) to line segments within image bounds.
    This is similar to what HoughLinesP does - it finds line segments along a detected line.
    
    Args:
        rho: Distance from origin to line
        theta: Angle of line normal
        img_shape: (height, width) of image
        min_line_length: Minimum length of line segment
        max_line_gap: Maximum gap between segments
    
    Returns:
        List of line segments [(x1, y1, x2, y2), ...]
    """
    H, W = img_shape
    segments = []
    
    # Calculate line endpoints at image boundaries
    # Line equation: rho = x*cos(theta) + y*sin(theta)
    # Rearranging: y = (rho - x*cos(theta)) / sin(theta) if sin(theta) != 0
    #              x = (rho - y*sin(theta)) / cos(theta) if cos(theta) != 0
    
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Avoid division by zero
    if abs(sin_theta) < 1e-6:  # Vertical line
        x = rho / cos_theta if abs(cos_theta) > 1e-6 else 0
        if 0 <= x < W:
            return [(int(x), 0, int(x), H-1)]
        return []
    
    if abs(cos_theta) < 1e-6:  # Horizontal line
        y = rho / sin_theta
        if 0 <= y < H:
            return [(0, int(y), W-1, int(y))]
        return []
    
    # Find intersections with image boundaries
    points = []
    
    # Top edge (y = 0)
    x_top = (rho) / cos_theta
    if 0 <= x_top < W:
        points.append((x_top, 0))
    
    # Bottom edge (y = H-1)
    x_bottom = (rho - (H-1) * sin_theta) / cos_theta
    if 0 <= x_bottom < W:
        points.append((x_bottom, H-1))
    
    # Left edge (x = 0)
    y_left = rho / sin_theta
    if 0 <= y_left < H:
        points.append((0, y_left))
    
    # Right edge (x = W-1)
    y_right = (rho - (W-1) * cos_theta) / sin_theta
    if 0 <= y_right < H:
        points.append((W-1, y_right))
    
    # Remove duplicate points and keep only valid ones
    unique_points = []
    for x, y in points:
        x, y = int(x), int(y)
        if 0 <= x < W and 0 <= y < H:
            if (x, y) not in unique_points:
                unique_points.append((x, y))
    
    # If we have 2 points, create a line segment
    if len(unique_points) >= 2:
        # Use the two points that are furthest apart
        p1 = unique_points[0]
        p2 = unique_points[-1]
        x1, y1 = p1
        x2, y2 = p2
        
        # Check if line segment is long enough
        length = np.hypot(x2 - x1, y2 - y1)
        if length >= min_line_length:
            segments.append((x1, y1, x2, y2))
    
    return segments


def find_line_segments(edge_img, lines, min_line_length=50, max_line_gap=100):
    """
    Find line segments from Hough lines by checking which edge points lie on each line.
    
    Args:
        edge_img: Binary edge image
        lines: List of (rho, theta, votes) tuples from Hough transform
        min_line_length: Minimum length of line segment
        max_line_gap: Maximum gap between segments
    
    Returns:
        List of line segments [(x1, y1, x2, y2), ...]
    """
    H, W = edge_img.shape
    all_segments = []
    
    # Find all edge points
    y_idxs, x_idxs = np.nonzero(edge_img == 255)
    
    for rho, theta, votes in lines:
        # Find edge points that lie on this line (within tolerance)
        tolerance = 2  # pixels
        points_on_line = []
        
        for i in range(len(x_idxs)):
            x = x_idxs[i]
            y = y_idxs[i]
            # Calculate distance from point to line
            dist = abs(x * np.cos(theta) + y * np.sin(theta) - rho)
            if dist <= tolerance:
                points_on_line.append((x, y))
        
        if len(points_on_line) < 2:
            continue
        
        # Sort points along the line
        if abs(np.cos(theta)) > abs(np.sin(theta)):
            # Sort by x
            points_on_line.sort(key=lambda p: p[0])
        else:
            # Sort by y
            points_on_line.sort(key=lambda p: p[1])
        
        # Group points into segments
        segments = []
        current_segment = [points_on_line[0]]
        
        for i in range(1, len(points_on_line)):
            prev_point = current_segment[-1]
            curr_point = points_on_line[i]
            
            # Calculate gap
            gap = np.hypot(curr_point[0] - prev_point[0], curr_point[1] - prev_point[1])
            
            if gap <= max_line_gap:
                current_segment.append(curr_point)
            else:
                # End current segment and start new one
                if len(current_segment) >= 2:
                    x1, y1 = current_segment[0]
                    x2, y2 = current_segment[-1]
                    length = np.hypot(x2 - x1, y2 - y1)
                    if length >= min_line_length:
                        segments.append((x1, y1, x2, y2))
                current_segment = [curr_point]
        
        # Add last segment
        if len(current_segment) >= 2:
            x1, y1 = current_segment[0]
            x2, y2 = current_segment[-1]
            length = np.hypot(x2 - x1, y2 - y1)
            if length >= min_line_length:
                segments.append((x1, y1, x2, y2))
        
        all_segments.extend(segments)
    
    return all_segments

# Process all ROI edge images
edge_paths = glob.glob(step4_folder + "*_edges_roi.jpg")

print("Step 5: Apply Hough Transform")
print("=" * 60)
print(f"Found {len(edge_paths)} ROI edge images\n")

for edge_path in edge_paths:
    filename = os.path.basename(edge_path)
    # Extract basename (remove "_edges_roi.jpg")
    basename = filename.replace("_edges_roi.jpg", "")
    print(f"Processing: {basename}")
    
    # Load edges within ROI
    edges_roi = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
    if edges_roi is None:
        print(f"  Could not load edge image: {edge_path}")
        continue
    
    # Apply Custom Hough Transform (no built-in functions)
    # Hough transform detects lines by finding intersections in parameter space
    # It groups edge points that lie on the same line
    hough_lines = hough_transform(edges_roi, rho_res=rho_res, theta_res=theta_res, threshold=threshold)
    
    # Convert Hough lines to line segments
    line_segments = find_line_segments(edges_roi, hough_lines, min_line_length, max_line_gap)
    
    # Visualize all detected lines
    hough_vis = cv2.cvtColor(edges_roi, cv2.COLOR_GRAY2BGR)
    line_count = 0
    
    for x1, y1, x2, y2 in line_segments:
        cv2.line(hough_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        line_count += 1
    
    # Save visualization
    cv2.imwrite(os.path.join(output_folder, f"{basename}_hough_lines.jpg"), hough_vis)
    
    # Save line segments data for next step (as a simple format)
    # We'll save as a text file with line coordinates
    with open(os.path.join(output_folder, f"{basename}_hough_lines.txt"), 'w') as f:
        for x1, y1, x2, y2 in line_segments:
            f.write(f"{x1},{y1},{x2},{y2}\n")
    
    print(f"  Detected {len(hough_lines)} Hough lines, {line_count} line segments")
   

print(f"\nStep 5 complete! Outputs saved to: {output_folder}")

