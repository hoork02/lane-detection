"""Step 6: Line Filtering

Filters detected lines by slope and separates into left/right lanes.

Input: Line segments from Step 5
Output: Visualizations of filtered lines
"""

import cv2
import numpy as np
import glob
import os
from pathlib import Path


class LineFilter:
    """Filters and classifies lane lines."""
    
    def __init__(self, step4_dir="data/output/step4_roi_masking",
                 step5_dir="data/output/step5_line_detection",
                 output_dir="data/output/step6_line_filtering",
                 min_slope=0.3, max_slope=2.0):
        """Initialize line filter.
        
        Args:
            step4_dir (str): ROI masking outputs
            step5_dir (str): Line detection outputs
            output_dir (str): Output directory
            min_slope (float): Minimum slope magnitude
            max_slope (float): Maximum slope magnitude
        """
        self.step4_dir = step4_dir
        self.step5_dir = step5_dir
        self.output_dir = output_dir
        self.min_slope = min_slope
        self.max_slope = max_slope
        
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def filter_lines_by_slope(line_segments, min_slope, max_slope):
        """Filter lines by slope and separate into lanes.
        
        Left lane: negative slope (y decreases as x increases)
        Right lane: positive slope (y increases as x increases)
        
        Args:
            line_segments (list): (x1, y1, x2, y2) tuples
            min_slope (float): Minimum slope magnitude
            max_slope (float): Maximum slope magnitude
        
        Returns:
            tuple: (left_lines, right_lines)
        """
        left_lines = []
        right_lines = []
        
        if not line_segments:
            return left_lines, right_lines
        
        for x1, y1, x2, y2 in line_segments:
            if abs(x2 - x1) < 1:
                continue
            
            slope = (y2 - y1) / (x2 - x1)
            
            # Filter by slope magnitude
            if abs(slope) < min_slope or abs(slope) > max_slope:
                continue
            
            # Classify by sign
            if slope < 0:
                left_lines.append((x1, y1, x2, y2, slope))
            else:
                right_lines.append((x1, y1, x2, y2, slope))
        
        return left_lines, right_lines
    
    def process_image(self, basename):
        """Filter lines for single image.
        
        Args:
            basename (str): Base filename
        
        Returns:
            tuple: (left_lines, right_lines) or (None, None) if failed
        """
        # Load edges
        edge_path = os.path.join(self.step4_dir, f"{basename}_edges_roi.jpg")
        edges = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        if edges is None:
            print(f"  ✗ Edge image not found: {basename}")
            return None, None
        
        # Load line segments
        lines_file = os.path.join(self.step5_dir, f"{basename}_lines.txt")
        segments = []
        if os.path.exists(lines_file):
            with open(lines_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) == 4:
                        segments.append(tuple(map(int, parts)))
        else:
            print(f"  ✗ Line file not found: {basename}")
            return None, None
        
        # Filter
        left_lines, right_lines = self.filter_lines_by_slope(segments, self.min_slope, self.max_slope)
        return left_lines, right_lines
    
    def run(self):
        """Process all line detections."""
        print("\n" + "="*70)
        print("Step 6: Line Filtering by Slope")
        print("="*70)
        
        # Find line files
        line_files = glob.glob(os.path.join(self.step5_dir, "*_lines.txt"))
        
        if not line_files:
            print(f"  ⚠ No Step 5 outputs found")
            return
        
        print(f"Found {len(line_files)} image(s)\n")
        
        for idx, path in enumerate(line_files, 1):
            filename = os.path.basename(path)
            basename = filename.replace("_lines.txt", "")
            
            print(f"[{idx}/{len(line_files)}] Processing: {basename}")
            
            # Process
            left_lines, right_lines = self.process_image(basename)
            if left_lines is None:
                continue
            
            # Visualize
            edge_path = os.path.join(self.step4_dir, f"{basename}_edges_roi.jpg")
            edges = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
            vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            # Draw left (blue) and right (red)
            for x1, y1, x2, y2, _ in left_lines:
                cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            for x1, y1, x2, y2, _ in right_lines:
                cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Save
            cv2.imwrite(os.path.join(self.output_dir, f"{basename}_filtered_lines.jpg"), vis)
            
            print(f"  ✓ Found {len(left_lines)} left, {len(right_lines)} right lines")
        
        print(f"\n✓ Step 6 complete! Outputs: {self.output_dir}\n")


if __name__ == "__main__":
    filter = LineFilter()
    filter.run()
