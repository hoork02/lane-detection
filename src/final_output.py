"""Step 7: Final Lane Fitting and Output

Applies linear regression to fit lanes and generates final output.

Input: Original images and line segments from Step 5
Output: Final images with drawn lane lines
"""

import cv2
import numpy as np
import glob
import os
from pathlib import Path


class LaneDetectionFinal:
    """Fits lanes and generates final output."""
    
    def __init__(self, input_dir="data/input",
                 step4_dir="data/output/step4_roi_masking",
                 step5_dir="data/output/step5_line_detection",
                 output_dir="data/output/step7_final_output",
                 min_slope=0.3, max_slope=2.0):
        """Initialize final processor.
        
        Args:
            input_dir (str): Original images
            step4_dir (str): ROI outputs
            step5_dir (str): Line detection outputs
            output_dir (str): Output directory
            min_slope, max_slope (float): Slope filtering parameters
        """
        self.input_dir = input_dir
        self.step4_dir = step4_dir
        self.step5_dir = step5_dir
        self.output_dir = output_dir
        self.min_slope = min_slope
        self.max_slope = max_slope
        
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def create_roi_mask(image_shape):
        """Create ROI mask (same as Step 4)."""
        height, width = image_shape[:2]
        vertices = np.array([
            [0, height],
            [int(width * 0.1), int(height * 0.5)],
            [int(width * 0.9), int(height * 0.5)],
            [width, height]
        ], dtype=np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [vertices], 255)
        return mask
    
    @staticmethod
    def filter_lines_by_slope(line_segments, min_slope, max_slope):
        """Filter lines by slope (same as Step 6)."""
        left_lines, right_lines = [], []
        if not line_segments:
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
    
    @staticmethod
    def fit_line_linear_regression(lines):
        """Fit line using least squares method.
        
        Args:
            lines (list): (x1, y1, x2, y2, slope) tuples
        
        Returns:
            tuple: (slope, intercept) or None
        """
        if not lines:
            return None
        
        # Collect points
        x_points, y_points = [], []
        for x1, y1, x2, y2, _ in lines:
            x_points.extend([x1, x2])
            y_points.extend([y1, y2])
        
        x_points = np.array(x_points)
        y_points = np.array(y_points)
        
        if len(x_points) < 2:
            return None
        
        # Least squares
        n = len(x_points)
        sum_x = np.sum(x_points)
        sum_y = np.sum(y_points)
        sum_xy = np.sum(x_points * y_points)
        sum_x2 = np.sum(x_points ** 2)
        
        denominator = n * sum_x2 - sum_x ** 2
        if abs(denominator) < 1e-10:
            return None
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        
        return (slope, intercept)
    
    @staticmethod
    def draw_lane_lines(image, left_line, right_line, roi_mask):
        """Draw final lane lines on image.
        
        Args:
            image (ndarray): Original image (BGR)
            left_line (tuple): (slope, intercept) or None
            right_line (tuple): (slope, intercept) or None
            roi_mask (ndarray): ROI binary mask
        
        Returns:
            ndarray: Image with drawn lanes
        """
        result = image.copy()
        height, width = image.shape[:2]
        
        # Draw ROI outline
        roi_contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, roi_contours, -1, (255, 0, 255), 2)
        
        # Draw left lane (green)
        if left_line is not None:
            slope, intercept = left_line
            y1 = int(height * 0.5)
            y2 = height
            x1 = int((y1 - intercept) / slope) if slope != 0 else 0
            x2 = int((y2 - intercept) / slope) if slope != 0 else 0
            x1 = max(0, min(width - 1, x1))
            x2 = max(0, min(width - 1, x2))
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw right lane (green)
        if right_line is not None:
            slope, intercept = right_line
            y1 = int(height * 0.5)
            y2 = height
            x1 = int((y1 - intercept) / slope) if slope != 0 else width
            x2 = int((y2 - intercept) / slope) if slope != 0 else width
            x1 = max(0, min(width - 1, x1))
            x2 = max(0, min(width - 1, x2))
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        return result
    
    def process_image(self, basename):
        """Process single image: final lane fitting.
        
        Args:
            basename (str): Base filename
        
        Returns:
            ndarray: Final result or None if failed
        """
        # Load original
        for ext in ['.jpg', '.png']:
            img_path = os.path.join(self.input_dir, f"{basename}{ext}")
            if os.path.exists(img_path):
                break
        else:
            print(f"  ✗ Original image not found: {basename}")
            return None
        
        bgr = cv2.imread(img_path)
        if bgr is None:
            return None
        
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
            print(f"  ✗ Lines file not found: {basename}")
            return None
        
        # Filter by slope
        left_lines, right_lines = self.filter_lines_by_slope(segments, self.min_slope, self.max_slope)
        
        # Fit lines
        left_line = self.fit_line_linear_regression(left_lines)
        right_line = self.fit_line_linear_regression(right_lines)
        
        # Create ROI mask
        roi_mask = self.create_roi_mask(bgr.shape)
        
        # Draw
        result = self.draw_lane_lines(bgr, left_line, right_line, roi_mask)
        return result
    
    def run(self):
        """Process all original images."""
        print("\n" + "="*70)
        print("Step 7: Lane Fitting and Final Output")
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
            result = self.process_image(basename)
            if result is None:
                continue
            
            # Save
            cv2.imwrite(os.path.join(self.output_dir, f"{basename}_final.jpg"), result)
            print(f"  ✓ Lane detection complete")
        
        print(f"\n✓ Step 7 complete! Outputs: {self.output_dir}\n")
        print("="*70)
        print("✅ Lane detection pipeline finished!")
        print("="*70 + "\n")


if __name__ == "__main__":
    final = LaneDetectionFinal()
    final.run()
