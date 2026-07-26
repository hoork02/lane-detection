"""Step 5: Line Detection using Hough Transform

Custom Hough Transform implementation for detecting lines in edge images.

Input: ROI-masked edges from Step 4
Output: Detected line segments
"""

import cv2
import numpy as np
import glob
import os
from pathlib import Path


class HoughTransform:
    """Custom Hough Transform implementation."""
    
    def __init__(self, step4_dir="data/output/step4_roi_masking",
                 output_dir="data/output/step5_line_detection",
                 rho_res=1, theta_res=np.pi/180, threshold=50,
                 min_line_length=50, max_line_gap=100):
        """Initialize Hough Transform.
        
        Args:
            step4_dir (str): ROI masking outputs
            output_dir (str): Output directory
            rho_res (float): Distance resolution (pixels)
            theta_res (float): Angular resolution (radians)
            threshold (int): Minimum votes for line detection
            min_line_length (int): Minimum line segment length
            max_line_gap (int): Maximum gap between line segments
        """
        self.step4_dir = step4_dir
        self.output_dir = output_dir
        self.rho_res = rho_res
        self.theta_res = theta_res
        self.threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def hough_transform(self, edge_image):
        """Apply Hough Transform to detect lines.
        
        Converts edge points from image space to parameter space (ρ, θ)
        and finds peaks in the accumulator matrix.
        
        Args:
            edge_image (ndarray): Binary edge map
        
        Returns:
            list: Detected lines as (rho, theta, votes) tuples
        """
        H, W = edge_image.shape
        max_rho = int(np.hypot(H, W))
        
        # Create parameter space
        rhos = np.arange(-max_rho, max_rho + 1, self.rho_res)
        thetas = np.arange(0, np.pi, self.theta_res)
        
        # Accumulator
        accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
        
        # Find edge points
        y_idxs, x_idxs = np.nonzero(edge_image == 255)
        
        # Vote for all lines passing through each edge point
        for i in range(len(x_idxs)):
            x, y = x_idxs[i], y_idxs[i]
            for t_idx, theta in enumerate(thetas):
                rho = x * np.cos(theta) + y * np.sin(theta)
                r_idx = int(np.round((rho + max_rho) / self.rho_res))
                if 0 <= r_idx < len(rhos):
                    accumulator[r_idx, t_idx] += 1
        
        # Find peaks
        lines = []
        for r_idx in range(accumulator.shape[0]):
            for t_idx in range(accumulator.shape[1]):
                if accumulator[r_idx, t_idx] >= self.threshold:
                    lines.append((rhos[r_idx], thetas[t_idx], accumulator[r_idx, t_idx]))
        
        return lines
    
    def find_line_segments(self, edge_image, lines):
        """Convert Hough lines to line segments.
        
        Args:
            edge_image (ndarray): Binary edge map
            lines (list): Lines from hough_transform
        
        Returns:
            list: Line segments as (x1, y1, x2, y2) tuples
        """
        H, W = edge_image.shape
        y_idxs, x_idxs = np.nonzero(edge_image == 255)
        all_segments = []
        
        for rho, theta, _ in lines:
            # Find points on this line
            tolerance = 2
            points_on_line = []
            
            for i in range(len(x_idxs)):
                x, y = x_idxs[i], y_idxs[i]
                dist = abs(x * np.cos(theta) + y * np.sin(theta) - rho)
                if dist <= tolerance:
                    points_on_line.append((x, y))
            
            if len(points_on_line) < 2:
                continue
            
            # Sort points
            if abs(np.cos(theta)) > abs(np.sin(theta)):
                points_on_line.sort(key=lambda p: p[0])
            else:
                points_on_line.sort(key=lambda p: p[1])
            
            # Group into segments
            current = [points_on_line[0]]
            for i in range(1, len(points_on_line)):
                gap = np.hypot(points_on_line[i][0] - current[-1][0],
                              points_on_line[i][1] - current[-1][1])
                if gap <= self.max_line_gap:
                    current.append(points_on_line[i])
                else:
                    if len(current) >= 2:
                        x1, y1 = current[0]
                        x2, y2 = current[-1]
                        length = np.hypot(x2 - x1, y2 - y1)
                        if length >= self.min_line_length:
                            all_segments.append((x1, y1, x2, y2))
                    current = [points_on_line[i]]
            
            # Add final segment
            if len(current) >= 2:
                x1, y1 = current[0]
                x2, y2 = current[-1]
                length = np.hypot(x2 - x1, y2 - y1)
                if length >= self.min_line_length:
                    all_segments.append((x1, y1, x2, y2))
        
        return all_segments
    
    def process_image(self, basename):
        """Process single image: Hough line detection.
        
        Args:
            basename (str): Base filename
        
        Returns:
            list: Line segments or None if failed
        """
        # Load edges
        edge_path = os.path.join(self.step4_dir, f"{basename}_edges_roi.jpg")
        edges = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        
        if edges is None:
            print(f"  ✗ ROI edge image not found: {basename}")
            return None
        
        # Detect lines
        hough_lines = self.hough_transform(edges)
        if not hough_lines:
            print(f"  ⚠ No lines detected")
            return []
        
        # Convert to segments
        segments = self.find_line_segments(edges, hough_lines)
        return segments
    
    def run(self):
        """Process all ROI edge images."""
        print("\n" + "="*70)
        print("Step 5: Hough Line Detection")
        print("="*70)
        
        # Find ROI edges
        edge_paths = glob.glob(os.path.join(self.step4_dir, "*_edges_roi.jpg"))
        
        if not edge_paths:
            print(f"  ⚠ No Step 4 outputs found")
            return
        
        print(f"Found {len(edge_paths)} image(s)\n")
        
        for idx, path in enumerate(edge_paths, 1):
            filename = os.path.basename(path)
            basename = filename.replace("_edges_roi.jpg", "")
            
            print(f"[{idx}/{len(edge_paths)}] Processing: {basename}")
            
            # Process
            segments = self.process_image(basename)
            if segments is None:
                continue
            
            # Visualize
            edges = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            for x1, y1, x2, y2 in segments:
                cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Save
            cv2.imwrite(os.path.join(self.output_dir, f"{basename}_lines.jpg"), vis)
            
            # Save segments
            with open(os.path.join(self.output_dir, f"{basename}_lines.txt"), 'w') as f:
                for x1, y1, x2, y2 in segments:
                    f.write(f"{x1},{y1},{x2},{y2}\n")
            
            print(f"  ✓ Detected {len(segments)} line segments")
        
        print(f"\n✓ Step 5 complete! Outputs: {self.output_dir}\n")


if __name__ == "__main__":
    hough = HoughTransform()
    hough.run()
