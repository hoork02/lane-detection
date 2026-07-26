"""Step 3: Edge Detection

Applies custom Canny edge detection to extract image edges.

Input: RGB images and filtered binary masks from Step 2
Output: Edge maps (before and after binary filtering)
"""

import cv2
import numpy as np
import glob
import os
from pathlib import Path
from filters import calculate_filter_size, calculate_gradient, convolve2d
from gradient import compute_gradient_magnitude, compute_gradient_direction
from suppression import non_max_suppression
from thresholding import hysteresis_thresholding


class CannyEdgeDetector:
    """Custom Canny edge detection implementation."""
    
    def __init__(self, step1_dir="data/output/step1_color_conversion",
                 step2_dir="data/output/step2_color_filtering",
                 input_dir="data/input",
                 output_dir="data/output/step3_edge_detection",
                 sigma=1.4, low_threshold=10, high_threshold=40):
        """Initialize edge detector.
        
        Args:
            step1_dir, step2_dir (str): Previous step outputs
            input_dir (str): Original images
            output_dir (str): Output directory
            sigma (float): Gaussian kernel sigma
            low_threshold (int): Low threshold for hysteresis
            high_threshold (int): High threshold for hysteresis
        """
        self.step1_dir = step1_dir
        self.step2_dir = step2_dir
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.sigma = sigma
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Pre-compute filters
        filter_size, _ = calculate_filter_size(sigma)
        self.Gx, self.Gy, self.scale = calculate_gradient(filter_size, sigma)
    
    def detect_edges(self, image):
        """Detect edges using Canny algorithm.
        
        Args:
            image (ndarray): Grayscale input image
        
        Returns:
            ndarray: Binary edge map
        """
        # Compute gradients
        fx = convolve2d(image, self.Gx)
        fy = convolve2d(image, self.Gy)
        
        # Magnitude and direction
        magnitude, _ = compute_gradient_magnitude(fx, fy, self.scale)
        direction = compute_gradient_direction(fx, fy)
        
        # Non-maximum suppression
        suppressed = non_max_suppression(magnitude, direction)
        
        # Hysteresis thresholding
        edges = hysteresis_thresholding(suppressed, self.low_threshold, self.high_threshold)
        
        return edges
    
    def process_image(self, basename):
        """Process single image: Canny edge detection.
        
        Args:
            basename (str): Base filename
        
        Returns:
            tuple: (edges, edges_masked) or (None, None) if failed
        """
        # Load original image
        for ext in ['.jpg', '.png']:
            img_path = os.path.join(self.input_dir, f"{basename}{ext}")
            if os.path.exists(img_path):
                break
        else:
            print(f"  ✗ Original image not found: {basename}")
            return None, None
        
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        
        # Detect edges
        edges = self.detect_edges(gray)
        
        # Load binary mask from Step 2
        binary_path = os.path.join(self.step2_dir, f"{basename}_binary_filtered.jpg")
        if not os.path.exists(binary_path):
            print(f"  ✗ Binary mask not found: {basename}")
            return None, None
        
        binary = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)
        binary_norm = (binary > 0).astype(np.uint8)
        
        # Apply mask
        edges_masked = edges * binary_norm
        
        return edges, edges_masked
    
    def run(self):
        """Process all images."""
        print("\n" + "="*70)
        print("Step 3: Canny Edge Detection")
        print("="*70)
        
        # Find binary images from Step 2
        binary_paths = glob.glob(os.path.join(self.step2_dir, "*_binary_filtered.jpg"))
        
        if not binary_paths:
            print(f"  ⚠ No Step 2 outputs found")
            return
        
        print(f"Found {len(binary_paths)} image(s)\n")
        
        for idx, path in enumerate(binary_paths, 1):
            filename = os.path.basename(path)
            basename = filename.replace("_binary_filtered.jpg", "")
            
            print(f"[{idx}/{len(binary_paths)}] Processing: {basename}")
            
            # Process
            edges, edges_masked = self.process_image(basename)
            if edges is None:
                continue
            
            # Save
            cv2.imwrite(os.path.join(self.output_dir, f"{basename}_edges.jpg"), edges)
            cv2.imwrite(os.path.join(self.output_dir, f"{basename}_edges_masked.jpg"), edges_masked)
            
            print(f"  ✓ Edge detection complete")
        
        print(f"\n✓ Step 3 complete! Outputs: {self.output_dir}\n")


if __name__ == "__main__":
    detector = CannyEdgeDetector()
    detector.run()
