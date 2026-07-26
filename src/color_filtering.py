"""Step 2: Color Filtering

Applies Gaussian filtering and extracts white/yellow lane pixels.

Input: HSV images from Step 1
Output: Filtered binary masks and RGB visualizations
"""

import cv2
import numpy as np
import glob
import os
from pathlib import Path


class ColorFilter:
    """Filters images to isolate white and yellow lane colors."""
    
    # HSV thresholds for lane colors
    YELLOW_LOWER = np.array([18, 80, 80])
    YELLOW_UPPER = np.array([35, 255, 255])
    WHITE_LOWER = np.array([0, 0, 200])
    WHITE_UPPER = np.array([180, 40, 255])
    
    def __init__(self, step1_dir="data/output/step1_color_conversion",
                 input_dir="data/input",
                 output_dir="data/output/step2_color_filtering"):
        """Initialize color filter.
        
        Args:
            step1_dir (str): Directory with Step 1 outputs
            input_dir (str): Directory with original images
            output_dir (str): Directory for output images
        """
        self.step1_dir = step1_dir
        self.input_dir = input_dir
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def process_image(self, basename):
        """Process single image: color filtering.
        
        Args:
            basename (str): Base filename without extension
        
        Returns:
            tuple: (binary_filtered, filtered_rgb) or (None, None) if failed
        """
        # Load original image
        img_path = os.path.join(self.input_dir, f"{basename}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.input_dir, f"{basename}.png")
        
        bgr = cv2.imread(img_path)
        if bgr is None:
            print(f"  ✗ Original image not found: {basename}")
            return None, None
        
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        # Load HSV from Step 1
        hsv_path = os.path.join(self.step1_dir, f"{basename}_hsv.jpg")
        hsv_bgr = cv2.imread(hsv_path)
        if hsv_bgr is None:
            print(f"  ✗ HSV image not found: {basename}")
            return None, None
        
        hsv = cv2.cvtColor(hsv_bgr, cv2.COLOR_BGR2HSV)
        
        # Load binary from Step 1
        binary_path = os.path.join(self.step1_dir, f"{basename}_binary.jpg")
        binary = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)
        if binary is None:
            print(f"  ✗ Binary image not found: {basename}")
            return None, None
        
        # Apply Gaussian blur
        hsv_blurred = cv2.GaussianBlur(hsv, (5, 5), 0)
        
        # Create color masks
        yellow_mask = cv2.inRange(hsv_blurred, self.YELLOW_LOWER, self.YELLOW_UPPER)
        white_mask = cv2.inRange(hsv_blurred, self.WHITE_LOWER, self.WHITE_UPPER)
        lane_mask = cv2.bitwise_or(yellow_mask, white_mask)
        
        # Apply mask to binary
        binary_filtered = cv2.bitwise_and(binary, lane_mask)
        
        # Create visualization
        lane_mask_3c = cv2.merge([lane_mask, lane_mask, lane_mask])
        filtered_rgb = cv2.bitwise_and(rgb, lane_mask_3c // 255)
        
        return binary_filtered, filtered_rgb
    
    def run(self):
        """Process all images from Step 1."""
        print("\n" + "="*70)
        print("Step 2: Color Filtering (White/Yellow)")
        print("="*70)
        
        # Find all HSV images from Step 1
        hsv_paths = glob.glob(os.path.join(self.step1_dir, "*_hsv.jpg"))
        
        if not hsv_paths:
            print(f"  ⚠ No Step 1 outputs found in {self.step1_dir}")
            return
        
        print(f"Found {len(hsv_paths)} image(s)\n")
        
        for idx, path in enumerate(hsv_paths, 1):
            filename = os.path.basename(path)
            basename = filename.replace("_hsv.jpg", "")
            
            print(f"[{idx}/{len(hsv_paths)}] Processing: {basename}")
            
            # Process
            binary_filt, rgb_filt = self.process_image(basename)
            if binary_filt is None:
                continue
            
            # Save
            cv2.imwrite(os.path.join(self.output_dir, f"{basename}_binary_filtered.jpg"), binary_filt)
            cv2.imwrite(os.path.join(self.output_dir, f"{basename}_filtered_rgb.jpg"),
                       cv2.cvtColor(rgb_filt, cv2.COLOR_RGB2BGR))
            
            print(f"  ✓ Color filtering complete")
        
        print(f"\n✓ Step 2 complete! Outputs: {self.output_dir}\n")


if __name__ == "__main__":
    filter = ColorFilter()
    filter.run()
