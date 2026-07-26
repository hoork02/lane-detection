"""Step 1: Color Space Conversion

Converts RGB images to HSV color space and creates initial binary masks.

Input: RGB images from data/input/
Output: HSV representations and binary initializations
"""

import cv2
import numpy as np
import glob
import os
from pathlib import Path


class ColorConverter:
    """Handles RGB to HSV conversion and binary image initialization."""
    
    def __init__(self, input_dir="data/input", output_dir="data/output/step1_color_conversion"):
        """Initialize converter with input/output directories.
        
        Args:
            input_dir (str): Directory containing input images
            output_dir (str): Directory for output images
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def process_image(self, image_path):
        """Process single image: RGB to HSV conversion.
        
        Args:
            image_path (str): Path to input image
        
        Returns:
            tuple: (hsv_image, binary_mask) or (None, None) if failed
        """
        # Load image (OpenCV loads in BGR format)
        bgr = cv2.imread(image_path)
        if bgr is None:
            print(f"  ✗ Could not load image: {image_path}")
            return None, None
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        # Convert RGB to HSV
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        
        # Create binary image (all pixels set to white/255)
        height, width = hsv.shape[:2]
        binary_mask = np.ones((height, width), dtype=np.uint8) * 255
        
        return hsv, binary_mask
    
    def run(self):
        """Process all images in input directory."""
        print("\n" + "="*70)
        print("Step 1: RGB to HSV Color Space Conversion")
        print("="*70)
        
        # Find all image files
        image_paths = glob.glob(os.path.join(self.input_dir, "*.jpg"))
        image_paths += glob.glob(os.path.join(self.input_dir, "*.png"))
        
        if not image_paths:
            print(f"  ⚠ No images found in {self.input_dir}")
            return
        
        print(f"Found {len(image_paths)} image(s)\n")
        
        for idx, path in enumerate(image_paths, 1):
            filename = os.path.basename(path)
            basename = os.path.splitext(filename)[0]
            
            print(f"[{idx}/{len(image_paths)}] Processing: {filename}")
            
            # Process image
            hsv, binary = self.process_image(path)
            if hsv is None:
                continue
            
            # Save outputs
            hsv_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imwrite(os.path.join(self.output_dir, f"{basename}_hsv.jpg"), hsv_bgr)
            cv2.imwrite(os.path.join(self.output_dir, f"{basename}_binary.jpg"), binary)
            
            print(f"  ✓ Saved HSV and binary mask")
        
        print(f"\n✓ Step 1 complete! Outputs: {self.output_dir}\n")


if __name__ == "__main__":
    converter = ColorConverter()
    converter.run()
