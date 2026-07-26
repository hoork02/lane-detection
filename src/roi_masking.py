"""Step 4: Region of Interest (ROI) Masking

Defines and applies a trapezoidal region of interest for lane detection.

Input: Edge maps from Step 3
Output: ROI masks and masked edge maps
"""

import cv2
import numpy as np
import glob
import os
from pathlib import Path


class ROIMasker:
    """Applies region of interest masking."""
    
    def __init__(self, step3_dir="data/output/step3_edge_detection",
                 output_dir="data/output/step4_roi_masking"):
        """Initialize ROI masker.
        
        Args:
            step3_dir (str): Edge detection outputs
            output_dir (str): Output directory
        """
        self.step3_dir = step3_dir
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def create_roi_mask(image_shape):
        """Create trapezoidal ROI mask.
        
        Covers the lower half of image where lanes are typically visible.
        Assumption: Camera position and lane geometry remain constant.
        
        Args:
            image_shape (tuple): (height, width, ...)
        
        Returns:
            ndarray: Binary ROI mask
        """
        height, width = image_shape[:2]
        
        # Define trapezoid vertices
        vertices = np.array([
            [0, height],                      # Bottom-left
            [int(width * 0.1), int(height * 0.5)],   # Top-left
            [int(width * 0.9), int(height * 0.5)],   # Top-right
            [width, height]                   # Bottom-right
        ], dtype=np.int32)
        
        # Create mask
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [vertices], 255)
        
        return mask
    
    def process_image(self, basename):
        """Apply ROI mask to edge image.
        
        Args:
            basename (str): Base filename
        
        Returns:
            tuple: (roi_mask, masked_edges) or (None, None) if failed
        """
        # Load edges
        edge_path = os.path.join(self.step3_dir, f"{basename}_edges_masked.jpg")
        edges = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        
        if edges is None:
            print(f"  ✗ Edge image not found: {basename}")
            return None, None
        
        # Create ROI
        roi_mask = self.create_roi_mask(edges.shape)
        
        # Apply mask
        masked_edges = cv2.bitwise_and(edges, roi_mask)
        
        return roi_mask, masked_edges
    
    def run(self):
        """Process all edge images."""
        print("\n" + "="*70)
        print("Step 4: Region of Interest (ROI) Masking")
        print("="*70)
        
        # Find edges
        edge_paths = glob.glob(os.path.join(self.step3_dir, "*_edges_masked.jpg"))
        
        if not edge_paths:
            print(f"  ⚠ No Step 3 outputs found")
            return
        
        print(f"Found {len(edge_paths)} image(s)\n")
        
        for idx, path in enumerate(edge_paths, 1):
            filename = os.path.basename(path)
            basename = filename.replace("_edges_masked.jpg", "")
            
            print(f"[{idx}/{len(edge_paths)}] Processing: {basename}")
            
            # Process
            roi_mask, masked_edges = self.process_image(basename)
            if roi_mask is None:
                continue
            
            # Save
            cv2.imwrite(os.path.join(self.output_dir, f"{basename}_roi_mask.jpg"), roi_mask)
            cv2.imwrite(os.path.join(self.output_dir, f"{basename}_edges_roi.jpg"), masked_edges)
            
            print(f"  ✓ ROI masking complete")
        
        print(f"\n✓ Step 4 complete! Outputs: {self.output_dir}\n")


if __name__ == "__main__":
    masker = ROIMasker()
    masker.run()
