#!/usr/bin/env python3
"""Main pipeline runner for lane detection.

Runs all 7 steps of the lane detection pipeline sequentially.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.color_conversion import ColorConverter
from src.color_filtering import ColorFilter
from src.edge_detection import CannyEdgeDetector
from src.roi_masking import ROIMasker
from src.line_detection import HoughTransform
from src.line_filtering import LineFilter
from src.final_output import LaneDetectionFinal


def main():
    """Run complete lane detection pipeline."""
    
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "  Lane Detection Pipeline - Classical Computer Vision".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70 + "\n")
    
    try:
        # Step 1: Color Conversion
        print("\n[1/7] Starting Step 1: Color Conversion...")
        converter = ColorConverter()
        converter.run()
        
        # Step 2: Color Filtering
        print("\n[2/7] Starting Step 2: Color Filtering...")
        filter = ColorFilter()
        filter.run()
        
        # Step 3: Edge Detection
        print("\n[3/7] Starting Step 3: Canny Edge Detection...")
        print("      (This may take a moment due to custom implementation)")
        detector = CannyEdgeDetector()
        detector.run()
        
        # Step 4: ROI Masking
        print("\n[4/7] Starting Step 4: ROI Masking...")
        masker = ROIMasker()
        masker.run()
        
        # Step 5: Line Detection
        print("\n[5/7] Starting Step 5: Hough Transform...")
        hough = HoughTransform()
        hough.run()
        
        # Step 6: Line Filtering
        print("\n[6/7] Starting Step 6: Line Filtering...")
        line_filter = LineFilter()
        line_filter.run()
        
        # Step 7: Final Output
        print("\n[7/7] Starting Step 7: Final Lane Fitting...")
        final = LaneDetectionFinal()
        final.run()
        
        print("\n" + "#"*70)
        print("#" + "  ✅ PIPELINE COMPLETE!".center(68) + "#")
        print("#" + " "*68 + "#")
        print("#" + "  Check 'data/output/' for results".center(68) + "#")
        print("#" + " "*68 + "#")
        print("#"*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
