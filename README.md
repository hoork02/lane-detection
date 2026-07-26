# Lane Detection Pipeline - Step-by-Step Implementation

This pipeline implements lane detection in 7 separate steps, each saved as a different Python file.

## Overview

All steps process images from the `test/` folder and save outputs to separate folders.

## Step Files

### Step 1: `step1_hsv_binary.py`
- **Purpose**: Convert RGB to HSV and create initial binary image
- **Input**: RGB images from `test/` folder
- **Output**: 
  - `output_step1_hsv_binary/` folder
  - `*_hsv.jpg` - HSV color space images
  - `*_binary_initial.jpg` - Binary images (all 1s/255)

### Step 2: `step2_filter_colors.py`
- **Purpose**: Apply Gaussian filter and filter lane colors (white/yellow)
- **Input**: HSV images and binary images from step1
- **Output**:
  - `output_step2_filter_colors/` folder
  - `*_binary_filtered.jpg` - Binary image after color filtering
  - `*_filtered_rgb.jpg` - RGB image with only lane colors

### Step 3: `step3_canny_edges.py`
- **Purpose**: Apply custom Canny edge detection
- **Input**: RGB images from `test/` and filtered binary images from step2
- **Output**:
  - `output_step3_canny/` folder
  - `*_canny_edges.jpg` - Canny edges before filtering
  - `*_canny_filtered.jpg` - Canny edges filtered by binary mask B

**Note**: Uses custom Canny implementation from assignment 2 (no built-in functions)

### Step 4: `step4_roi.py`
- **Purpose**: Define Region of Interest (trapezoidal mask)
- **Input**: Filtered Canny edges from step3
- **Output**:
  - `output_step4_roi/` folder
  - `*_roi_mask.jpg` - ROI mask visualization
  - `*_edges_roi.jpg` - Edges within ROI

**Assumption**: Camera remains in constant place and lanes are flat

### Step 5: `step5_hough.py`
- **Purpose**: Apply custom Hough Transform to detect lines
- **Input**: Edges within ROI from step4
- **Output**:
  - `output_step5_hough/` folder
  - `*_hough_lines.jpg` - Visualization of detected lines
  - `*_hough_lines.txt` - Line segments data (x1,y1,x2,y2 format)

**Note**: Uses **custom Hough Transform implementation** (no built-in functions)
- Converts edge points from image space to parameter space (rho, theta)
- Finds peaks in accumulator to detect lines
- Converts detected lines to line segments

### Step 6: `step6_filter_lines.py`
- **Purpose**: Filter lines by slope and separate into left/right lanes
- **Input**: Line segments from step5
- **Output**:
  - `output_step6_filter_lines/` folder
  - `*_filtered_lines.jpg` - Visualization (blue=left, red=right)

**Filtering**:
- Left Lane: negative slope (y decreases as x increases)
- Right Lane: positive slope (y increases as x increases)
- Filters out lines that are too horizontal (slope < 0.3) or too vertical (slope > 2.0)

### Step 7: `step7_final.py`
- **Purpose**: Apply linear regression and draw final results
- **Input**: Original images, line segments from step5, ROI mask from step4
- **Output**:
  - `output_step7_final/` folder
  - `*_final_result.jpg` - Final result with fitted lane lines

**Note**: Uses **custom Linear Regression implementation** (no built-in functions)
- Implements least squares method manually
- Fits a single line through each group of line segments
- Helps fill gaps in dashed lane lines

## Running the Pipeline

Run each step sequentially:

```bash
python3 step1.py
python3 step2.py
python3 step3.py    # May take time due to custom Canny
python3 step4.py
python3 step5.py
python3 step6.py
python3 step7.py
```

## Custom Implementations

### Hough Transform (Step 5)
- **No built-in functions used**
- Implements standard Hough Transform:
  1. Creates accumulator array in (rho, theta) parameter space
  2. For each edge point, votes for all possible lines passing through it
  3. Line equation: `rho = x*cos(theta) + y*sin(theta)`
  4. Finds peaks in accumulator (votes >= threshold)
  5. Converts detected lines to line segments

### Linear Regression (Step 7)
- **No built-in functions used**
- Implements least squares method manually:
  - Slope: `m = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)`
  - Intercept: `b = (Σy - m*Σx) / n`
- Fits line through all points from line segments in each group

## Dependencies

- OpenCV (cv2) - for image I/O and basic operations
- NumPy - for numerical operations
- Custom modules from assignment 2:
  - `masks.py` - Gaussian derivative filters
  - `grad.py` - Gradient computation
  - `non.py` - Non-maximum suppression
  - `hyst.py` - Hysteresis thresholding

