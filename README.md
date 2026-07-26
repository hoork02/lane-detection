# Lane Detection Pipeline

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive lane detection system using classical computer vision techniques: **Hough Transform** and **spatial line fitting**. This pipeline implements a modular 7-step approach to detect road lanes from video frames.

## 📋 Overview

This project demonstrates advanced image processing techniques without relying on deep learning. It combines:
- Custom edge detection (Canny)
- Hough Transform for line detection
- Linear regression for lane fitting
- Region of Interest (ROI) masking

## 🏗️ Architecture

The pipeline consists of 7 sequential processing steps:

| Step | Module | Purpose | Key Technique |
|------|--------|---------|---------------|
| 1 | `color_conversion.py` | RGB → HSV conversion | Color space transformation |
| 2 | `color_filtering.py` | Extract white/yellow lanes | Gaussian blur + HSV thresholding |
| 3 | `edge_detection.py` | Detect edges | Custom Canny implementation |
| 4 | `roi_masking.py` | Define region of interest | Trapezoidal mask |
| 5 | `line_detection.py` | Detect lines | Custom Hough Transform |
| 6 | `line_filtering.py` | Separate left/right lanes | Slope-based filtering |
| 7 | `final_output.py` | Fit and draw lanes | Linear regression |

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- OpenCV
- NumPy

### Installation

```bash
# Clone the repository
git clone https://github.com/hoork02/lane-detection.git
cd lane-detection

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Run the entire pipeline
python run_pipeline.py

# Or run individual steps
python -m src.color_conversion
python -m src.color_filtering
python -m src.edge_detection
python -m src.roi_masking
python -m src.line_detection
python -m src.line_filtering
python -m src.final_output
```

**Input:** Place test images in the `data/input/` folder

**Output:** Results are saved to `data/output/` with subdirectories for each step

## 📊 Pipeline Details

### Step 1: Color Space Conversion
- **Input:** RGB images
- **Process:** Convert to HSV color space for better color segmentation
- **Output:** HSV representations, binary initialization

### Step 2: Color Filtering
- **Input:** HSV images
- **Process:** Apply Gaussian blur and threshold for white/yellow pixels
- **Output:** Filtered binary images

### Step 3: Edge Detection
- **Input:** Grayscale images
- **Process:** Custom Canny edge detection (no OpenCV built-in)
  - Gaussian derivatives
  - Gradient magnitude and direction
  - Non-maximum suppression
  - Hysteresis thresholding
- **Output:** Edge maps

### Step 4: ROI Masking
- **Input:** Edge images
- **Process:** Define trapezoidal region of interest
- **Assumption:** Camera position and lane geometry remain constant
- **Output:** Masked edge images

### Step 5: Line Detection
- **Input:** ROI edge images
- **Process:** Custom Hough Transform (no OpenCV built-in)
  - Accumulator voting in (ρ, θ) parameter space
  - Peak detection for line candidates
  - Conversion to line segments
- **Output:** Detected line segments

### Step 6: Line Filtering
- **Input:** Detected line segments
- **Process:** Filter by slope magnitude, separate left/right lanes
  - Left lane: negative slope
  - Right lane: positive slope
  - Remove near-horizontal and near-vertical lines
- **Output:** Classified lane segments

### Step 7: Final Output
- **Input:** Original images + classified lines
- **Process:** Linear regression using least squares method
  - Fits single line through each lane group
  - Fills gaps in dashed lane markings
- **Output:** Final image with drawn lane lines

## 🔧 Configuration

Key parameters can be adjusted in each module:

```python
# Color filtering thresholds (HSV)
yellow_lower = np.array([18, 80, 80])
yellow_upper = np.array([35, 255, 255])
white_lower = np.array([0, 0, 200])
white_upper = np.array([180, 40, 255])

# Canny edge detection
sigma = 1.4              # Gaussian kernel sigma
Th = 40                  # High threshold
Tl = 10                  # Low threshold

# Hough Transform
rho_res = 1              # Distance resolution (pixels)
theta_res = np.pi/180   # Angular resolution (radians)
threshold = 50           # Minimum votes for line detection

# Line filtering
min_slope = 0.3          # Minimum slope magnitude
max_slope = 2.0          # Maximum slope magnitude
```

## 📁 Project Structure

```
lane-detection/
├── src/
│   ├── __init__.py
│   ├── color_conversion.py      # Step 1
│   ├── color_filtering.py       # Step 2
│   ├── edge_detection.py        # Step 3
│   ├── roi_masking.py           # Step 4
│   ├── line_detection.py        # Step 5
│   ├── line_filtering.py        # Step 6
│   ├── final_output.py          # Step 7
│   ├── filters.py               # Custom filter kernels
│   ├── gradient.py              # Gradient computation
│   ├── suppression.py           # Non-maximum suppression
│   └── thresholding.py          # Hysteresis thresholding
├── data/
│   ├── input/                   # Place test images here
│   └── output/                  # Results (auto-created)
├── run_pipeline.py              # Main pipeline script
├── requirements.txt             # Dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 🎯 Key Features

✅ **Pure Python Implementation**: No deep learning, fully interpretable algorithms

✅ **Custom Implementations**: Canny edge detection and Hough Transform built from scratch

✅ **Modular Design**: Each step is independent and can be reused

✅ **Well-Documented**: Comprehensive docstrings and comments

✅ **Configurable**: Easy to adjust parameters for different scenarios

## 📈 Performance Notes

- **Speed**: Step 3 (Canny) may take longer on large images due to custom implementation
- **Accuracy**: Works best on clear lane markings (white/yellow)
- **Robustness**: Sensitive to lighting conditions; test with various environments

## 🧮 Mathematical Foundations

### Hough Transform
Line equation in parameter space:
```
ρ = x·cos(θ) + y·sin(θ)
```
Where:
- `ρ`: Perpendicular distance from origin to line
- `θ`: Angle of the perpendicular
- `(x, y)`: Point on the line

### Linear Regression
Least squares fitting:
```
m = (n·Σxy - Σx·Σy) / (n·Σx² - (Σx)²)
b = (Σy - m·Σx) / n
```
Where:
- `m`: Slope
- `b`: Y-intercept
- `n`: Number of points

## 📦 Dependencies

- **opencv-python** (cv2) - Image I/O and basic operations
- **numpy** - Numerical computing

Optional:
- **matplotlib** - Visualization and debugging

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Adaptive thresholding for varying lighting conditions
- Curved lane detection
- Real-time video processing
- Performance optimization

## 📄 License

MIT License - see LICENSE file for details

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This project is designed for educational purposes to demonstrate classical computer vision techniques. For production lane detection systems, consider using state-of-the-art deep learning approaches.
