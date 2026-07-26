"""Non-maximum suppression for edge detection.

Implements edge thinning to produce thin, accurate edge maps.
"""

import numpy as np


def quantize_direction(theta):
    """Quantize gradient directions into 4 categories.
    
    Args:
        theta (ndarray): Gradient directions in degrees [0, 360]
    
    Returns:
        ndarray: Quantized directions (0, 1, 2, 3)
            0: Horizontal
            1: 45-degree diagonal
            2: Vertical
            3: 135-degree diagonal
    """
    q = np.zeros_like(theta, dtype=np.int32)
    
    # Horizontal (0°)
    q[(theta >= 0) & (theta < 22.5)] = 0
    q[(theta >= 157.5) & (theta < 202.5)] = 0
    q[(theta >= 337.5) & (theta <= 360)] = 0
    
    # 45-degree diagonal (45°)
    q[(theta >= 22.5) & (theta < 67.5)] = 1
    q[(theta >= 202.5) & (theta < 247.5)] = 1
    
    # Vertical (90°)
    q[(theta >= 67.5) & (theta < 112.5)] = 2
    q[(theta >= 247.5) & (theta < 292.5)] = 2
    
    # 135-degree diagonal (135°)
    q[(theta >= 112.5) & (theta < 157.5)] = 3
    q[(theta >= 292.5) & (theta < 337.5)] = 3
    
    return q


def non_max_suppression(magnitude, direction):
    """Apply non-maximum suppression to thin edges.
    
    Compares each pixel to its neighbors perpendicular to the gradient direction.
    Only pixels with local maxima are retained.
    
    Args:
        magnitude (ndarray): Gradient magnitude
        direction (ndarray): Gradient direction in degrees
    
    Returns:
        ndarray: Suppressed edge map
    """
    rows, cols = magnitude.shape
    suppressed = np.zeros((rows, cols), dtype=np.float32)
    quantized_dir = quantize_direction(direction)
    
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            # Get neighbors perpendicular to gradient
            current = magnitude[r, c]
            
            if quantized_dir[r, c] == 0:  # Horizontal
                neighbor1 = magnitude[r, c + 1]
                neighbor2 = magnitude[r, c - 1]
            elif quantized_dir[r, c] == 1:  # 45° diagonal
                neighbor1 = magnitude[r - 1, c + 1]
                neighbor2 = magnitude[r + 1, c - 1]
            elif quantized_dir[r, c] == 2:  # Vertical
                neighbor1 = magnitude[r - 1, c]
                neighbor2 = magnitude[r + 1, c]
            else:  # 135° diagonal
                neighbor1 = magnitude[r - 1, c - 1]
                neighbor2 = magnitude[r + 1, c + 1]
            
            # Keep only local maxima
            if current >= neighbor1 and current >= neighbor2:
                suppressed[r, c] = current
    
    return suppressed
