"""Hysteresis thresholding for edge detection.

Implements the final step of Canny edge detection to produce binary edge maps.
"""

import numpy as np
from scipy import ndimage


def hysteresis_thresholding(edges, low_threshold, high_threshold):
    """Apply hysteresis thresholding to edge map.
    
    Pixels are classified as:
    - Strong edge: magnitude >= high_threshold
    - Weak edge: low_threshold <= magnitude < high_threshold
    - Non-edge: magnitude < low_threshold
    
    Weak edges are kept only if connected to strong edges.
    
    Args:
        edges (ndarray): Gradient magnitude from non-maximum suppression
        low_threshold (float): Low threshold value
        high_threshold (float): High threshold value
    
    Returns:
        ndarray: Binary edge map (0 or 255)
    """
    # Initialize output
    output = np.zeros_like(edges, dtype=np.uint8)
    
    # Identify strong and weak edges
    strong = edges >= high_threshold
    weak = (edges >= low_threshold) & (edges < high_threshold)
    
    output[strong] = 255
    
    # Propagate weak edges connected to strong edges
    # Use binary dilation to connect weak edges to strong edges
    strong_edges = output.copy()
    
    # Iteratively dilate strong edges and apply weak mask
    for _ in range(max(edges.shape)):
        dilated = ndimage.binary_dilation(strong_edges > 0)
        output = np.where(weak & dilated, 255, output)
        if np.array_equal(output, strong_edges):
            break
        strong_edges = output.copy()
    
    return output
