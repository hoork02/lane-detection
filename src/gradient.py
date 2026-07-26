"""Gradient computation utilities for edge detection.

Computes gradient magnitude and direction from image derivatives.
"""

import numpy as np


def compute_gradient_magnitude(fx, fy, scale=1.0):
    """Compute gradient magnitude from x and y derivatives.
    
    Args:
        fx (ndarray): X-direction derivative
        fy (ndarray): Y-direction derivative
        scale (float): Scale factor
    
    Returns:
        tuple: (magnitude_float, magnitude_uint8)
    """
    magnitude = np.sqrt(fx**2 + fy**2) / scale
    magnitude_uint8 = np.clip(magnitude * 255 / magnitude.max(), 0, 255).astype(np.uint8)
    return magnitude, magnitude_uint8


def compute_gradient_direction(fx, fy):
    """Compute gradient direction in degrees.
    
    Args:
        fx (ndarray): X-direction derivative
        fy (ndarray): Y-direction derivative
    
    Returns:
        ndarray: Direction in degrees [0, 360)
    """
    # Compute angle
    theta = np.arctan2(fy, fx) * 180 / np.pi
    # Convert to [0, 360)
    theta = (theta + 180) % 360
    return theta


def normalize_to_uint8(image):
    """Normalize image to uint8 range [0, 255].
    
    Args:
        image (ndarray): Input image
    
    Returns:
        ndarray: Normalized image as uint8
    """
    if image.max() == image.min():
        return np.zeros_like(image, dtype=np.uint8)
    normalized = (image - image.min()) / (image.max() - image.min())
    return (normalized * 255).astype(np.uint8)
