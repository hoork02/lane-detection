"""Custom filter implementations for image processing.

This module implements Gaussian derivative filters and convolution operations
used in the Canny edge detection pipeline.
"""

import numpy as np
from scipy import signal


def calculate_filter_size(sigma, T=0.01):
    """Calculate Gaussian filter size based on sigma.
    
    Args:
        sigma (float): Standard deviation of Gaussian kernel
        T (float): Threshold for filter cutoff
    
    Returns:
        tuple: (filter_size, half_size)
    """
    filter_size = int(2 * np.ceil(3 * sigma) + 1)
    if filter_size % 2 == 0:
        filter_size += 1
    return filter_size, filter_size // 2


def calculate_gradient(filter_size, sigma):
    """Calculate Gaussian derivative filters (Gx, Gy).
    
    Args:
        filter_size (int): Size of the filter kernel
        sigma (float): Standard deviation
    
    Returns:
        tuple: (Gx, Gy, scale) - x and y derivative filters and scale factor
    """
    half = filter_size // 2
    x = np.arange(-half, half + 1)
    y = np.arange(-half, half + 1)
    X, Y = np.meshgrid(x, y)
    
    # Gaussian function
    gaussian = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    
    # Derivatives
    Gx = -X * gaussian / (sigma**2)
    Gy = -Y * gaussian / (sigma**2)
    
    # Normalization
    scale = np.sqrt(np.sum(Gx**2) + np.sum(Gy**2))
    Gx = Gx / scale
    Gy = Gy / scale
    
    return Gx, Gy, scale


def convolve2d(image, kernel):
    """Apply 2D convolution to image.
    
    Args:
        image (ndarray): Input image
        kernel (ndarray): Convolution kernel
    
    Returns:
        ndarray: Convolved image
    """
    return signal.convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0)
