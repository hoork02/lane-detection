import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from masks import calculate_filter_size, calculate_gradient, convolve2d
from grad import normalize_to_uint8, compute_gradient_magnitude, compute_gradient_direction

def quantize_direction(theta):
    """
    Quantize gradient directions into 4 categories (0,1,2,3).
    theta is in degrees [0–360].
    """
    q = np.zeros_like(theta, dtype=np.int32)

    # Horizontal (0)
    q[(theta >= 0) & (theta < 22.5)] = 0
    q[(theta >= 157.5) & (theta < 202.5)] = 0
    q[(theta >= 337.5) & (theta <= 360)] = 0

    # 45-degree diagonal (1)
    q[(theta >= 22.5) & (theta < 67.5)] = 1
    q[(theta >= 202.5) & (theta < 247.5)] = 1

    # Vertical (2)
    q[(theta >= 67.5) & (theta < 112.5)] = 2
    q[(theta >= 247.5) & (theta < 292.5)] = 2

    # 135-degree diagonal (3)
    q[(theta >= 112.5) & (theta < 157.5)] = 3
    q[(theta >= 292.5) & (theta < 337.5)] = 3

    return q

def non_max_suppression(M, theta):
    """
    Apply Non-Maximum Suppression (NMS) on gradient magnitude M
    using quantized gradient direction theta.
    """
    rows, cols = M.shape
    Z = np.zeros((rows, cols), dtype=np.float32)
    direction = quantize_direction(theta)

    for r in range(1, rows-1):
        for c in range(1, cols-1):
            q = 255
            r_ = 255

            if direction[r, c] == 0:   # Horizontal → compare left & right
                q = M[r, c+1]
                r_ = M[r, c-1]
            elif direction[r, c] == 1: # 45° diag → compare top-right & bottom-left
                q = M[r-1, c+1]
                r_ = M[r+1, c-1]
            elif direction[r, c] == 2: # Vertical → compare top & bottom
                q = M[r-1, c]
                r_ = M[r+1, c]
            elif direction[r, c] == 3: # 135° diag → compare top-left & bottom-right
                q = M[r-1, c-1]
                r_ = M[r+1, c+1]

            if (M[r, c] >= q) and (M[r, c] >= r_):
                Z[r, c] = M[r, c]
            else:
                Z[r, c] = 0

    return Z

