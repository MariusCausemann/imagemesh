import numpy as np
from numba import njit, prange


@njit(cache=True)
def compute_yvv_coeffs(sigma):
    """
    Computes the 3rd-order Young-van Vliet filter coefficients.
    """
    if sigma < 0.5:
        # For very small sigmas, return an identity-like pass-through
        return 1.0, 0.0, 0.0, 0.0

    # Calculate q based on the exact YvV piece-wise formulation
    if sigma >= 2.5:
        q = 0.98711 * sigma - 0.96330
    else:
        q = 3.97156 - 4.14554 * np.sqrt(1.0 - 0.26891 * sigma)

    # Compute raw coefficients
    b0 = 1.57825 + 2.44413 * q + 1.4281 * q**2 + 0.422205 * q**3
    b1 = 2.44413 * q + 2.85619 * q**2 + 1.26661 * q**3
    b2 = -(1.4281 * q**2 + 1.26661 * q**3)
    b3 = 0.422205 * q**3

    # Normalize by b0
    b1 /= b0
    b2 /= b0
    b3 /= b0

    # B is the scale factor ensuring a DC gain of 1.0
    B = 1.0 - (b1 + b2 + b3)

    return B, b1, b2, b3


@njit(cache=True)
def filter_1d(line, out, B, b1, b2, b3, burn_in):
    """
    Applies the forward and backward IIR pass on a 1D line with reflecting boundaries.
    Results are written directly into 'out'.
    """
    n = len(line)
    if n == 0:
        return

    # --- Forward Pass ---
    v1, v2, v3 = line[0], line[0], line[0]

    # Forward burn-in: simulate reflection by reading the start of the array backwards
    for i in range(burn_in - 1, -1, -1):
        idx = i if i < n else n - 1
        v_new = B * line[idx] + b1 * v1 + b2 * v2 + b3 * v3
        v3, v2, v1 = v2, v1, v_new

    # Main forward pass
    for i in range(n):
        v_new = B * line[i] + b1 * v1 + b2 * v2 + b3 * v3
        out[i] = v_new
        v3, v2, v1 = v2, v1, v_new

    # --- Backward Pass ---
    v1, v2, v3 = out[n - 1], out[n - 1], out[n - 1]

    # Backward burn-in: simulate reflection by reading the end of the array forwards
    for i in range(n - burn_in, n):
        idx = i if i >= 0 else 0
        v_new = B * out[idx] + b1 * v1 + b2 * v2 + b3 * v3
        v3, v2, v1 = v2, v1, v_new

    # Main backward pass (done in-place on 'out')
    for i in range(n - 1, -1, -1):
        v_new = B * out[i] + b1 * v1 + b2 * v2 + b3 * v3
        out[i] = v_new
        v3, v2, v1 = v2, v1, v_new


@njit(parallel=True, cache=True)
def yvv_gaussian_filter_3d(image, sigma):
    """
    Applies a separable 3D Young-van Vliet Gaussian filter in parallel.
    """
    B, b1, b2, b3 = compute_yvv_coeffs(sigma)

    # Burn-in distance (6*sigma is generally sufficient for 64-bit float convergence)
    burn_in = int(max(10, round(6 * sigma)))

    Z, Y, X = image.shape

    # Allocate buffers (float64 is highly recommended for IIR stability)
    temp1 = np.empty_like(image, dtype=np.float64)
    temp2 = np.empty_like(image, dtype=np.float64)

    # Pass 1: Filter along X (axis 2). Read from 'image', write to 'temp1'
    for z in prange(Z):
        for y in range(Y):
            filter_1d(image[z, y, :], temp1[z, y, :], B, b1, b2, b3, burn_in)

    # Pass 2: Filter along Y (axis 1). Read from 'temp1', write to 'temp2'
    for z in prange(Z):
        for x in range(X):
            filter_1d(temp1[z, :, x], temp2[z, :, x], B, b1, b2, b3, burn_in)

    # Pass 3: Filter along Z (axis 0). Read from 'temp2', write back to 'temp1'
    for y in prange(Y):
        for x in range(X):
            filter_1d(temp2[:, y, x], temp1[:, y, x], B, b1, b2, b3, burn_in)

    return temp1


def gaussian(image, sigma):
    return yvv_gaussian_filter_3d(image, sigma)
