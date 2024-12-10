#!/usr/bin/env python
# coding: utf-8

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def calc_1d_FWHM(psf_integrated, psf_abscissa):
    """
    Calculate the 1D Full Width at Half Maximum (FWHM) of a PSF profile.

    Parameters
    ----------
    psf_integrated : ndarray
        Integrated intensity of the PSF along a specific axis.
    psf_abscissa : ndarray
        Pixel positions corresponding to the intensity values.

    Returns
    -------
    float
        The FWHM of the PSF profile along the given axis.

    Notes
    -----
    - The function determines the positions where the intensity is equal to half of the maximum value.
    - Linear interpolation is used to enhance the precision of the estimated FWHM.

    """
    max_intensity = np.max(psf_integrated)  # Peak intensity
    half_max = max_intensity / 2           # Half-maximum intensity

    # Position of the peak intensity
    max_pos = np.argmax(psf_integrated)

    # Identify positions around the half-maximum to the left of the peak
    half_pos_left_low = np.where((psf_abscissa < max_pos) & (psf_integrated < half_max))[0]
    half_pos_left_high = np.where((psf_abscissa < max_pos) & (psf_integrated >= half_max))[0]

    # Interpolate to find the exact position of the half-maximum intensity
    xp_left = np.array([half_pos_left_low[-1], half_pos_left_high[0]])
    fp_left = np.array([psf_integrated[half_pos_left_low[-1]], psf_integrated[half_pos_left_high[0]]])
    x_interp_half_max_left = np.linspace(xp_left[0], xp_left[1], 1000)
    y_interp_half_max_left = np.interp(x_interp_half_max_left, xp_left, fp_left)
    half_max_pos_left = x_interp_half_max_left[np.argmin(abs(y_interp_half_max_left - half_max))]

    # Identify positions around the half-maximum to the right of the peak
    half_pos_right_low = np.where((psf_abscissa > max_pos) & (psf_integrated < half_max))[0]
    half_pos_right_high = np.where((psf_abscissa > max_pos) & (psf_integrated >= half_max))[0]

    # Interpolate to find the exact position of the half-maximum intensity
    xp_right = np.array([half_pos_right_high[-1], half_pos_right_low[0]])
    fp_right = np.array([psf_integrated[half_pos_right_high[-1]], psf_integrated[half_pos_right_low[0]]])
    x_interp_half_max_right = np.linspace(xp_right[0], xp_right[1], 1000)
    y_interp_half_max_right = np.interp(x_interp_half_max_right, xp_right, fp_right)
    half_max_pos_right = x_interp_half_max_right[np.argmin(abs(y_interp_half_max_right - half_max))]

    # Calculate the FWHM as the difference between the left and right positions
    fwhm_1d = half_max_pos_right - half_max_pos_left

    return fwhm_1d


def calc_FWHM(psf_data, pixel_scale):
    """
    Calculate the 2D Full Width at Half Maximum (FWHM) of a PSF.

    Parameters
    ----------
    psf_data : ndarray
        2D PSF data array. For 3D data with only one slice, the function automatically squeezes to 2D.
    pixel_scale : float
        Pixel scale of the data in physical units (e.g., arcseconds per pixel).

    Returns
    -------
    float
        The FWHM of the PSF in physical units.

    Notes
    -----
    - The FWHM is computed as the average of the FWHMs along the x and y axes.

    """
    # Handle 3D PSF data with a single slice
    if len(psf_data.shape) == 3 and psf_data.shape[0] == 1:
        psf_data = psf_data.squeeze(axis=0)

    # Compute the integrated PSF profiles along the x and y axes
    psf_x = np.sum(psf_data, axis=0)
    psf_y = np.sum(psf_data, axis=1)

    # Create arrays for pixel indices along each axis
    psf_indices_x = np.arange(psf_data.shape[1])
    psf_indices_y = np.arange(psf_data.shape[0])

    # Calculate the FWHM along each axis
    fwhm_x = calc_1d_FWHM(psf_x, psf_indices_x)
    fwhm_y = calc_1d_FWHM(psf_y, psf_indices_y)

    # Compute the average FWHM and scale it to physical units
    fwhm = 0.5 * (fwhm_x + fwhm_y) * pixel_scale

    return fwhm