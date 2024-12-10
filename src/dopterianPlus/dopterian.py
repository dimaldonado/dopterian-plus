import numpy as np
import scipy.ndimage as scndi
import astropy.io.fits as pyfits
import astropy.convolution as apcon
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import statsmodels.api as sm
import scipy.integrate as scint 
import numpy.random as npr
import astropy.io.fits as pyfits
import scipy.optimize as scopt
import astropy.modeling as apmodel
import warnings
import scipy.ndimage as scndi
from . import cosmology as cosmos
import astropy.convolution as apcon
from astropy.cosmology import FlatLambdaCDM
import kcorrect
from scipy.optimize import curve_fit
from . import calc_FWHM
import os

version = '1.0.0'   


version = '1.0.0'   
c = 299792458. ## speed of light


## SDSS maggies to lupton
magToLup = {'u':1.4e-10,'g':0.9e-10,'r':1.2e-10,'i':1.8e-10,'z':7.4e-10}

#==============================================================================
# HELPER FUNCTIONS
#==============================================================================

def nu2lam(nu):
    """
    Convert frequency from Hz to wavelength in Angstrom.

    This function takes a frequency value in hertz (Hz) and converts it to
    wavelength in angstroms using the speed of light.

    Parameters
    ----------
    nu : float
        Frequency in hertz (Hz).

    Returns
    -------
    float
        Wavelength in angstroms.

    Notes
    -----
    The speed of light (`c`) is assumed to be a constant defined as 299792458 m/s.
    """
    return c / nu * 1e-10

    
def lam2nu(lam):
    """
    Convert wavelength from Angstrom to frequency in Hz.

    This function takes a wavelength value in angstroms and converts it to 
    frequency in hertz (Hz) using the speed of light.

    Parameters
    ----------
    lam : float
        Wavelength in angstroms.

    Returns
    -------
    float
        Frequency in hertz (Hz).

    Notes
    -----
    The speed of light (`c`) is assumed to be a constant defined as 299792458 m/s.
    """
    return c / lam * 1e10
    

def maggies2mags(maggies):
    """
    Convert flux density from maggies to magnitudes.

    This function converts a flux density value given in maggies 
    to the astronomical magnitude scale.

    Parameters
    ----------
    maggies : float or ndarray
        Flux density in maggies.

    Returns
    -------
    float or ndarray
        Equivalent magnitude(s) in the astronomical scale.
    """
    return -2.5 * np.log10(maggies)


def mags2maggies(mags):
    """
    Convert magnitudes to flux density in maggies.

    This function converts an astronomical magnitude value 
    to the equivalent flux density in maggies.

    Parameters
    ----------
    mags : float or ndarray
        Astronomical magnitude(s).

    Returns
    -------
    float or ndarray
        Equivalent flux density in maggies.
    """
    return 10 ** (-0.4 * mags)


def maggies2fnu(maggies):
    """
    Convert flux density from maggies to units of [erg s⁻¹ Hz⁻¹ cm⁻²].

    Parameters
    ----------
    maggies : float or ndarray
        Flux density in maggies.

    Returns
    -------
    float or ndarray
        Flux density in units of [erg s⁻¹ Hz⁻¹ cm⁻²].
    """
    return 3531e-23 * maggies


def fnu2maggies(fnu):
    """
    Convert flux density from units of [erg s⁻¹ Hz⁻¹ cm⁻²] to maggies.

    Parameters
    ----------
    fnu : float or ndarray
        Flux density in units of [erg s⁻¹ Hz⁻¹ cm⁻²].

    Returns
    -------
    float or ndarray
        Flux density in maggies.
    """
    return 3631e23 * fnu


def fnu2flam(fnu, lam):
    """
    Convert flux density from frequency space (fnu) to wavelength space (flam).

    Parameters
    ----------
    fnu : float or ndarray
        Flux density in units of [erg s⁻¹ Hz⁻¹ cm⁻²].
    lam : float or ndarray
        Wavelength in angstroms.

    Returns
    -------
    float or ndarray
        Flux density in units of [erg s⁻¹ Å⁻¹ cm⁻²].
    """
    return c * 1e10 / lam**2 * fnu


def flam2fnu(flam, lam):
    """
    Convert flux density from wavelength space (flam) to frequency space (fnu).

    Parameters
    ----------
    flam : float or ndarray
        Flux density in units of [erg s⁻¹ Å⁻¹ cm⁻²].
    lam : float or ndarray
        Wavelength in angstroms.

    Returns
    -------
    float or ndarray
        Flux density in units of [erg s⁻¹ Hz⁻¹ cm⁻²].
    """
    return flam / c / 1.0e10 * lam**2


def lambda_eff(lam, trans):
    """
    Calculate the effective wavelength of a filter.

    This function computes the mean wavelength of a filter by integrating 
    the product of wavelength and transmission over the wavelength range.

    Parameters
    ----------
    lam : ndarray
        Wavelength array in angstroms.
    trans : ndarray
        Transmission values corresponding to the wavelength array.

    Returns
    -------
    float
        Effective wavelength in angstroms.

    Raises
    ------
    ValueError
        If all wavelengths in `lam` are zero.
    """
    indexs = np.where(lam != 0)[0]
    if len(indexs) == 0:
        raise ValueError('ERROR: no non-zero wavelengths')
    else:
        Lambda = np.squeeze(lam[indexs])
        Transmission = np.squeeze(trans[indexs])
        return scint.simps(Lambda * Transmission, Lambda) / scint.simps(Transmission, Lambda)
    

def cts2maggies(cts, exptime, zp):
    """
    Convert counts to flux density in maggies.

    This function calculates flux density in maggies from the total counts, 
    exposure time, and zero-point magnitude.

    Parameters
    ----------
    cts : float or ndarray
        Total counts from the observation.
    exptime : float
        Exposure time in seconds.
    zp : float
        Zero-point magnitude.

    Returns
    -------
    float or ndarray
        Flux density in maggies.
    """
    return cts / exptime * 10**(-0.4 * zp)

    
def cts2mags(cts, exptime, zp):
    """
    Convert counts to magnitudes.

    This function calculates astronomical magnitudes from the total counts, 
    exposure time, and zero-point magnitude.

    Parameters
    ----------
    cts : float or ndarray
        Total counts from the observation.
    exptime : float
        Exposure time in seconds.
    zp : float
        Zero-point magnitude.

    Returns
    -------
    float or ndarray
        Astronomical magnitudes.
    """
    return maggies2mags(cts2maggies(cts, exptime, zp))


def maggies2cts(maggies, exptime, zp):
    """
    Convert flux density in maggies to counts.

    This function calculates the total counts from flux density in maggies, 
    exposure time, and zero-point magnitude.

    Parameters
    ----------
    maggies : float or ndarray
        Flux density in maggies.
    exptime : float
        Exposure time in seconds.
    zp : float
        Zero-point magnitude.

    Returns
    -------
    float or ndarray
        Total counts.
    """
    return maggies * exptime / 10**(-0.4 * zp)

    
def mags2cts(mags, exptime, zp):
    """
    Convert magnitudes to counts.

    This function calculates the total counts from astronomical magnitudes, 
    exposure time, and zero-point magnitude.

    Parameters
    ----------
    mags : float or ndarray
        Astronomical magnitudes.
    exptime : float
        Exposure time in seconds.
    zp : float
        Zero-point magnitude.

    Returns
    -------
    float or ndarray
        Total counts.
    """
    return maggies2cts(mags2maggies(mags), exptime, zp)


def maggies2lup(maggies, filtro):
    """
    Convert flux density in maggies to Lupton magnitudes.

    This function calculates Lupton magnitudes from flux density in maggies
    for a given filter.

    Parameters
    ----------
    maggies : float or ndarray
        Flux density in maggies.
    filtro : str
        Filter name (e.g., 'u', 'g', 'r', 'i', 'z').

    Returns
    -------
    float or ndarray
        Lupton magnitudes.

    Notes
    -----
    The conversion relies on a predefined mapping of filters to their corresponding
    scaling factor (`magToLup` dictionary).
    """
    b = magToLup[filtro]
    return -2.5 / np.log(10) * (np.arcsinh(maggies / b * 0.5) + np.log(b))
    
def lup2maggies(lup, filtro):
    """
    Convert Lupton magnitudes to flux density in maggies.

    This function calculates flux density in maggies from Lupton magnitudes
    for a given filter.

    Parameters
    ----------
    lup : float or ndarray
        Lupton magnitudes.
    filtro : str
        Filter name (e.g., 'u', 'g', 'r', 'i', 'z').

    Returns
    -------
    float or ndarray
        Flux density in maggies.

    Notes
    -----
    The conversion relies on a predefined mapping of filters to their corresponding
    scaling factor (`magToLup` dictionary).
    """
    b = magToLup[filtro]
    return 2 * b * np.sinh(-0.4 * np.log(10) * lup - np.log(b))

    
def random_indices(size, indexs):
    """
    Generate a random selection of unique indices.

    This function returns an array containing a specified number of unique
    indices randomly chosen from a given list or array.

    Parameters
    ----------
    size : int
        Number of unique indices to select.
    indexs : array-like
        Array or list of indices to sample from.

    Returns
    -------
    ndarray
        Array of randomly selected unique indices.

    Raises
    ------
    ValueError
        If `size` is larger than the number of available indices in `indexs`.
    """
    return npr.choice(indexs, size=size, replace=False)


def edge_index(a, rx, ry):
    """
    Generate indices of a ring-shaped region in a 2D array.

    This function creates an index list of elements forming a ring with a 
    width of 1 around the center of the array, defined by radii `rx` and `ry`.

    Parameters
    ----------
    a : ndarray
        Input 2D array.
    rx : int
        Radius along the x-axis from the center of the array.
    ry : int
        Radius along the y-axis from the center of the array.

    Returns
    -------
    tuple of ndarrays
        Tuple containing the row and column indices of the ring elements.
    """
    N, M = a.shape
    XX, YY = np.meshgrid(np.arange(N), np.arange(M))
    
    Y = np.abs(XX - N / 2.0).astype(np.int64)
    X = np.abs(YY - M / 2.0).astype(np.int64)
    
    idx = np.where(((X == rx) * (Y <= ry)) + ((Y == ry) * (X <= rx)))
    return idx


def predict_redshifted_FWHM(a_low, z_low, z_high):
    """
    Predict the Full Width at Half Maximum (FWHM) at a higher redshift.

    This function calculates the predicted FWHM at a higher redshift based on
    the FWHM at a lower redshift and the corresponding luminosity distances.

    Parameters
    ----------
    a_low : float
        FWHM at the lower redshift.
    z_low : float
        Lower redshift value.
    z_high : float
        Higher redshift value.

    Returns
    -------
    float
        Predicted FWHM at the higher redshift.

    Notes
    -----
    The calculation considers the luminosity distances and scaling factors 
    for the redshifts.
    """
    d_low = cosmos.luminosity_distance(z_low) 
    d_high = cosmos.luminosity_distance(z_high)

    a_high = a_low * (d_low / (1 + z_low)**2) / (d_high / (1 + z_high)**2)

    return a_high


def check_PSF_FWHM(psf_low, psf_high, pixcale_low_z, pixscale_high_z, z_low, z_high):
    """
    Verify if the PSF FWHM at low redshift is compatible with the PSF at high redshift.

    This function checks whether the redshifted FWHM of the PSF at low redshift
    is smaller than or equal to the FWHM of the PSF at high redshift.

    Parameters
    ----------
    psf_low : ndarray
        PSF image at the lower redshift.
    psf_high : ndarray
        PSF image at the higher redshift.
    pixcale_low_z : float
        Pixel scale at the lower redshift.
    pixscale_high_z : float
        Pixel scale at the higher redshift.
    z_low : float
        Lower redshift value.
    z_high : float
        Higher redshift value.

    Returns
    -------
    bool
        True if the redshifted low-z PSF FWHM is compatible with the high-z PSF,
        False otherwise.
    """
    low_z_fwhm = calc_FWHM.calc_FWHM(psf_low, pixcale_low_z)
    high_z_fwhm = calc_FWHM.calc_FWHM(psf_high, pixscale_high_z)

    redshifted_low_z_fwhm = predict_redshifted_FWHM(low_z_fwhm, z_low, z_high)

    if redshifted_low_z_fwhm > high_z_fwhm:
        return False
    
    return True
    

def predict_redshifted_image_size(z_low, z_high, n_low, pixscale_low_z, pixscale_high_z):
    """
    Predict the image size at a higher redshift.

    This function calculates the expected size of an image at a higher redshift
    based on its size at a lower redshift, accounting for luminosity distances,
    redshift scaling factors, and pixel scales.

    Parameters
    ----------
    z_low : float
        Lower redshift value.
    z_high : float
        Higher redshift value.
    n_low : int
        Image size (number of pixels) at the lower redshift.
    pixscale_low_z : float
        Pixel scale at the lower redshift.
    pixscale_high_z : float
        Pixel scale at the higher redshift.

    Returns
    -------
    int
        Predicted image size (number of pixels) at the higher redshift.
    """
    d_low = cosmos.luminosity_distance(z_low) 
    d_high = cosmos.luminosity_distance(z_high)

    mag_factor = (d_low / d_high) * ((1 + z_high)**2 / (1 + z_low)**2) * (pixscale_low_z / pixscale_high_z)
    
    n_high = int(round(n_low * mag_factor))

    return n_high
   

def dist_ellipse(img, xc, yc, q, ang):
    """
    Compute distances to the center in elliptical apertures.

    This function calculates the distance of each pixel in an image
    to the center `(xc, yc)` using elliptical apertures, accounting
    for axis ratio and rotation angle.

    Parameters
    ----------
    img : ndarray
        Input 2D image.
    xc : float
        X-coordinate of the center.
    yc : float
        Y-coordinate of the center.
    q : float
        Axis ratio of the ellipse (minor/major).
    ang : float
        Rotation angle of the ellipse in degrees.

    Returns
    -------
    ndarray
        A 2D array where each value represents the distance of the
        corresponding pixel to the center `(xc, yc)` in the elliptical frame.
    """
    ang = np.radians(ang)

    X, Y = np.meshgrid(range(img.shape[1]), range(int(img.shape[0])))
    rX = (X - xc) * np.cos(ang) - (Y - yc) * np.sin(ang)
    rY = (X - xc) * np.sin(ang) + (Y - yc) * np.cos(ang)
    dmat = np.sqrt(rX * rX + (1 / (q * q)) * rY * rY)
    return dmat


def robust_linefit(x, y):
    """
    Perform a robust linear fit.

    This function fits a robust linear model to the data using the
    HuberT norm to reduce the influence of outliers.

    Parameters
    ----------
    x : array-like
        Independent variable.
    y : array-like
        Dependent variable.

    Returns
    -------
    ndarray
        Parameters of the fitted linear model, where the first element
        is the intercept and the second element is the slope.

    Notes
    -----
    This function uses the `RLM` (Robust Linear Model) from the `statsmodels`
    library with the HuberT norm.
    """
    print(x)
    print(y)
    X = sm.add_constant(x)  # Agrega una constante para el término independiente
    robust_model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
    results = robust_model.fit()
    return results.params

def resistent_mean(a, k):
    """
    Compute the mean value of an array using a k-sigma clipping method.

    This function iteratively removes outliers beyond `k` standard deviations
    from the mean, recalculating the mean and standard deviation until no
    outliers remain.

    Parameters
    ----------
    a : array-like 
        Input array of values.
    k : float
        Number of standard deviations for clipping.

    Returns
    -------
    float
        Clipped mean of the array.
    float
        Clipped standard deviation of the array.
    int
        Number of rejected outliers.

    Notes
    -----
    Values of zero in the input array are excluded from the calculations.
    """
    a = np.asanyarray(a)
    media = np.nanmean(a)
    dev = np.nanstd(a)

    back = a.copy()
    back = back[back != 0]
    thresh = media + k * dev
    npix = len(a[a >= thresh])
    while npix > 0:
        back = back[back < thresh]
        media = np.mean(back)
        dev = np.std(back)
        thresh = media + k * dev
        npix = len(back[back >= thresh])
        
    nrej = np.size(a[a >= thresh])
    return media, dev, nrej


def ring_sky(image, width0, nap, x=None, y=None, q=1, pa=0, rstart=None, nw=None):
    """
    Measure the flux around a position in elliptical rings.

    This function calculates the flux around a specified position `(x, y)`
    in elliptical rings with a given axis ratio, position angle, and
    aperture configuration.

    Parameters
    ----------
    image : ndarray
        Input 2D image array.
    width0 : float
        Initial width of the rings in pixels.
    nap : int
        Number of apertures. Must be greater than 3.
    x : float, optional
        X-coordinate of the center. Defaults to the image center.
    y : float, optional
        Y-coordinate of the center. Defaults to the image center.
    q : float, optional
        Axis ratio of the ellipse (minor/major). Defaults to 1 (circular).
    pa : float, optional
        Position angle of the ellipse in degrees. Defaults to 0.
    rstart : float, optional
        Starting radius for the rings. Defaults to 5% of the smallest image dimension.
    nw : int, optional
        If set, limits the number of apertures to `nw`.

    Returns
    -------
    float
        Measured sky flux in the elliptical rings.

    Raises
    ------
    ValueError
        If `nap` is less than or equal to 3.
        If only one of `x` or `y` is specified.
    """
    if nap <= 3:
        raise ValueError('Number of apertures must be greater than 3.')

    if type(image) == str:
        image = pyfits.getdata(image)
    
    N, M = image.shape
    if rstart is None:
        rstart = 0.05 * min(N, M)
    
    if x is None and y is None:
        x = N * 0.5
        y = M * 0.5
    elif x is None or y is None:
        raise ValueError('X and Y must both be set to a value')

    rad = dist_ellipse(image, x, y, q, pa)
    max_rad = 0.95 * np.amax(rad)
    
    if nw is None:
        width = width0
    else:
        width = max_rad / float(width0)
    
    media, sig, nrej = resistent_mean(image, 3)
    sig *= np.sqrt(np.size(image) - 1 - nrej)
    
    if rstart is None:
        rhi = width
    else:
        rhi = rstart
    
    nmeasures = 2
    r = np.array([])
    flux = np.array([])
    i = 0

    while rhi <= max_rad:
        extra = 0
        ct = 0
        while ct < 10:
            idx = (rad <= rhi + extra) * (rad >= rhi - extra) * (np.abs(image) < 3 * sig)
            ct = np.size(image[idx])
            extra += 1
            if extra > max(N, M) * 2:
                break
            
        if ct < 5:
            sky = flux[len(flux) - 1]
        else:
            sky = resistent_mean(image[idx], 3)[0]
        
        r = np.append(r, rhi - 0.5 * width)
        flux = np.append(flux, sky)
        
        i += 1
        if np.size(flux) > nap:
            valid_indices = np.isfinite(r[i - nap + 1:i]) & np.isfinite(flux[i - nap + 1:i])
            if np.sum(valid_indices) > 0:
                pars, err = scopt.curve_fit(lambda x, a, b: a * x + b, r[i - nap + 1:i][valid_indices], flux[i - nap + 1:i][valid_indices])
                slope = pars[0]
                if slope > 0 and nmeasures == 0:
                    break
                elif slope > 0:
                    nmeasures -= 1

        rhi += width
    sky = resistent_mean(flux[i - nap + 1:i], 3)[0]        
    return sky


def ferengi_make_psf_same(psf1, psf2):
    """
    Adjust the sizes of two PSF images by zero-padding the smaller one.

    This function compares the sizes of two PSF (Point Spread Function) images
    and pads the smaller image with zeros to match the size of the larger image.

    Parameters
    ----------
    psf1 : ndarray
        First PSF image.
    psf2 : ndarray
        Second PSF image.

    Returns
    -------
    tuple of ndarray
        A tuple containing the two PSF images with matching sizes. The order
        corresponds to the input: `(psf1, psf2)` or `(psf2, psf1)` depending
        on which was smaller.

    Notes
    -----
    The padding is applied symmetrically, centering the smaller PSF within the
    larger PSF's dimensions.
    """
    if np.size(psf1) > np.size(psf2):
        case = True
        big = psf1
        small = psf2
    else:
        big = psf2
        small = psf1
        case = False

    Nb, Mb = big.shape
    Ns, Ms = small.shape
    
    center = int(np.floor(Nb / 2))
    small_side = int(np.floor(Ns / 2))
    new_small = np.zeros(big.shape)
    new_small[center-small_side:center+small_side+1, center-small_side:center+small_side+1] = small
    if case:
        return psf1, new_small
    else:
        return new_small, psf2


def barycenter(img, segmap):
    """
    Compute the barycenter of a galaxy from an image and segmentation map.

    This function calculates the barycenter (center of mass) of a galaxy
    using its intensity values in the input image and a corresponding
    segmentation map.

    Parameters
    ----------
    img : ndarray
        2D image array representing the intensity of the galaxy.
    segmap : ndarray
        2D segmentation map where the galaxy is defined. Values should be 
        binary (1 for the galaxy and 0 for the background).

    Returns
    -------
    tuple of float
        Coordinates of the barycenter `(X, Y)`.

    Notes
    -----
    The barycenter is computed as a weighted average of the pixel positions,
    where the weights are given by the intensity values in `img` masked by
    `segmap`.
    """
    N, M = img.shape
    XX, YY = np.meshgrid(range(M), range(N))
    gal = abs(img * segmap)
    Y = np.average(XX, weights=gal)
    X = np.average(YY, weights=gal)     
    return X, Y


def ferengi_psf_centre(psf):
    """
    Center the PSF image using its light barycenter.

    This function shifts the input PSF (Point Spread Function) image so that
    its barycenter aligns with the center of the image grid. If the PSF is not
    square, an error is raised.

    Parameters
    ----------
    psf : ndarray
        2D array representing the PSF image.

    Returns
    -------
    ndarray
        PSF image with the barycenter aligned to the center of the grid.

    Raises
    ------
    AssertionError
        If the input PSF is not square.

    Notes
    -----
    If warnings occur during the Gaussian fit to the PSF, the barycenter is
    approximated as the geometric center of the grid.
    """
    
    if len(psf.shape) == 3 and psf.shape[0] == 1:
        psf = psf.squeeze(axis=0)  

    N, M = psf.shape

    assert N == M, 'PSF image must be square'
    
    if N % 2 == 0:
        center_psf = np.zeros([N + 1, M + 1])
    else:
        center_psf = np.zeros([N, M])

    center_psf[0:N, 0:M] = psf
    N1, M1 = center_psf.shape

    X, Y = barycenter(center_psf, np.ones(center_psf.shape))
    G2D_model = apmodel.models.Gaussian2D(np.amax(psf), X, Y, 3, 3)

    fit_data = apmodel.fitting.LevMarLSQFitter()
    X, Y = np.meshgrid(np.arange(N1), np.arange(M1))
    with warnings.catch_warnings(record=True) as w:
        pars = fit_data(G2D_model, X, Y, center_psf)

    if len(w) == 0:
        cenY = pars.x_mean.value
        cenX = pars.y_mean.value
    else:
        for warn in w:
            print(warn)
        cenX = center_psf.shape[0] / 2 + N % 2
        cenY = center_psf.shape[1] / 2 + N % 2
    
    dx = (cenX - center_psf.shape[0] / 2)
    dy = (cenY - center_psf.shape[1] / 2)
    
    center_psf = scndi.shift(center_psf, [-dx, -dy])
    
    return center_psf
    

def ferengi_deconvolve(wide, narrow):
    """
    Compute the transformation PSF between two PSF images.

    This function calculates the transformation PSF that maps one PSF 
    (narrow-field) to another (wide-field) in the Fourier domain. The result
    can be used to simulate or analyze how one optical system relates to another.

    Parameters
    ----------
    wide : ndarray
        Wide-field PSF image.
    narrow : ndarray
        Narrow-field PSF image. Must have the same shape as `wide`.

    Returns
    -------
    ndarray
        Transformation PSF used to map the narrow-field PSF to the wide-field PSF.

    Notes
    -----
    - Both PSF images must be centered (odd pixel dimensions) and normalized.
    - The computation involves zero-padding to ensure compatibility for Fourier transforms.
    - The maximum array size for processing is capped at 2048x2048 pixels.
    """
    Nn, Mn = narrow.shape  # Assumes narrow and wide have the same shape

    smax = max(Nn, Mn) 
    bigsz = 2    
    while bigsz < smax:
        bigsz *= 2

    if bigsz > 2048:
        print('Requested PSF array is larger than 2x2k!')
    
    psf_n_2k = np.zeros([bigsz, bigsz], dtype=np.double)
    psf_w_2k = np.zeros([bigsz, bigsz], dtype=np.double)
    
    psf_n_2k[0:Nn, 0:Mn] = narrow
    psf_w_2k[0:Nn, 0:Mn] = wide
    
    psf_n_2k = psf_n_2k.astype(np.complex_)
    psf_w_2k = psf_w_2k.astype(np.complex_)
    fft_n = np.fft.fft2(psf_n_2k)
    fft_w = np.fft.fft2(psf_w_2k)
    
    fft_n = np.absolute(fft_n) / (np.absolute(fft_n) + 1e-9) * fft_n
    fft_w = np.absolute(fft_w) / (np.absolute(fft_w) + 1e-9) * fft_w
    
    psf_ratio = fft_w / fft_n

    psf_intermed = np.real(np.fft.fft2(psf_ratio))
    psf_corr = np.zeros(narrow.shape, dtype=np.double)
    lo = bigsz - Nn // 2
    hi = Nn // 2
    psf_corr[0:hi, 0:hi] = psf_intermed[lo:bigsz, lo:bigsz]    
    psf_corr[hi:Nn-1, 0:hi] = psf_intermed[0:hi, lo:bigsz]    
    psf_corr[hi:Nn-1, hi:Nn-1] = psf_intermed[0:hi, 0:hi]    
    psf_corr[0:hi, hi:Nn-1] = psf_intermed[lo:bigsz, 0:hi]        
    
    psf_corr = np.rot90(psf_corr, 2)
    return psf_corr / np.sum(psf_corr)


def ferengi_clip_edge(image, auto_frac=2, clip_also=None, norm=False):
    """
    Dynamically clip the edges of an image based on standard deviation analysis.

    This function removes noisy or irrelevant border pixels by analyzing the 
    standard deviation of pixel values in concentric rings centered on the image. 
    Optionally, it can normalize the clipped image and apply the same clipping 
    to a secondary array.

    Parameters
    ----------
    image : ndarray
        Input 2D image array.
    auto_frac : int, optional
        Initial fraction to define the size of concentric rings for analysis.
        Default is 2.
    clip_also : ndarray, optional
        Secondary array to apply the same clipping. Must have the same shape as `image`.
        Default is None.
    norm : bool, optional
        If True, normalizes the clipped image and `clip_also` (if provided) by 
        dividing each by its sum. Default is False.

    Returns
    -------
    int
        Number of pixels clipped from each side of the image.
    ndarray
        Clipped and optionally normalized image.
    ndarray, optional
        Clipped and optionally normalized secondary array (`clip_also`), if provided.

    Notes
    -----
    - The clipping is based on detecting significant deviations (≥10 standard deviations) 
      in the pixel values of the concentric rings.
    - If the number of significant deviations exceeds three times the rejected outliers,
      a warning is printed ("Large gap?").
    - The function assumes the input image is a 2D array.

    Raises
    ------
    ValueError
        If `clip_also` does not have the same shape as `image`.
    """
    N, M = image.shape
    rx = int(N / 2 / auto_frac)
    ry = int(M / 2 / auto_frac)
    
    sig = np.array([])
    r = np.array([])
    while True:
        idx = edge_index(image, rx, ry)
        if np.size(idx[0]) == 0:
            break
        med, sigma, nrej = resistent_mean(image, 3)
        sigma *= np.sqrt(np.size(image) - 1 - nrej)
        sig = np.append(sig, sigma)
        r = np.append(r, rx)
        rx += 1
        ry += 1
    
    new_med, new_sig, new_nrej = resistent_mean(sig, 3)
    new_sig *= np.sqrt(np.size(sig) - 1 - new_nrej)
    
    i = np.where(sig >= new_med * 10 * new_sig)
    if np.size(i) > 0:
        lim = np.min(r[i])
        if np.size(i) > new_nrej * 3:
            print('Large gap?')
        npix = round(N / 2.0 - lim)
        
        if clip_also is not None:
            clip_also = clip_also[npix:N-1-npix, npix:M-1-npix]
        image = image[npix:N-1-npix, npix:M-1-npix]
    
    if norm:
        image /= np.sum(image)
        if clip_also is not None:
            clip_also /= np.sum(clip_also)
    
    if clip_also is not None:
        return npix, image, clip_also
    else:
        return npix, image
    

def rebin2d(img, Nout, Mout, flux_scale=False):
    """
    Rebin a 2D image to a new shape with non-integer magnification.

    This function reduces or enlarges a 2D image to the specified output shape
    `(Nout, Mout)`, preserving flux or pixel values depending on the `flux_scale`
    parameter. The implementation is based on the FREBIN function from IDL Astrolib.

    Parameters
    ----------
    img : ndarray
        Input 2D array representing the image.
    Nout : int
        Number of rows in the output array.
    Mout : int
        Number of columns in the output array.
    flux_scale : bool, optional
        If True, scales pixel values to conserve total flux.
        If False, normalizes pixel values by the scaling factor. Default is False.

    Returns
    -------
    ndarray
        Rebinned 2D array with shape `(Nout, Mout)`.

    Raises
    ------
    ValueError
        If the input image shape is incompatible with the specified output shape.

    Warnings
    --------
    RuntimeWarning
        May be raised during calculations for empty slices or divisions by zero.
    """
    N, M = img.shape

    xbox = N / float(Nout)
    ybox = M / float(Mout)

    temp_y = np.zeros([N, Mout])

    for i in range(Mout):
        rstart = i * ybox
        istart = int(rstart)

        rstop = rstart + ybox
        if int(rstop) > M - 1:
            istop = M - 1
        else:
            istop = int(rstop)

        frac1 = rstart - istart
        frac2 = 1.0 - (rstop - istop)
        if istart == istop:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                temp_y[:, i] = (1.0 - frac1 - frac2) * img[:, istart]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                temp_y[:, i] = (
                    np.sum(img[:, istart:istop + 1], 1)
                    - frac1 * img[:, istart]
                    - frac2 * img[:, istop]
                )

    temp_y = temp_y.transpose()
    img_bin = np.zeros([Mout, Nout])

    for i in range(Nout):
        rstart = i * xbox
        istart = int(rstart)

        rstop = rstart + xbox
        if int(rstop) > N - 1:
            istop = N - 1
        else:
            istop = int(rstop)

        frac1 = rstart - istart
        frac2 = 1.0 - (rstop - istop)

        if istart == istop:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                img_bin[:, i] = (1.0 - frac1 - frac2) * temp_y[:, istart]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                img_bin[:, i] = (
                    np.sum(temp_y[:, istart:istop + 1], 1)
                    - frac1 * temp_y[:, istart]
                    - frac2 * temp_y[:, istop]
                )

    if flux_scale:
        return img_bin.transpose()
    else:
        return img_bin.transpose() / (xbox * ybox)
        
        
def lum_evolution(zlow, zhigh):
    """
    Calculate the luminosity evolution factor between two redshifts.

    This function computes the ratio of luminosities at two redshifts (`zhigh` and `zlow`)
    using the luminosity evolution model from Sobral et al. (2013).

    Parameters
    ----------
    zlow : float
        Lower redshift value.
    zhigh : float
        Higher redshift value.

    Returns
    -------
    float
        The luminosity evolution factor.
    """
    def luminosity(z):
        logL = 0.45 * z + 41.87
        return 10 ** logL
    return luminosity(zhigh) / luminosity(zlow)


def ferengi_downscale(image_low, z_low, z_high, pix_low, pix_hi, upscale=False, nofluxscale=False, evo=None):
    """
    Downscale an image to simulate a higher redshift.

    This function adjusts the size, flux, and luminosity of an input image
    to simulate how it would appear at a higher redshift. The scaling factors
    are calculated based on angular and luminosity distances, pixel scales,
    and optional evolutionary corrections.

    Parameters
    ----------
    image_low : ndarray
        Input 2D image at the lower redshift.
    z_low : float
        Redshift of the input image.
    z_high : float
        Target redshift for the simulation.
    pix_low : float
        Pixel scale of the input image (arcsec/pixel) at the lower redshift.
    pix_hi : float
        Pixel scale of the output image (arcsec/pixel) at the higher redshift.
    upscale : bool, optional
        If True, inverts the scaling to simulate an image at a lower redshift.
        Default is False.
    nofluxscale : bool, optional
        If True, skips flux scaling. Default is False.
    evo : float, optional
        Evolutionary correction factor to adjust for luminosity changes with redshift.
        If provided, the luminosity scaling is computed using:
            `evo_fact = 10 ** (-0.4 * evo * z_high)`.
        If None, the Sobral et al. (2013) luminosity evolution model is used.

    Returns
    -------
    ndarray
        The adjusted 2D image at the target redshift.

    Notes
    -----
    - If `nofluxscale` is True, the flux scaling is skipped, and the image
      retains its original flux.
    - Uses the `rebin2d` function to scale the image dimensions while conserving flux.

    Raises
    ------
    ValueError
        If the input image shape is invalid or scaling factors result in unrealistic dimensions.
    """
    da_in = cosmos.angular_distance(z_low)
    da_out = cosmos.angular_distance(z_high)

    dl_in = cosmos.luminosity_distance(z_low)
    dl_out = cosmos.luminosity_distance(z_high)

    if evo is not None:
        evo_fact = 10 ** (-0.4 * evo * z_high) # UPDATED TO MATCH FERENGI ALGORITHM
    else:
        evo_fact = lum_evolution(z_low, z_high)

    mag_factor = (da_in / da_out) * (pix_low / pix_hi)
    if upscale:
        mag_factor = 1.0 / mag_factor

    #  lum_factor = (dl_in/dl_out)**2
    lum_factor = (dl_in / dl_out) ** 2 * (1.0 + z_high) / (1.0 + z_low) ### UPDATED TO MATCH FERENGI ALGORITHM

    if nofluxscale:
        lum_factor = 1.0
    else:
        lum_factor = (da_in / da_out) ** 2

    N, M = image_low.shape
    N_out = int(round(N * mag_factor))
    M_out = int(round(M * mag_factor))

    img_out = rebin2d(image_low, N_out, M_out, flux_scale=True) * lum_factor * evo_fact

    return img_out


def ferengi_odd_n_square():
    #TBD : in principle avoidable if PSF already square image
    #feregi_psf_centre already includes number of odd pixels
    raise NotImplementedError('In principle avoidable if PSF already square image')
    return


def ferengi_transformation_psf(psf_low, psf_high, z_low, z_high, pix_low, pix_high, same_size=None):
    """
    Compute the transformation PSF between two redshifts.

    This function calculates the transformation PSF that maps a low-redshift
    PSF (`psf_low`) to a high-redshift PSF (`psf_high`). It uses the angular
    distances, pixel scales, and redshift values to perform the necessary adjustments.

    Parameters
    ----------
    psf_low : ndarray
        PSF at the lower redshift.
    psf_high : ndarray
        PSF at the higher redshift.
    z_low : float
        Redshift of the low-redshift PSF.
    z_high : float
        Redshift of the high-redshift PSF.
    pix_low : float
        Pixel scale of the low-redshift PSF (arcsec/pixel).
    pix_high : float
        Pixel scale of the high-redshift PSF (arcsec/pixel).
    same_size : bool, optional
        If True, ensures that the output PSFs have the same dimensions. Default is None.

    Returns
    -------
    tuple of ndarray
        - The adjusted low-redshift PSF.
        - The adjusted high-redshift PSF.
        - The transformation PSF mapping `psf_low` to `psf_high`.

    Notes
    -----
    - The function ensures that the PSFs are centered and normalized.
    - If the output PSF dimensions are even, additional padding is applied to ensure odd dimensions.
    - The transformation PSF is calculated using the `ferengi_deconvolve` function.

    Raises
    ------
    ValueError
        If the PSF adjustment exceeds three times the original PSF size.
    """
    # Center the input PSFs to ensure proper alignment
    psf_l = ferengi_psf_centre(psf_low)
    psf_h = ferengi_psf_centre(psf_high)

    # Calculate angular distances for redshift scaling
    da_in = cosmos.angular_distance(z_low)
    da_out = cosmos.angular_distance(z_high)

    # Initialize dimensions and padding for even-sized PSFs
    N, M = psf_l.shape
    add = 0
    out_size = round((da_in / da_out) * (pix_low / pix_high) * (N + add))
    
    # Ensure the output PSF has odd dimensions by adding padding as needed
    while out_size % 2 == 0:
        add += 2
        psf_l = np.pad(psf_l, 1, mode='constant')
        out_size = round((da_in / da_out) * (pix_low / pix_high) * (N + add))
        if add > N * 3:
            raise ValueError("PSF adjustment exceeded acceptable limits.")

    # Downscale the low-redshift PSF to match the higher redshift
    psf_l = ferengi_downscale(psf_l, z_low, z_high, pix_low, pix_high, nofluxscale=True)
    psf_l = ferengi_psf_centre(psf_l)

    # Ensure both PSFs have the same dimensions
    psf_l, psf_h = ferengi_make_psf_same(psf_l, psf_h)

    # Center and normalize the PSFs
    psf_l = ferengi_psf_centre(psf_l)
    psf_h = ferengi_psf_centre(psf_h)

    psf_l /= np.sum(psf_l)
    psf_h /= np.sum(psf_h)

    # Compute the transformation PSF
    transformation_psf = ferengi_psf_centre(ferengi_deconvolve(psf_h, psf_l))

    return psf_l, psf_h, transformation_psf

def ferengi_convolve_plus_noise(image, psf, sky, exptime, nonoise=False, border_clip=None, extend=False):
    """
    Convolve an image with a PSF and optionally add noise.

    This function convolves the input image with a Point Spread Function (PSF)
    in the Fourier domain. Optionally, it adds noise based on the provided
    sky background and exposure time.

    Parameters
    ----------
    image : ndarray
        Input 2D array representing the image.
    psf : ndarray
        Point Spread Function (PSF) to convolve with the image.
    sky : ndarray
        2D array representing the sky background.
    exptime : float
        Exposure time used to calculate noise levels.
    nonoise : bool, optional
        If True, skips adding noise. Default is False.
    border_clip : int, optional
        Number of pixels to clip from the borders of the PSF. Default is None.
    extend : bool, optional
        If True, keeps the extended image dimensions after convolution.
        If False, crops the image to its original size. Default is False.

    Returns
    -------
    ndarray
        Convolved image, optionally with added noise.

    Raises
    ------
    ValueError
        If the dimensions of the sky background are insufficient for the output image.
    """
    # Clip PSF borders if specified
    if border_clip is not None: 
        Npsf, Mpsf = psf.shape
        psf = psf[border_clip:Npsf-border_clip, border_clip:Mpsf-border_clip]
    
    # Get the new PSF dimensions after clipping
    Npsf, Mpsf = psf.shape
    Nimg, Mimg = image.shape

    # Pad the input image to match the PSF size for convolution
    out = np.pad(image, Npsf, mode='constant')
    
    # Perform convolution in the Fourier domain with a normalized PSF
    out = apcon.convolve_fft(out, psf / np.sum(psf))
    
    # Adjust the output image dimensions
    Nout, Mout = out.shape
    if not extend:
        out = out[Npsf:Nout-Npsf, Mpsf:Mout-Mpsf]
        
    # Update output dimensions after cropping
    Nout, Mout = out.shape
    Nsky, Msky = sky.shape

    if not nonoise:
        # Calculate effective noise padding
        ef = Nout % 2
        try:
            # Add noise based on the sky background and exposure time
            out += (
                sky[Nsky//2 - Nout//2:Nsky//2 + Nout//2 + ef, Msky//2 - Mout//2:Msky//2 + Mout//2 + ef]
                + np.sqrt(np.abs(out * exptime)) * npr.normal(size=out.shape) / exptime
            )
        except ValueError:
            # Return an error matrix if the sky background dimensions are insufficient
            return -99 * np.ones(out.shape)
    
    return out


def dump_results(image, psf, imgname_in, bgimage_in, names_out, lowz_info, highz_info):
    """
    Save processed image and PSF to FITS files with metadata.

    This function writes the processed image and its associated PSF to separate
    FITS files. The header of the image file is populated with metadata about
    the input parameters, including low- and high-redshift information.

    Parameters
    ----------
    image : ndarray
        Processed 2D image to save.
    psf : ndarray
        PSF associated with the processed image.
    imgname_in : str or list of str
        Input image file name(s).
    bgimage_in : str or list of str
        Background image file name(s).
    names_out : tuple of str
        Output file names for the image and PSF, in the format `(image_out, psf_out)`.
    lowz_info : dict
        Dictionary of parameters for the low-redshift input, including PSF, pixel scale,
        and other related information.
    highz_info : dict
        Dictionary of parameters for the high-redshift input, including PSF, pixel scale,
        and other related information.

    Returns
    -------
    None
    """
    name_imout, name_psfout = names_out
    
    Hprim = pyfits.PrimaryHDU(data=image)
    hdu = pyfits.HDUList([Hprim])
    hdr_img = hdu[0].header
    
    # Utility function to add multi-entry parameters as separate numbered columns
    def add_multi_entry(header, base_key, values, description, enumerate_single=False):
        if isinstance(values, list):
            if len(values) == 1 and not enumerate_single:  # No enumeration if only one element and enumerate_single is False
                header[base_key] = (str(values[0]), description)
            else:
                for i, value in enumerate(values, 1):
                    # Shorten base_key to 6 characters to keep within 8-character limit when adding "_i" suffix
                    header[f"{base_key[:6]}_{i}"] = (str(value), f"{description} {i}")
        else:
            header[base_key] = (str(values), description)

    # Convert input image names and background image names to filenames only
    imgname_list = [os.path.basename(str(path)) for path in imgname_in] if isinstance(imgname_in, list) else [os.path.basename(str(imgname_in))]
    bgimage_list = [os.path.basename(str(path)) for path in bgimage_in] if isinstance(bgimage_in, list) else [os.path.basename(str(bgimage_in))]

    # Add each entry as separate columns
    add_multi_entry(hdr_img, 'INPUT', imgname_list, 'Input image')
    add_multi_entry(hdr_img, 'SKYIM', bgimage_list, 'Background image')  # Changed 'SKY_IMG' to 'SKYIM'

    # Process low-z information with multi-column handling and enumeration for all lists
    for key, value in lowz_info.items():
        base_key = f"{key[:4].upper()}_I"
        if key.lower() == "psf":
            value_list = [os.path.basename(str(v)) for v in value] if isinstance(value, list) else [os.path.basename(str(value))]
        else:
            value_list = value if isinstance(value, list) else [value]
        add_multi_entry(hdr_img, base_key, value_list, f"{key} input lowz", enumerate_single=True)

    hdr_img['comment'] = f'Using ferengi.py version {version}'

    # Process high-z information, avoiding enumeration if there’s a single value in lists
    for key, value in highz_info.items():
        base_key = f"{key[:4].upper()}_O"
        if key.lower() == "psf":
            value_list = [os.path.basename(str(v)) for v in value] if isinstance(value, list) else [os.path.basename(str(value))]
        else:
            value_list = value if isinstance(value, list) else [value]
        # For high-z info, set enumerate_single=False to avoid suffix if there's a single value
        add_multi_entry(hdr_img, base_key, value_list, f"{key} input highz", enumerate_single=False)
    
    # Write the image and PSF to files
    hdu.writeto(name_imout, overwrite=True)
    pyfits.writeto(name_psfout, psf, overwrite=True)
    return


def kcorrect_maggies(image, im_err, lowz_info, highz_info, lambda_lo, lambda_hi, err0_mag=None, evo=None, noflux=None, kc_obj=None):
    """
    Perform K-correction on multi-band images.

    This function applies K-corrections to images observed in multiple bands,
    adjusting fluxes and errors to account for redshift differences and instrumental
    characteristics.

    Parameters
    ----------
    image : list of ndarray
        List of 2D images for each observed band.
    im_err : list of ndarray
        List of 2D error maps corresponding to `image`.
    lowz_info : dict
        Dictionary containing information about the low-redshift input (e.g., redshift,
        pixel scale, exposure time, zero-point magnitudes).
    highz_info : dict
        Dictionary containing information about the high-redshift input.
    lambda_lo : array-like
        Central wavelengths of the filters at low redshift.
    lambda_hi : array-like
        Central wavelengths of the filters at high redshift.
    err0_mag : float, optional
        Minimum magnitude error for all bands. Default is None.
    evo : float, optional
        Evolutionary correction factor. Default is None.
    noflux : bool, optional
        If True, skips flux scaling. Default is None.
    kc_obj : object, optional
        Pre-initialized K-correction object. Default is None.

    Returns
    -------
    tuple
        - Corrected image(s) adjusted for redshift differences.
        - Background image adjusted for redshift.
    """
   # Number of bands in the input data
    n_bands = len(image)

    # Calculate the relative difference in filter wavelengths to find the closest match
    dz = np.abs(lambda_hi / lambda_lo - 1)
    idx_bestfilt = np.argmin(dz)

    # Determine weights for the closest filters in rest-frame
    dz1 = np.abs(lambda_hi - lambda_lo)
    ord = np.argsort(dz1)  # Indices of filters sorted by proximity
    weight = np.ones(n_bands)  # Default weight for all bands

    # Assign weights based on the number of bands and proximity of filters
    if dz1[ord[0]] == 0:
        if n_bands == 2:
            weight[ord] = [10, 4]
        elif n_bands == 3:
            weight[ord] = [10, 4, 4]
        elif n_bands >= 4:
            weight[ord] = [10, 4, 4] + [1] * (n_bands - 3)
    else:
        if n_bands == 2:
            weight[ord] = [10, 8]
        elif n_bands == 3 or n_bands == 4:
            weight[ord] = [10, 8] + [4] * (n_bands - 2)
        elif n_bands > 4:
            weight[ord] = [10, 8, 4, 4] + [1] * (n_bands - 4)

    # Lists to store downscaled images and error maps
    img_downscale , imerr_downscale = [], []

    for i in range(n_bands):
        # Downscale the image and error map to match the high redshift
        img_downscale.append(ferengi_downscale(image[i], lowz_info['redshift'], highz_info['redshift'], lowz_info['pixscale'], highz_info['pixscale'], evo=evo, nofluxscale=noflux))
        img_downscale[i] -= ring_sky(img_downscale[i], 50, 15, nw=True)  # Subtract sky background

        imerr_downscale.append(ferengi_downscale(im_err[i], lowz_info['redshift'], highz_info['redshift'], lowz_info['pixscale'], highz_info['pixscale'], evo=evo, nofluxscale=noflux))

        # Convert errors from counts to magnitudes
        imerr_downscale[i] = 2.5 / np.log(10) * imerr_downscale[i] / img_downscale[i]

        # Convert flux from counts to maggies
        img_downscale[i] = cts2maggies(img_downscale[i], lowz_info['exptime'][i], lowz_info['zp'][i])

    # Define the sigma threshold for K-correction
    siglim = 2
    sig = np.zeros(n_bands)  # Array to store sigma values for each band
    npix = np.size(img_downscale[0])  # Number of pixels in a single band

    # Find the filter closest to the high redshift filter
    zmin = np.abs(lambda_hi / lambda_lo - 1 - highz_info['redshift'])
    filt_i = np.argmin(zmin)

    # Initialize sigma maps for each band and count of high-sigma pixels
    nsig = np.zeros_like(img_downscale)
    nhi = np.zeros_like(img_downscale[0])

    # Create sigma maps and identify high-sigma pixels
    for i in range(n_bands):
        m, s, n = resistent_mean(img_downscale[i], 3)  # Robust mean and sigma
        sig[i] = s * np.sqrt(npix - 1 - n)  # Adjusted sigma for the band
        nsig[i] = scndi.median_filter(img_downscale[i], size=3) / sig[i]  # Sigma map
        hi = np.where(np.abs(nsig[i]) > siglim)  # High-sigma pixels
        if hi[0].size > 0:
            nhi[hi] += 1  # Increment count for high-sigma pixels

    # Select "good" pixels from the closest filter based on sigma range
    good1 = np.where((np.abs(nsig[filt_i]) > 0.25) & (np.abs(nsig[filt_i]) <= siglim))

    # Select 50% of pixels in the range 0.25 < sigma < siglim
    if good1[0].size > 0:
        n_selec = round(good1[0].size * 0.5)
        good1_indices = np.random.choice(good1[0].size, size=n_selec, replace=False)
        good1 = (good1[0][good1_indices], good1[1][good1_indices])

    # Select pixels based on high-sigma count across all bands
    good = np.where((nhi >= 3) & (np.abs(nsig[filt_i]) > siglim))
    if good[0].size == 0:
        print('No filters have 3+ high sigma pixels')
        good = np.where((nhi >= 2) & (np.abs(nsig[filt_i]) > siglim))
    if good[0].size == 0:
        print('No filters have 2+ high sigma pixels')
        good = np.where((nhi >= 1) & (np.abs(nsig[filt_i]) > siglim))

    # Combine indices of "good" pixels and remove duplicates
    if good1[0].size > 0:
        good = (np.concatenate((good[0], good1[0])), np.concatenate((good[1], good1[1])))
        combined_indices = np.vstack((good[0], good[1])).T
        unique_indices = np.unique(combined_indices, axis=0)
        good = (unique_indices[:, 0], unique_indices[:, 1])

    # Process "good" pixels if any are identified
    ngood = good[0].size
    if ngood == 0:
        print('No pixels to process')
    else:
        print(f'{ngood} pixels to process')

        maggies = []  # Flux in maggies for K-correction
        err = []  # Error associated with each pixel
        nsig_2d = []  # Sigma values for selected pixels

        # Extract values for K-correction
        for i in range(ngood):
            aux_maggies, aux_err, aux_nsig = [], [], []
            
            for j in range(n_bands):
                aux_maggies.append(img_downscale[j][good[0][i]][good[1][i]])
                aux_err.append(imerr_downscale[j][good[0][i]][good[1][i]])
                aux_nsig.append(nsig[j][good[0][i]][good[1][i]])
            maggies.append(np.array(aux_maggies))
            nsig_2d.append(np.array(aux_nsig))
            err.append(np.array(aux_err))

        # Handle infinite errors by setting them to a large value
        for i in range(ngood):
            inf = np.where(~np.isfinite(err[i]))
            if inf[0].size > 0:
                err[i][inf] = 99999
            err[i] = np.abs(err[i])
            err[i] = np.where(err[i] < 99999, err[i], 99999)

        # Create arrays for minimum errors and weights
        err0 = np.tile(err0_mag, (ngood, 1))
        wei = np.tile(weight, (ngood, 1))

        # Combine minimum errors with calculated errors
        err = np.array(err)
        err = np.sqrt(err0**2 + err**2) / wei

        # Convert errors to inverse variance
        ivar = (2.5 / np.log(10) / err / maggies)**2
        inf = np.where(~np.isfinite(ivar))
        if inf[0].size > 0:
            ivar[inf] = np.max(ivar[np.isfinite(ivar)])

        # Setup K-correction object
        responses_lo = lowz_info['filter']
        responses_hi = highz_info['filter']
        redshift_lo = lowz_info['redshift'] * np.ones(ngood)
        redshift_hi = highz_info['redshift'] * np.ones(ngood)

        if kc_obj is None:
            print("Creating kcorrect object...")
            cos = FlatLambdaCDM(H0=cosmos.H0, Om0=cosmos.Omat, Ob0=cosmos.Obar)
            kc = kcorrect.kcorrect.Kcorrect(responses=responses_lo, responses_out=[responses_hi], responses_map=[responses_lo[idx_bestfilt]], cosmo=cos)
        else:
            kc = kc_obj

        # Perform K-correction calculations
        coeffs = kc.fit_coeffs(redshift=redshift_lo, maggies=maggies, ivar=ivar)
        k_values = kc.kcorrect(redshift=redshift_lo, coeffs=coeffs)

        # Reconstruct magnitudes at the high redshift
        r_maggies = kc.reconstruct_out(redshift=redshift_hi, coeffs=coeffs)

    # Scale the background using the high redshift factor
    bg = img_downscale[filt_i] / (1.0 + highz_info['redshift'])
    img_downscale = bg

    # Apply K-corrections to selected pixels
    if isinstance(good, tuple) and len(good) == 2 and isinstance(good[0], np.ndarray) and isinstance(good[1], np.ndarray):
        for i in range(ngood):
            img_downscale[good[0][i]][good[1][i]] = r_maggies[i] / (1.0 + highz_info['redshift'])

    # Convert back to counts
    img_downscale = maggies2cts(img_downscale, highz_info['exptime'], highz_info['zp'])
    bg = maggies2cts(bg, highz_info['exptime'], highz_info['zp'])

    return img_downscale, bg


def filer_list():
    """
    Retrieve the list of available filters in the kcorrect package.

    This function returns a list of all the filters available in the
    `kcorrect` package. These filters are used for K-correction calculations
    and are pre-defined within the package.

    Returns
    -------
    list
        A list of filter names available in the `kcorrect` package.
    """
    return kcorrect.response.all_responses()


def ferengi(images, background, lowz_info, highz_info, namesout, imerr=None, err0_mag=None, kc_obj=None, noflux=False, evo=None, minimum_output_size=None, noconv=False, extend=False, nonoise=False, check_psf_FWHM=False, border_clip=3):
    """
    Simulate galaxy redshifting with instrumental and cosmological effects.

    Simulates how galaxies would appear at higher redshifts, accounting for angular 
    resolution degradation, surface brightness dimming, and wavelength shifts. The process 
    integrates PSFs for both redshifts, photometric band adjustments, and optional evolutionary 
    corrections to produce realistic observational conditions at the target redshift.

    Parameters
    ----------
    images : list of str
        List of paths to the input images, one for each band. All files must be in FITS format and in units of [counts].
    background : list of str
        List of paths to the background sky images, one for each band. All files must be in FITS format and in units of [counts/sec].
    lowz_info : dict
        Dictionary containing information about the low-redshift input. Keys include:
        - 'redshift': float, redshift of the input images.
        - 'psf': list of str, paths to the PSFs for each band at low redshift (FITS format, normalized total flux to 1).
        - 'zp': list of float, zero-point magnitudes for each band [magnitudes].
        - 'exptime': list of float, exposure times for each band [seconds].
        - 'filter': list of str, names of the input filters for each band. Filters must exist in the `kcorrect` catalog.
        - 'pixscale': float, pixel scale of the images [arcseconds per pixel].
        - 'lambda': list of float, central wavelengths of the filters [angstroms].
    highz_info : dict
        Dictionary containing information about the high-redshift simulation. Keys include:
        - 'redshift': float, target redshift.
        - 'psf': str, path to the PSF file for the simulation (FITS format, normalized total flux to 1).
        - 'zp': float, zero-point magnitude [magnitudes].
        - 'exptime': float, exposure time [seconds].
        - 'filter': str, name of the output filter. Must exist in the `kcorrect` catalog.
        - 'pixscale': float, pixel scale of the simulation [arcseconds per pixel].
        - 'lambda': float, central wavelength of the output filter [angstroms].
    namesout : list of str
        Paths to the output files: [output_sci_image, output_psf_image]. The output will be in FITS format.
    imerr : list of str, optional
        List of paths to error maps for the input images (FITS format, in units of [counts]). Default is None, in which case
        error maps are generated from the square root of the input images.
    err0_mag : float, optional
        Minimum magnitude error for all bands [magnitudes]. Default is None.
    kc_obj : object, optional
        Pre-initialized K-correction object. If not provided, it will be created automatically.
    noflux : bool, optional
        If True, skips flux scaling. Default is False.
    evo : float, optional
        Evolutionary correction factor. Two models are used:
        - If `evo` is specified, it applies the equation:
          `evo_fact = 10 ** (-0.4 * evo * z_high)`
        - If `evo` is None, it uses the Sobral et al. (2013) luminosity evolution model.
        Default is None.
    minimum_output_size : int, optional
        Minimum size (in pixels) for the output redshifted image. Default is None.
    noconv : bool, optional
        If True, skips convolution with the transformation PSF. Default is False.
    extend : bool, optional
        If True, keeps extended image dimensions after convolution. Default is False.
    nonoise : bool, optional
        If True, skips adding noise to the output image. Default is False.
    check_psf_FWHM : bool, optional
        If True, ensures the PSF at high redshift is broader than the low-redshift PSF.
        If the condition is not met, the simulation is not performed. Default is False.
    border_clip : int, optional
        Number of pixels to clip from the PSF borders before convolution. Default is 3.

    Returns
    -------
    tuple
        - ndarray: The redshifted and convolved image.
        - ndarray: The reconstructed PSF.
        - int: Status code for the operation (-99 if an error occurs).

    Raises
    ------
    ValueError
        If the input lists (images, background, or PSFs) have inconsistent lengths.
        If the input images have inconsistent shapes.
    ZeroDivisionError
        If an input image contains non-positive values, leading to division by zero during error calculation.
    RuntimeError
        If an unexpected error occurs during the simulation process.
    TypeError
        If the PSF enlargement process fails during convolution.

    Notes
    -----
    - Filters specified in the `lowz_info` and `highz_info` dictionaries must be part of the catalog in the `kcorrect` package.
      Use the `filter_list` function from `dopterian` to view the available filters.
    - If the parameter `kc_obj` is not provided, dopterian will automatically create the K-correction object.
      However, for simulations involving large sets of galaxies using the same input and output filters,
      it is recommended to provide the K-correction object as a parameter to save computation time.
    - At least two input filters are required to apply the K-correction. If only one filter is provided, the correction is skipped.

    """
     # Determine the number of input bands
    n_bands = len(images)

    # Determine whether to apply K-correction based on the number of bands
    apply_kcorrect = True if n_bands > 1 else False

    # Initialize lists for PSFs, sky backgrounds, images, and error maps

    Pl, Ph, sky, image, im_err = [], [], [], [], []

    # Convert the wavelength information to NumPy arrays for calculations
    if lowz_info['lambda'] is not None:
        lambda_lo = np.array(lowz_info['lambda'])
    if highz_info['lambda'] is not None:
        lambda_hi = np.array(highz_info['lambda'])

    # Load input images, PSFs, and background files
    for i in range(n_bands):
        image.append(pyfits.getdata(images[i]))
        Pl.append(pyfits.getdata(lowz_info['psf'][i]))
        sky.append(pyfits.getdata(background[i]))

    # Validate the consistency of input list lengths
    lengths = [len(images), len(background), len(lowz_info['psf'])]
    if not all(length == lengths[0] for length in lengths):
        print('All input lists must have the same number of entries')
        return -99, -99

    # Ensure all input images have the same shape
    shapes = [banda.shape for banda in image]
    if len(set(shapes)) != 1:
        print("Error: All images must have the same shape")
        return -99, -99

    # Validate the size of the redshifted output image
    if minimum_output_size is not None and predict_redshifted_image_size(lowz_info['redshift'], highz_info['redshift'], image[0].shape[0], lowz_info['pixscale'], highz_info['pixscale']) < minimum_output_size:
        print("The size of the redshifted image is smaller than the minimum size. The simulation is not possible")
        return -99, -99

    # Load the high-redshift PSF
    Ph.append(pyfits.getdata(highz_info['psf']))

    # Create error maps if they are not provided
    if imerr is None:
        for i in range(n_bands):
            im_err.append(1 / np.sqrt(np.abs(image[i])))  # Poisson noise
    else:
        for i in range(n_bands):
            im_err.append(pyfits.getdata(imerr[i]))

    # Determine scaling for images based on K-correction
    if apply_kcorrect == False:  # K-correction is not applied
        img_nok = maggies2cts(cts2maggies(image[0], lowz_info['exptime'], lowz_info['zp'][0]), highz_info['exptime'], highz_info['zp'])
        psf_lo = Pl[0]
        psf_hi = Ph[0]
    else:
        # Select the best matching PSF for output redshift
        dz = np.abs(lambda_hi / lambda_lo - 1)
        idx_bestfilt = np.argmin(dz)
        psf_lo = Pl[idx_bestfilt]
        psf_hi = Ph[0]

    # Verify that the high-redshift PSF is broader than the low-redshift PSF
    if check_psf_FWHM == True and check_PSF_FWHM(psf_lo, psf_hi, lowz_info['pixscale'], highz_info['pixscale'], lowz_info['redshift'], highz_info['redshift']) == False:
        print("The FWHM of the redshifted PSF is broader than that of the high-z PSF. The simulation is not possible")
        return -99, -99

    # Apply scaling and downscaling based on K-correction status
    if apply_kcorrect == False:  # Without K-correction
        img_downscale = ferengi_downscale(img_nok, lowz_info['redshift'], highz_info['redshift'], lowz_info['pixscale'], highz_info['pixscale'], evo=evo, nofluxscale=noflux)
    else:  # With K-correction
        img_downscale, bg = kcorrect_maggies(image, im_err, lowz_info, highz_info, lambda_lo, lambda_hi, err0_mag, evo, noflux, kc_obj)

    # Replace invalid or zero-value pixels with median values
    med = scndi.median_filter(img_downscale, size=3)
    idx = np.where(~np.isfinite(img_downscale))
    if idx[0].size > 0:
        img_downscale[idx] = med[idx]

    idx = np.where(img_downscale == 0)
    if idx[0].size > 0:
        img_downscale[idx] = med[idx]

    # Handle high-sigma pixels if K-correction is applied
    if apply_kcorrect == True:
        m, sig, nrej = resistent_mean(img_downscale, 3)
        sig = sig * np.sqrt(np.size(img_downscale) - 1 - nrej)
        idx = np.where((np.abs(img_downscale) > 10 * sig) & (img_downscale != bg))
        if idx[0].size > 0:
            print('High sigma pixels detected')
            fit = robust_linefit(np.abs(bg[idx]), np.abs(img_downscale[idx]))
            delta = np.abs(img_downscale[idx]) - (fit[0] + fit[1] * np.abs(bg[idx]))

            m, sig, nrej = resistent_mean(delta, 3)
            sig *= np.sqrt(img_downscale.size - 1 - nrej)

            idx1 = np.where(delta / sig > 50)
            if idx1[0].size > 0:
                img_downscale[idx[0][idx1]] = med[idx[0][idx1]]

    # Subtract sky background from the image
    img_downscale -= ring_sky(img_downscale, 50, 15, nw=True)

    # Output results without convolution if requested
    if noconv == True:
        dump_results(img_downscale / highz_info['exptime'], psf_lo / np.sum(psf_lo), images, background, namesout, lowz_info, highz_info)
        return img_downscale / highz_info['exptime'], psf_lo / np.sum(psf_lo)

    # Calculate the transformation PSF
    try:
        psf_low, psf_high, psf_t = ferengi_transformation_psf(psf_lo, psf_hi, lowz_info['redshift'], highz_info['redshift'], lowz_info['pixscale'], highz_info['pixscale'])
    except TypeError as err:
        print('Enlarging PSF failed! Skipping Galaxy.')
        return -99, -99

    # Reconstruct the PSF using convolution
    try:
        if psf_lo.ndim > 2 or psf_t.ndim > 2:
            psf_lo = np.squeeze(psf_lo)
            psf_t = np.squeeze(psf_t)
        recon_psf = ferengi_psf_centre(apcon.convolve_fft(psf_lo, psf_t))
    except ZeroDivisionError as err:
        print('Reconstruction PSF failed!')
        return -99, -99

    # Normalize the reconstructed PSF
    recon_psf /= np.sum(recon_psf)

    # Convolve the image with the transformation PSF and add noise if applicable
    if n_bands == 1:
        img_downscale = ferengi_convolve_plus_noise(img_downscale / highz_info['exptime'], psf_t, sky[0], highz_info['exptime'], nonoise=nonoise, border_clip=border_clip, extend=extend)
    else:
        img_downscale = ferengi_convolve_plus_noise(img_downscale / highz_info['exptime'], psf_t, sky[idx_bestfilt], highz_info['exptime'], nonoise=nonoise, border_clip=border_clip, extend=extend)

    # Validate the convolved image
    if np.amax(img_downscale) == -99:
        print('Sky Image not big enough!')
        return -99, -99

    # Save the results to output files
    dump_results(img_downscale, recon_psf, images, background, namesout, lowz_info, highz_info)

    return img_downscale, recon_psf
            
