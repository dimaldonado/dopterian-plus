from dopterianPlus import dopterian
import importlib.resources as pkg_resources

# Base path for accessing packaged data files
base_path = pkg_resources.files("dopterianPlus.data")

# List available filters for K-correction
filter_list = dopterian.filer_list()

# Low-z parameters
sci_fits_path = [
    base_path / "SCI_F160W.fits",
    base_path / "SCI_F475W.fits",
    base_path / "SCI_F625W.fits",
    base_path / "SCI_F775W.fits",
    base_path / "SCI_F814W.fits",
]

sky_fits_path = [
    base_path / "SKY_F160W.fits",
    base_path / "SKY_F475W.fits",
    base_path / "SKY_F625W.fits",
    base_path / "SKY_F775W.fits",
    base_path / "SKY_F814W.fits",
]

imerr_fits_path = [
    base_path / "RMS_F160W.fits",
    base_path / "RMS_F475W.fits",
    base_path / "RMS_F625W.fits",
    base_path / "RMS_F775W.fits",
    base_path / "RMS_F814W.fits",
]

lowz_psf_path = [
    base_path / "PSF_F160W.fits",
    base_path / "PSF_F475W.fits",
    base_path / "PSF_F625W.fits",
    base_path / "PSF_F775W.fits",
    base_path / "PSF_F814W.fits",
]

lowz_zero_point = [25.946, 26.056, 25.899, 25.661, 25.946]
lowz_time_exp = [5029.34, 4128.0, 4066.0, 4126.0, 8080.0]
lowz_filter = ["clash_wfc3_f160w", "clash_wfc3_f475w", "clash_wfc3_f625w", "clash_wfc3_f775w", "clash_wfc3_f814w"]
lowz_pixscale = 0.065
lowz_lambda = [15405, 4770, 6310, 7647, 8057]
err0_mag = [0.05, 0.05, 0.05, 0.05, 0.05]

# High-z parameters
highz_psf_path = base_path / "PSF_F814W.fits"
highz_zero_point = 25.94662243630965
highz_time_exp = 8080.0
highz_filter = "clash_wfc3_f814w"
highz_pixscale = 0.065
highz_lambda = 8057

# Output file paths
output_path = [
    base_path / "SCI_OUTPUT.fits",
    base_path / "PSF_OUTPUT.fits",
]

# Combine low-z and high-z parameters
lowz_info = {
    "redshift": 0.18,
    "psf": lowz_psf_path,
    "zp": lowz_zero_point,
    "exptime": lowz_time_exp,
    "filter": lowz_filter,
    "pixscale": lowz_pixscale,
    "lambda": lowz_lambda,
}

highz_info = {
    "redshift": 2.0,
    "psf": highz_psf_path,
    "zp": highz_zero_point,
    "exptime": highz_time_exp,
    "filter": highz_filter,
    "pixscale": highz_pixscale,
    "lambda": highz_lambda,
}

# Run the simulation with the ferengi function
imOUT, psfOUT = dopterian.ferengi(
    images=sci_fits_path,
    background=sky_fits_path,
    lowz_info=lowz_info,
    highz_info=highz_info,
    namesout=output_path,
    imerr=imerr_fits_path,
    err0_mag=err0_mag,
    kc_obj=None,
    noflux=False,
    evo=None,
    minimum_output_size=None,
    noconv=False,
    extend=False,
    check_psf_FWHM=False,
    nonoise=True,
)

#grafricar imout
import matplotlib.pyplot as plt
plt.imshow(imOUT)
plt.show()
