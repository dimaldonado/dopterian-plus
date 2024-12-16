Usage examples
==============

Introduction
------------

The Dopterian-Plus package includes example FITS files of a galaxy from the Abell 209 cluster, observed with the Hubble Space Telescope. These files allow users to test the functionalities of the package without requiring additional data preparation.

The package provides images taken in five photometric filters:

- **F160W**
- **F475W**
- **F625W**
- **F775W**
- **F814W**

For each filter, the package includes:

- The galaxy's science image.
- The corresponding point spread function (PSF).
- The local input error image, used for uncertainty estimation.
- A sky background image for background correction.

Input filter requirement
------------------------

The input image filters **must** be included in the `kcorrect` library catalog; otherwise, the simulation will fail. To check which filters are available in the `kcorrect` library, you can use the following code:

.. code-block:: python

   from dopterianPlus import dopterian

   # List all available filters in the kcorrect library
   filters = dopterian.filer_list()
   print(filters)

Simulation example
------------------

Below is an example of how to simulate the appearance of a galaxy at a higher redshift using the FITS files provided with the package:

.. code-block:: python

   from dopterianPlus import dopterian
   import importlib.resources as pkg_resources

   # Base path for accessing packaged data files
   base_path = pkg_resources.files("dopterianPlus.data")

   # Define input parameters (low-z and high-z)
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

   lowz_info = {
       "redshift": 0.18,
       "psf": lowz_psf_path,
       "zp": [25.946, 26.056, 25.899, 25.661, 25.946],
       "exptime": [5029.34, 4128.0, 4066.0, 4126.0, 8080.0],
       "filter": ["clash_wfc3_f160w", "clash_wfc3_f475w", "clash_wfc3_f625w", "clash_wfc3_f775w", "clash_wfc3_f814w"],
       "pixscale": 0.065,
       "lambda": [15405, 4770, 6310, 7647, 8057],
   }

   highz_psf_path = base_path / "PSF_F814W.fits"
   highz_info = {
       "redshift": 2.0,
       "psf": highz_psf_path,
       "zp": 25.94662243630965,
       "exptime": 8080.0,
       "filter": "clash_wfc3_f814w",
       "pixscale": 0.065,
       "lambda": 8057,
   }

   # Define output parameters
   output_path = [
       base_path / "SCI_OUTPUT.fits",
       base_path / "PSF_OUTPUT.fits",
   ]

   # Run the simulation
   imOUT, psfOUT = dopterian.ferengi(
       images=sci_fits_path,
       background=sky_fits_path,
       lowz_info=lowz_info,
       highz_info=highz_info,
       namesout=output_path,
       imerr=imerr_fits_path,
       err0_mag=[0.05, 0.05, 0.05, 0.05, 0.05],
       kc_obj=None,
       noflux=False,
       evo=None,
       minimum_output_size=None,
       noconv=False,
       extend=False,
       check_psf_FWHM=False,
       nonoise=True,
   )

For more details about the ``ferengi`` function, including its parameters and usage, refer to its :py:func:`dopterianPlus.dopterian.ferengi`. This will provide an in-depth explanation of how the function operates and additional options that can be configured.

Efficient use of the kcorrect object
---------------------------------------

When running simulations on datasets with a considerable number of galaxies, all sharing the same input and output filters, it is **recommended** to create the ``kcorrect`` object beforehand. This avoids redundant computations and significantly improves the performance of the simulation.

To dynamically calculate the ``responses_map`` parameter based on the closest filter in ``lowz_info``, use the following code:

.. code-block:: python

    from dopterianPlus import cosmology as cosmos
    from astropy.cosmology import FlatLambdaCDM
    import kcorrect
    import numpy as np

    # Define cosmology using values from dopterianPlus
    cos = FlatLambdaCDM(H0=cosmos.H0, Om0=cosmos.Omat, Ob0=cosmos.Obar)

    # Dynamically calculate responses_map (closest filter in lowz_info to highz_info)
    dz = np.abs(np.array(lowz_info['lambda']) / highz_info['lambda'] - 1)
    idx_bestfilt = np.argmin(dz)  # Index of the closest matching low-redshift filter
    responses_map = [lowz_info['filter'][idx_bestfilt]]  # The closest low-redshift filter as a list

    # Create the kcorrect object
    kc = kcorrect.kcorrect.Kcorrect(
        responses=lowz_info['filter'],          # Filters at low redshift
        responses_out=[highz_info['filter']],  # Single filter at high redshift
        responses_map=responses_map,           # Dynamically computed responses_map
        cosmo=cos
    )

By precomputing the ``kcorrect`` object and passing it as the ``kc_obj`` parameter to the ``ferengi`` function, you can avoid recalculating the same information for every galaxy, reducing computation time significantly:

.. code-block:: python

    imOUT, psfOUT = dopterian.ferengi(
       images=sci_fits_path,
       background=sky_fits_path,
       lowz_info=lowz_info,
       highz_info=highz_info,
       namesout=output_path,
       imerr=imerr_fits_path,
       err0_mag=[0.05, 0.05, 0.05, 0.05, 0.05],
       kc_obj=kc,
       noflux=False,
       evo=None,
       minimum_output_size=None,
       noconv=False,
       extend=False,
       check_psf_FWHM=False,
       nonoise=True,
   )
