import os
import numpy as np
from astropy.io import fits

# FUNCTION THAT CUTS A POSTAGE STAMP IMAGE AND SUBTRACTS THE BACKGROUND
# (Pierluigi Cerulo 24/04/2019)

def cut_stamp(parent_image_data, parent_image_header, postage_stamp_image_prefix, size_x, size_y, x_image, y_image, background_value, RA_object, DEC_object):
    """
    Extract a postage stamp image from a parent astronomical image and subtract the background.

    This function creates a smaller image centered on a specified position within a larger parent image.
    It updates the image header with the coordinates of the extracted region and subtracts a background value.

    Parameters
    ----------
    parent_image_data : ndarray
        The 2D data array of the parent image.
    parent_image_header : `astropy.io.fits.Header`
        Header of the parent image containing metadata such as dimensions and WCS information.
    postage_stamp_image_prefix : str
        Prefix for the output postage stamp FITS file.
    size_x : int
        Half-width of the postage stamp in pixels along the x-axis.
    size_y : int
        Half-height of the postage stamp in pixels along the y-axis.
    x_image : float
        X-coordinate of the center of the postage stamp in the parent image (in pixels).
    y_image : float
        Y-coordinate of the center of the postage stamp in the parent image (in pixels).
    background_value : float
        Background value to subtract from the postage stamp image.
    RA_object : float
        Right Ascension (RA) of the object at the center of the postage stamp (in degrees).
    DEC_object : float
        Declination (DEC) of the object at the center of the postage stamp (in degrees).

    Returns
    -------
    None
        The function writes the postage stamp image to a FITS file and does not return any value.
    """
    #removing output from previous run
    os.system('rm '+postage_stamp_image_prefix+'.fits')
   
    # defining sizes of parent image
    size_x_parent_image = parent_image_header['NAXIS1']
    size_y_parent_image = parent_image_header['NAXIS2']
       
    # cutting postage_stamp image
    postage_stamp_data = parent_image_data[ np.max([int(y_image-size_y), 0]):np.min([int(y_image+size_y), int(size_y_parent_image)]), np.max([int(x_image-size_x), 0]):np.min([int(x_image+size_x), int(size_x_parent_image)]) ]
   
    print (postage_stamp_image_prefix, int(y_image-size_y), int(y_image+size_y), int(x_image-size_x), int(x_image+size_x))
   
    # updating header
    # - defining centre of postage stamp image
    xc = size_x+1.0
    yc = size_y+1.0

    # - updating coordinates
    postage_stamp_header = parent_image_header

    postage_stamp_header['CRPIX1'] = xc
    postage_stamp_header['CRPIX2'] = yc

    postage_stamp_header['CRVAL1'] = RA_object
    postage_stamp_header['CRVAL2'] = DEC_object

    # subtracting background from image
    postage_stamp_data_output = postage_stamp_data #- background_value

    # setting nan values to 0
    #postage_stamp_data_nan = np.where(np.isnan(postage_stamp_data_output) == True)[0]
    postage_stamp_data_output_zero = np.nan_to_num(postage_stamp_data_output)
   
   
    # writing postage-stamp image to file
    hdu_postage_stamp = fits.PrimaryHDU(data=postage_stamp_data_output_zero, header=postage_stamp_header)
    hdu_postage_stamp.writeto(postage_stamp_image_prefix+'.fits')
