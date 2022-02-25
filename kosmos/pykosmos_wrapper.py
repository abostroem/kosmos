import os
import numpy as np
from matplotlib import pyplot as plt

import h5py
import pandas as pd
from astropy.table import Table
from astropy import units as u
from astropy import nddata
from datetime import datetime
from astropy.convolution import convolve, Box1DKernel

import sys
sys.path.append('/Users/bostroem/Desktop/research/not_my_code/kosmos')
import kosmos
kosmos.__version__
import specutils

#TODO list
#Currently write HDF5 files, should it write fits files (e.g. for DS( quick view))
#Record header information in output
#Better way to do plotting where you can combine after the fact?

def make_superbias(input_list_filename, output_filename='bias.hdf', 
                    input_format='ascii.no_header', input_directory='./' ,
                    output_directory=None, interactive_plots=True, ax=None, **kwargs):
    """ Make a super bias. Output file and plots will be put in the output directory, default is the input_directory"""
    assert output_filename.endswith('hdf'), 'output_file must end in .hdf'
    bias_files = Table.read(os.path.join(input_directory, input_list_filename), names=['impath'], format=input_format)
    bias_files['impath'] = [os.path.join(input_directory, ifile) for ifile in bias_files['impath']]
    bias = kosmos.biascombine(bias_files['impath'], **kwargs)
    #Save
    if output_directory is None:
        output_directory = input_directory
    with h5py.File(os.path.join(output_directory, output_filename), 'w') as output_file:
        dset = output_file.create_dataset(name='bias', data=bias)
    #Plot
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(8, 4))
    vmin, vmax=np.percentile(bias, (5, 98))
    im = ax.imshow(bias, origin='lower', aspect='auto', cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
    ax.set_title('median bias frame {}'.format(output_filename))
    plt.colorbar(mappable=im, ax=ax)
    plt.savefig(os.path.join(output_directory, output_filename.replace('hdf', 'pdf')))
    plt.draw()
    if interactive_plots:
        input('Press enter to exit interactive')
    plt.close()
    return bias

def make_superflat(input_list_filename, bias, output_filename='flat.hdf', 
                   input_format='ascii.no_header', input_directory='./' ,
                   output_directory=None, interactive_plots=True, verbose=False, ax=None, **kwargs):
    """ Make a super flat. Output file and plots will be put in the output directory, default is the input_directory"""
    assert output_filename.endswith('hdf'), 'output_file must end in .hdf'
    flat_files = Table.read(os.path.join(input_directory, input_list_filename), names=['impath'], format=input_format)
    flat_files['impath'] = [os.path.join(input_directory, ifile) for ifile in flat_files['impath']]
    flat_output = kosmos.flatcombine(flat_files['impath'], bias=bias, **kwargs)
    if isinstance(flat_output, tuple):
        flat, illum = flat_output
    else:
        flat = flat_output
        illum = None
    #Save
    if output_directory is None:
        output_directory = input_directory
    with h5py.File(os.path.join(output_directory, output_filename), 'w') as output_file:
        dset = output_file.create_dataset(name='flat', data=flat)
        if illum is not None:
            illumset = output_file.create_dataset(name='illum', data=illum)
    if verbose:
        if illum is not None:
            print('illuminated shape {}'.format(illum.shape))
        print('flat shape {}'.format(flat.shape))
        print('flat unit {}'.format(flat.unit))
    #Plot
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(8, 4))
    im = ax.imshow(flat, origin='lower', aspect='auto', cmap=plt.cm.inferno, vmin=0.9, vmax=1.1)
    ax.set_title('median flat frame, bias & response corrected {}'.format(output_filename))
    plt.colorbar(mappable=im, ax=ax)
    fig.savefig(os.path.join(output_directory, output_filename.replace('hdf', 'pdf')))
    if interactive_plots:
        input('Press enter to exit interactive')
    plt.close()
    if illum is not None:
        return flat, illum
    else:
        return flat

def do_2d_reduction(input_filename, bias, flat, illum=None, trim=True, input_directory='./' ,
                   output_filename=None, output_directory=None, interactive_plots=True, verbose=False, ax=None, **kwargs):
    """ Bias and Flat field a 2D spectrogram"""
    sciimg = kosmos.proc(os.path.join(input_directory, input_filename), 
                             bias=bias, flat=flat, ilum=illum, trim=trim, **kwargs)
    #Save
    if output_directory is None:
        output_directory = input_directory
    if output_filename is None:
        output_filename = input_filename.replace('.fits', '_flt.hdf')
    assert output_filename.endswith('hdf'), 'output_file must end in .hdf'

    with h5py.File(os.path.join(output_directory, output_filename), 'w') as output_file:
        dset = output_file.create_dataset(name='sciimage', data=sciimg)
    if verbose:
        print('sci shape {}'.format(sciimg.shape))
        print('sci unit {}'.format(sciimg.unit))
    #Plot
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(8, 4))
    vmin, vmax = np.percentile(sciimg, (5, 98))
    im = ax.imshow(sciimg, origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    if illum is not None:
        corrections = 'bias, flat, and response'
    else:
        corrections = 'bias and flat'
    ax.set_title('science frame, {} corrected {}'.format(corrections, output_filename))
    plt.colorbar(mappable=im, ax=ax)
    fig.savefig(os.path.join(output_directory, output_filename.replace('hdf', 'pdf')))
    if interactive_plots:
        input('Press enter to exit interactive')
    plt.close()
    return sciimg

def trace_spectrum(image2d,  output_filename, input_directory='./',
                 output_directory=None, interactive_plots=True,  **kwargs):
    assert output_filename.endswith('hdf'), 'output_file must end in .hdf'
    if output_directory is None:
        output_directory = input_directory
    trace = kosmos.trace(image2d, display=True, **kwargs)
    with h5py.File(os.path.join(output_directory, output_filename), 'w') as output_file:
        dset = output_file.create_dataset(name='trace', data=trace)
    #TODO: Figure out how to capture this output - right now I can't get the figure object
    fig = plt.gcf()
    fig.savefig(os.path.join(output_directory, output_filename.replace('hdf', 'pdf')))
    if interactive_plots:
        input('Press enter to exit interactive')
    plt.close(fig)
    return trace

def extract_spectrum(image2d, trace, output_filename, input_directory='./',
                 output_directory=None, interactive_plots=True, arc=False, **kwargs):
    """
    Output file is a fits file
    """
    assert output_filename.endswith('fits'), 'output_file must end in .fits'
    if output_directory is None:
        output_directory = input_directory
    obj_spectrum, sky_spectrum = kosmos.BoxcarExtract(image2d, trace, display=True, **kwargs)
    if 'x1d' not in output_filename:
        output_filename = output_filename.replace('.fits', '_x1d.fits')
    obj_spectrum.write(os.path.join(output_directory, output_filename), format='tabular-fits', overwrite=True)
    sky_spectrum.write(os.path.join(output_directory, output_filename.replace('.fits', '_sky.fits')), format='tabular-fits', overwrite=True)
    #TODO: Figure out how to capture this output - right now I can't get the figure object
    fig = plt.gcf()
    fig.savefig(os.path.join(output_directory, output_filename.replace('.fits', '_extract.pdf')))
    fig_spectrum, ax_spectrum = plt.subplots(1,1)
    if not arc:
        ax_spectrum.plot(obj_spectrum.spectral_axis.value, obj_spectrum.flux.value - sky_spectrum.flux.value)
    else:
        ax_spectrum.plot(obj_spectrum.spectral_axis.value, obj_spectrum.flux.value)
    ax_spectrum.set_xlabel(obj_spectrum.spectral_axis.unit)
    ax_spectrum.set_ylabel(obj_spectrum.flux.unit)
    ax_spectrum.set_title('Boxcar extraction')
    if interactive_plots:
        input('Press enter to exit interactive')
    fig_spectrum.savefig(os.path.join(output_directory, output_filename.replace('.fits', '.pdf')))
    plt.close(fig)
    plt.close(fig_spectrum)
    return obj_spectrum, sky_spectrum

def calibrate_wavelengths(image, arc_spectrum, obj_spectrum, sky_spectrum, output_filename, 
                        output_directory='./', blue_left=True, interactive_plots=True, **kwargs):
    assert output_filename.endswith('.fits'), 'output_filename must end in .fits'
    if 'wx1d' not in output_filename:
        output_filename = output_filename.replace('.fits', '_wx1d.fits')
    wapprox = (np.arange(image.shape[1])-image.shape[1]/2)
    if not blue_left:
        wapprox = wapprox[::-1]
    wapprox = wapprox*image.header['DISPDW']+ image.header['DISPWC']
    wapprox = wapprox*u.angstrom
    kosmos_dir = os.path.dirname(kosmos.__file__)
    henear_tbl = Table.read(os.path.join(kosmos_dir, 'resources/linelists/apohenear.dat'),
                       names=['wave', 'name'], format='ascii')
    henear_tbl['wave'].unit = u.angstrom
    apo_henear = henear_tbl['wave']
    #Map pixel to wavelength
    sci_xpts, sci_wpts = kosmos.identify_nearest(arc_spectrum, wapprox=wapprox, linewave=apo_henear, **kwargs)
    #Create sky subtracted spectrum
    obj_flux = obj_spectrum.flux-sky_spectrum.flux
    sky_sub_obj_spectrum = specutils.Spectrum1D(flux=obj_flux, spectral_axis=obj_spectrum.spectral_axis, 
                                    uncertainty=obj_spectrum.uncertainty)
    #Apply the wavelength solution to the sky subtracted spectrum
    obj_spectrum_wavelength_calibrated = kosmos.fit_wavelength(sky_sub_obj_spectrum, sci_xpts, sci_wpts, mode='interp', deg=3)
    obj_spectrum_wavelength_calibrated.write(os.path.join(output_directory, output_filename), format='tabular-fits', overwrite=True)
    fig,(ax1, ax2) = plt.subplots(2,1, figsize=(8,6))
    ax1.plot(wapprox.value, arc_spectrum.flux.value)
    ax1.set_xlabel(wapprox.unit)
    ax1.set_ylabel(arc_spectrum.flux.unit)
    ymin, ymax = ax1.get_ylim()
    ax1.vlines(sci_wpts, color='r', ls=':', ymin=ymin, ymax=ymax)

    ax2.plot(obj_spectrum_wavelength_calibrated.spectral_axis, obj_spectrum_wavelength_calibrated.flux)
    ax2.set_title('Wavelength Calibrated Spectrum')
    plt.tight_layout()
    if interactive_plots:
        input('Press enter to exit interactive')
    fig.savefig(os.path.join(output_directory, output_filename.replace('.fits', '.pdf')))

    return obj_spectrum_wavelength_calibrated

def make_sens_func(std_filename, std_spectrum, mode='linear', **kwargs):
    standardstar = kosmos.onedstd(std_filename)
    sens_func = kosmos.standard_sensfunc(std_spectrum, standardstar, mode=mode, **kwargs)
    fig, ax = plt.subplots(1,1)
    ax.plot(sens_func.spectral_axis, sens_func.flux)
    ax.set_xlabel(sens_func.spectral_axis.unit)
    ax.set_ylabel('Sensitivity ({})'.format(sens_func.flux.unit))
    #TODO: this should write a file
    return sens_func

def apply_airmass_correction(header, spectrum, extinction_file='apoextinct.dat'):
    """
    Spectrum should be wavelength calibrated
    """
    # Get the airmass from the Headers... no fancy way to do this I guess? 
    ZD = header['ZD'] / 180.0 * np.pi
    airmass = 1.0/np.cos(ZD) # approximate Zenith Distance -> Airmass conversion
    # Select the observatory-specific airmass extinction profile from the provided "extinction" library
    Xfile = kosmos.obs_extinction(extinction_file)
    spectrum_airmass_corr = kosmos.airmass_cor(spectrum, airmass, Xfile)
    #TODO: this should write a file and maybe make a plot?
    return spectrum_airmass_corr

def flux_calibrate(obj_spectrum_wavelength_calibrated, sens_func, **kwargs):
    flux_calib_spectrum = kosmos.apply_sensfunc(obj_spectrum_wavelength_calibrated, sens_func, **kwargs)
    #TODO: this should write a file and maybe make a plot
    return flux_calib_spectrum

