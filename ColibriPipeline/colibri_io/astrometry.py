# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Astrometry: astrometry.net plate solving + pixel -> RA/Dec WCS transforms.

Relocated and merged from the top-level ``astrometrynet_funcs.py`` (plate solve)
and ``getRAdec.py`` (WCS pixel->world transforms) during the module-level
refactor (Wave 1). Behaviour-preserving.

The duplicated ``getRAdec_arrays()`` that previously lived in BOTH ``getRAdec.py``
and ``coordsfinder.py`` is consolidated here as the single canonical copy (the two
copies were identical). ``coordsfinder.py`` now imports it from here.

Original headers:
    astrometrynet_funcs.py -- Created Fri Apr 8 2022; Updated Thurs Jun 23 2022;
                              @author: Rachel A. Brown
    getRAdec.py            -- Created Thu Jul 8 2021; Update Jan. 24 2022;
                              @author: Rachel A Brown
"""

import time, subprocess, os
from astropy.io.fits import Header

import astropy
from astropy.io import fits
import numpy as np
import pathlib
import datetime
from astropy import wcs


#--------------------------- astrometry.net plate solve ----------------------#

def getSolution(image_file, save_file, order):
    '''send request to solve image from astrometry.net
    input: path to the image file to submit, filepath to save the WCS solution header to, order of soln
    returns: WCS solution header'''
    from astroquery.astrometry_net import AstrometryNet
    #astrometry.net API
    ast = AstrometryNet()

    #key for astrometry.net account
    ast.api_key = 'vbeenheneoixdbpb'    #key for Rachel Brown's account (040822)
    wcs_header = ast.solve_from_image(image_file, crpix_center = True, tweak_order = order, force_image_upload=True)

    #save solution to file
    if not save_file.exists():
            wcs_header.tofile(save_file)

    return wcs_header

def getLocalSolution(image_file, save_file, order):
    """
    Obtain a WCS plate solution for a median-stacked FITS image.

    Strategy (in order):
      1. On Windows: delegate to WSL (solve-field installed inside WSL).
      2. On Linux/Mac: call solve-field natively if index files are present
         (detected by a successful solve run).
      3. Fallback: submit to the astrometry.net web API (requires internet).

    Args:
        image_file: path to the median-stacked FITS image (string).
        save_file:  basename of the output file (e.g. 'myfield_4th_wcs.fits').
        order:      SIP polynomial order (int).

    Returns:
        astropy.io.fits.Header with WCS keywords, or None on failure.
    """
    import platform

    wcs_header = None
    base_name = save_file.split(".fits")[0]

    print(image_file)
    print(base_name)

    # --- attempt 1: local solve-field ---
    try:
        if platform.system() == 'Windows':
            cwd = os.getcwd()
            os.chdir('d:\\')
            subprocess.run(
                'wsl time solve-field --no-plots -D /mnt/d/tmp -O'
                ' -o ' + base_name +
                ' -N ' + save_file +
                ' -t ' + str(order) +
                ' --scale-units arcsecperpix --scale-low 2.2 --scale-high 2.6 ' +
                image_file,
                shell=True
            )
            os.chdir(cwd)
            wcs_header = Header.fromfile('d:\\tmp\\' + base_name + '.wcs')

        else:
            # Linux / Mac: call solve-field directly
            tmp_dir = '/tmp/colibri_wcs'
            os.makedirs(tmp_dir, exist_ok=True)
            result = subprocess.run(
                [
                    'solve-field', '--no-plots',
                    '-D', tmp_dir, '-O',
                    '-o', base_name,
                    '-N', os.path.join(tmp_dir, save_file),
                    '-t', str(order),
                    '--scale-units', 'arcsecperpix',
                    '--scale-low', '2.2',
                    '--scale-high', '2.6',
                    image_file,
                ],
                timeout=300
            )
            wcs_file = os.path.join(tmp_dir, base_name + '.wcs')
            if os.path.exists(wcs_file):
                wcs_header = Header.fromfile(wcs_file)
            else:
                raise FileNotFoundError(f"solve-field did not produce {wcs_file} "
                                        "(index files may not be installed)")

    except Exception as e:
        print(f"WARNING: Local solve-field failed: {e}")
        print("Falling back to astrometry.net web API...")

    # --- attempt 2: web API fallback ---
    if wcs_header is None:
        try:
            from astroquery.astrometry_net import AstrometryNet
            ast = AstrometryNet()
            ast.api_key = 'vbeenheneoixdbpb'
            wcs_header = ast.solve_from_image(
                image_file,
                crpix_center=True,
                tweak_order=order,
                force_image_upload=True,
                scale_units='arcsecperpix',
                scale_lower=2.2,
                scale_upper=2.6,
            )
        except Exception as e:
            print(f"WARNING: Web API WCS solve also failed: {e}")

    return wcs_header


#--------------------------- pixel -> RA/Dec WCS transforms -------------------#

def getRAdec(transform, star_pos_file, savefile):
    '''get WCS transform from astrometry.net header
    input: astrometry.net output file (path object), star position file (.npy path object), filename to save to (path object)
    returns: coordinate transform'''

    #load in transformation information
#    transform_im = fits.open(transform_file)
#    transform = wcs.WCS(transform_im[0].header)

    #get star coordinates from observation image (.npy file)
    star_pos = np.load(star_pos_file)

    #get transformation
    world = transform.all_pix2world(star_pos, 0,ra_dec_order=True) #2022-07-21 Roman A. changed solution function to fit SIP distortion
   # print(world)
   # px = transform.wcs_world2pix(world, 0)
   # print(px)

    #optional: save text file with transformation
    with open(savefile, 'w') as filehandle:
        filehandle.write('#\n#\n#\n#\n#X  Y  RA  Dec\n')

        for i in range(0, len(star_pos)):
            #output table: | x | y | RA | Dec |
            filehandle.write('%f %f %f %f\n' %(star_pos[i][0], star_pos[i][1], world[i][0], world[i][1]))

    coords = np.array([star_pos[:,0], star_pos[:,1], world[:,0], world[:,1]]).transpose()
    return coords


def getRAdec_arrays(transform, star_pos):
    '''get WCS transform from astrometry.net header
    input: astrometry.net output file (path object), star position file (.npy path object), filename to save to (path object)
    returns: coordinate transform

    Canonical copy consolidated from getRAdec.py and coordsfinder.py (Wave 1
    refactor). The two prior copies were identical.'''

    #load in transformation information
#    transform_im = fits.open(transform_file)
#    transform = wcs.WCS(transform_im[0].header)

    #get transformation
    world = transform.all_pix2world(star_pos, 0,ra_dec_order=True) #2022-07-21 Roman A. changed solution function to fit SIP distortion

    # output table: | x | y | RA | Dec |
    #coords = np.array([star_pos[:,0], star_pos[:,1], world[:,0], world[:,1]]).transpose()
    coords = np.hstack((star_pos[:,0], star_pos[:,1], world[:,0], world[:,1]))
    return coords


def getRAdecfromFile(transform_file, star_pos_file, savefile):
    '''get WCS transform from astrometry.net header
    input: astrometry.net output file (path object), star position file (.npy path object), filename to save to (path object)
    returns: coordinate transform'''

    #load in transformation information
    transform_im = fits.open(transform_file)
    transform = wcs.WCS(transform_im[0].header)

    #get star coordinates from observation image (.npy file)
    star_pos = np.load(star_pos_file)

    #get transformation
    world = transform.all_pix2world(star_pos, 0,ra_dec_order=True) #2022-07-21 Roman A. changed solution function to fit SIP distortion
   # print(world)
   # px = transform.wcs_world2pix(world, 0)
   # print(px)

    #optional: save text file with transformation
    with open(savefile, 'w') as filehandle:
        filehandle.write('#\n#\n#\n#\n#X  Y  RA  Dec\n')

        for i in range(0, len(star_pos)):
            #output table: | x | y | RA | Dec |
            filehandle.write('%f %f %f %f\n' %(star_pos[i][0], star_pos[i][1], world[i][0], world[i][1]))

    coords = np.array([star_pos[:,0], star_pos[:,1], world[:,0], world[:,1]]).transpose()
    return coords


def getXY(transform_file, star_pos_file, savefile=None):
    '''get WCS transform ([RA,dec] -> [X,Y]) from astrometry.net header
    input: astrometry.net output file, star position file (.npy)
    returns: coordinate transform'''

    #load in transformation information
    transform_im = fits.open(transform_file)
    transform = wcs.WCS(transform_im[0].header)

    #get star coordinates from observation image (.npy file)
    star_pos = np.load(star_pos_file)

    #get transformation
    px = transform.wcs_world2pix(star_pos, 0)
   # print(px)

    #optional: save text file with transformation
    if savefile != None:
        with open(savefile, 'w') as filehandle:
            filehandle.write('#\n#\n#\n#\n#X  Y  RA  Dec\n')

            for i in range(0, len(star_pos)):
                #output table: | x | y | RA | Dec |
                filehandle.write('%f %f %f %f\n' %(star_pos[i][0], star_pos[i][1], px[i][0], px[i][1]))

    coords = np.array([star_pos[:,0], star_pos[:,1], px[:,0], px[:,1]]).transpose()
    return coords


def getRAdecSingle(transform, star_pos):
    '''get WCS transform from astrometry.net header for a single star
    input: astrometry.net transformation (object), star position (X, Y)
    returns: star position in RA/dec'''

    #get transformation
    star_pos = np.array([[star_pos[0], star_pos[1]]])

    world = transform.all_pix2world(np.array(star_pos), 0,ra_dec_order=True) #2022-07-21 Roman A. changed solution function to fit SIP distortion
  #  star_pos = np.array([star_pos[0], star_pos[1]])
   # world = transform.pixel_to_world(np.array(star_pos))

    return world[0]

def getXYSingle(transform, star_pos):
    '''get WCS transform from astrometry.net header for a single star
    input: astrometry.net transformation (object), star position (RA, dec)
    returns: star position in X/Y'''

    #get transformation
    star_pos = np.array([[star_pos[0], star_pos[1]]])

    px = transform.wcs_world2pix(np.array(star_pos), 0)

    return px[0]


#-----------------------------------main--------------------------------------#

if __name__ == '__main__':
    wcs_soln = getLocalSolution('/mnt/d/testmedstack.fits', 'test-newer.fits', 3)
    print(wcs_soln)
