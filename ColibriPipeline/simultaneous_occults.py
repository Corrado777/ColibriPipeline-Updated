# -*- coding: utf-8 -*-
"""
Created on Jul 29 10:04:14 2022

@author: Roman A.

Match dip detection txts throughout 3 telescopes based on time and coordinates, runs only on Green

2022-09-21 Roman A. simplified some steps and added data removal output
"""

# Module Imports
import numpy as np
import pandas as pd
from pathlib import Path
import os
import re
import time
import shutil
from astropy import wcs
from astropy.io import fits
import astrometrynet_funcs
import getRAdec
import itertools
from math import isclose
import sys
import fnmatch
import math
import argparse
from datetime import datetime,date,timedelta

# Custom Script Imports
import generate_specific_lightcurve as gsl
from detection import DetectionConfig
from detection.coincidence import active_telescopes, post_threshold_match, Detection


#-------------------------------global vars-----------------------------------#

# Path variables - environment-aware (sim vs real); see colibri_config.
# self_local=True: real-mode running scope (Green) is the locally-mounted D:/.
import colibri_config as cfg
_env = cfg.ENV
BASE_PATH = cfg.resolve_base_path(default_color='Green')
_scope_bases = cfg.telescope_base_dirs(self_local=True)
RED_BASE   = _scope_bases['REDBIRD']
GREEN_BASE = _scope_bases['GREENBIRD']
BLUE_BASE  = _scope_bases['BLUEBIRD']
DATA_PATH = BASE_PATH / 'ColibriData'
IMGE_PATH = BASE_PATH / 'ColibriImages'
ARCHIVE_PATH = BASE_PATH / 'ColibriArchive'
CENTRAL_PATH = BASE_PATH / 'CentralRepo'

# STRP formats
BARE_FORMAT = '%Y-%m-%d_%H%M%S_%f'
MINDIR_FORMAT = '%Y%m%d_%H.%M.%S.%f'
TIMESTAMP_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'
OBSDATE_FORMAT = '%Y%m%d'
NICE_FORMAT = '%Y-%m-%d_%H:%M:%S'

# Regex patterns
DET_TIME_REGEX = re.compile('det_(\d{4}-\d{2}-\d{2}_\d{6}_\d{6})\d{3}')

# Detection tolerances
TIME_TOLERANCE = 0.2  # seconds
COORD_TOLERANCE = 0.002  # degrees


#-------------------------------classes---------------------------------------#

class Telescope:

    def __init__(self, name, base_path, obs_date):

        # Telescope identifiers
        print(f"Initializing {name}...")
        self.name = name

        # Error logging
        self.errors = []

        # Path variables
        self.base_path = Path(base_path)
        self.obs_archive = self.base_path / "ColibriArchive" / hyphonateDate(obs_date)

        # Check if the archive for that night exists
        if self.obs_archive.exists():
            det_list = list(self.obs_archive.glob("det_*.txt"))
        else:
            self.addError(f"ERROR: No archive found for {self.name} on {obs_date}!")
            det_list = []

        # Analyze time of detections
        self.det_times = [datetime.strptime(DET_TIME_REGEX.match(det.name).group(1), BARE_FORMAT)
                          for det in det_list]
        self.det_dict = dict(zip(self.det_times, det_list))

        # Artificial generation required
        self.gen_artificial = []


    def addError(self, error_msg):

        self.errors.append(error_msg)
        print(error_msg)


    def writeGenerateCmds(self):

        if not self.obs_archive.exists():
            self.addError(f"ERROR: Could not write artificial generation command for {self.name}!")

        print(f"Writing artificial generation command(s) to {self.name}..")

        # Write the generate artificial command
        # If there are none are queued, touch the file
        if self.gen_artificial == []:
            (self.obs_archive / 'generate_artificial.txt').touch()
        
        # Otherwise, write the commands
        else:
            with open((self.obs_archive / 'generate_artificial.txt'), 'w') as gat:
                for gen_cmd in self.gen_artificial:
                    print(" -> " + gen_cmd)
                    gat.write(gen_cmd + '\n')

        return self.gen_artificial


#-------------------------------functions-------------------------------------#

def readRAdec(filepath):
    """
    Reads Ra and Dec line in the detection .txt file

    Parameters
    ----------
    filepath : path-like obj.
        Path of the detection .txt

    Returns
    -------
    star_ra : float
        RA of the occulted star.
    star_dec : float
        Dec of the occulted star.

    """
    
    with open(filepath,'r') as f:
        
        #loop through each line of the file
        for i, line in enumerate(f):
            
            if i==6:
                
                try:
                    star_coords = line.split(':')[1].split(' ')[1:3]
                    star_ra = float(star_coords[0])
                    star_dec = float(star_coords[1])

                    return star_ra, star_dec
                except:
                    print(f'ERROR: could not read RA and Dec in {filepath}! Please reprocess data.')
                    return float('inf'), float('inf')


def hyphonateDate(obsdate):

    # Convert the date to a datetime object
    obsdate = datetime.strptime(obsdate, OBSDATE_FORMAT)

    # Convert the date to a hyphonated string
    obsdate = obsdate.strftime('%Y-%m-%d')

    return obsdate


#-------------------------------main------------------------------------------#

def main():


###########################
## Argument Parser/Setup
###########################

    # Generate argument parser
    description = "Match occultation candidate events. Tiers are as follows:\n" +\
                   "1. Match to 1 second\n2. 2 files matched to 0.2s\n3. 3 files matched to 0.2s\n"
    arg_parser = argparse.ArgumentParser(description=description,
                                         formatter_class=argparse.RawTextHelpFormatter)
    
    # Available argument functionality
    arg_parser.add_argument('date', help='Observation date (YYYYMMDD) of data to be processed.')
    arg_parser.add_argument('--coord-tolerance', type=float, default=COORD_TOLERANCE,
                            help=f'Match cone radius in degrees (default {COORD_TOLERANCE} = 7 arcsec). '
                                 'Raise (e.g. 0.05) to recover nights from before a pointing correction.')
    arg_parser.add_argument('--coincidence-mode', choices=['post_threshold', 'joint_statistic'],
                            default='post_threshold',
                            help='Multi-telescope coincidence mode (default: post_threshold).')

    # Process argparse list as useful variables
    cml_args  = arg_parser.parse_args()
    obsdate   = cml_args.date
    coord_tolerance = cml_args.coord_tolerance
    coincidence_mode = cml_args.coincidence_mode

    # Per-scope archive bases (matches the coincidence module's scope names)
    SCOPE_BASES = {"REDBIRD": RED_BASE, "GREENBIRD": GREEN_BASE, "BLUEBIRD": BLUE_BASE}
    SCOPE_TELESCOPES = {scope: Telescope(scope, base, obsdate)
                        for scope, base in SCOPE_BASES.items()}

    # Initialize telescope classes (keep legacy names for writeGenerateCmds)
    Red   = SCOPE_TELESCOPES["REDBIRD"]
    Green = SCOPE_TELESCOPES["GREENBIRD"]
    Blue  = SCOPE_TELESCOPES["BLUEBIRD"]

    # Setup matched directory structure
    matched_dir = Green.obs_archive / "matched"
    if not matched_dir.exists():
        matched_dir.mkdir(parents=True, exist_ok=True)

    # Build the detection config carrying the current tolerances + mode
    config = DetectionConfig(time_tolerance_s=TIME_TOLERANCE,
                             coord_tolerance_deg=coord_tolerance,
                             coincidence_mode=coincidence_mode)


###########################
## Coincidence Matching
###########################

    # Determine which scopes actually produced detections for the night
    active = active_telescopes(obsdate)
    print(f"Active telescopes tonight: {sorted(active)}")

    # Parse each active scope's det_*.txt into Detection records
    detections_by_scope = {}
    for scope in active:
        tel = SCOPE_TELESCOPES[scope]
        dets = []
        for det_time, det_path in tel.det_dict.items():
            ra, dec = readRAdec(det_path)
            dets.append(Detection(time=det_time, ra=ra, dec=dec, path=str(det_path)))
        detections_by_scope[scope] = dets

    # Delegate the AND coincidence to the detection module
    matches = post_threshold_match(active, detections_by_scope, config)

    # Drive the matched/<timestamp>-Tier<N>/ directory output from the matches.
    # A match must involve >= 2 scopes to constitute a cross-telescope
    # coincidence (single-scope nights leave matched/ empty, as before).
    n_active = len(active)
    any_match = False
    for match in matches:
        scopes = match['scopes']
        tier = match['tier']

        # A lone detection (tier 1 on a multi-scope night) is not a coincidence.
        if tier < 2 and n_active >= 2:
            continue

        any_match = True

        # Representative timestamp drives the directory name (matches legacy)
        rep_time = match['time']
        if not isinstance(rep_time, datetime):
            rep_time = datetime.fromtimestamp(float(rep_time))
        dir_name = rep_time.strftime(BARE_FORMAT)

        match_dir = matched_dir / f"{dir_name}-Tier{tier}"
        if not match_dir.exists():
            match_dir.mkdir(parents=True, exist_ok=True)

        # Copy the participating det files
        for scope in scopes:
            det_path = match['paths'].get(scope)
            if det_path is not None:
                shutil.copy(det_path, match_dir)
        print(f"Tier{tier} match {dir_name}: {sorted(scopes)}")

        # Queue artificial-lightcurve generation for active scopes missing here
        timestamp = rep_time.strftime(TIMESTAMP_FORMAT)
        radec = (match['ra'], match['dec'])
        for scope in active:
            if scope not in scopes:
                command = f"{obsdate} {timestamp} {radec[0]} {radec[1]}"
                SCOPE_TELESCOPES[scope].gen_artificial.append(command)

    if not any_match:
        print("No coincident matches tonight!")

    # Write generate_artificial.txt on all 3 telescopes (touch if empty)
    Red.writeGenerateCmds()
    Green.writeGenerateCmds()
    Blue.writeGenerateCmds()

    print("Done!")


if __name__ == '__main__':
    main()
