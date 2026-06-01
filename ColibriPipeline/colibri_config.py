"""
Filename:   colibri_config.py
Purpose:    Single source of truth for environment-aware path resolution and the
            shared constants (telescope colors, datetime formats, site location)
            that the Colibri pipeline stages previously each re-implemented.

Before this module, the ~10-line "sim vs real" path block was copy-pasted into
all nine stage scripts, with subtle drift between copies (default color Red vs
Green; ``os.environ['COMPUTERNAME']`` with no default -> KeyError crash in sim).
Stages now call into here instead.

Resolution mirrors lightcurve_analysis/paths.py:
    real telescope -> base ``D:/``
    sim (COLIBRI_ENV=sim) -> ``<COLIBRI_SIM_ROOT>/<color>`` where color comes
        from the telescope label via TELESCOPE_COLORS.

Kept deliberately lightweight: NO matplotlib / pandas imports so any module can
import it cheaply. astropy is imported lazily for SITE_LOC only.
"""

import os
from pathlib import Path


#-------------------------------- telescopes ---------------------------------#

TELESCOPE_COLORS = {
    'REDBIRD': 'Red',
    'GREENBIRD': 'Green',
    'BLUEBIRD': 'Blue',
}
# Canonical processing order used by the cross-telescope stages.
TELESCOPE_ORDER = ['REDBIRD', 'GREENBIRD', 'BLUEBIRD']

# Default sim root if COLIBRI_SIM_ROOT is unset (matches the value the stages
# previously hard-coded as their fallback).
_DEFAULT_SIM_ROOT = '/home/agirmen/research_data/ColibriPipelineSimulatedDirs'

# Resolved once at import; ``COLIBRI_ENV`` selects sim vs real layout.
ENV = os.environ.get('COLIBRI_ENV', 'real').lower()
IS_SIM = ENV == 'sim'


def sim_root():
    """Return the sim tree root as a Path (env COLIBRI_SIM_ROOT, else default)."""
    return Path(os.environ.get('COLIBRI_SIM_ROOT', _DEFAULT_SIM_ROOT))


def current_telescope(default='GREENBIRD'):
    """Resolve this machine's telescope label without ever raising.

    Precedence: ``COLIBRI_TELESCOPE`` -> ``COMPUTERNAME`` -> ``default``.
    Returns an upper-cased canonical label (e.g. ``GREENBIRD``).
    """
    label = os.environ.get('COLIBRI_TELESCOPE',
                           os.environ.get('COMPUTERNAME', default))
    return label.upper()


def telescope_color(telescope=None, default_color='Green'):
    """Map a telescope label (or color) to its sim color subdirectory."""
    if telescope is None:
        telescope = current_telescope()
    key = str(telescope).upper()
    if key in TELESCOPE_COLORS:
        return TELESCOPE_COLORS[key]
    for color in TELESCOPE_COLORS.values():        # already a color?
        if key == color.upper():
            return color
    return default_color


def resolve_base_path(telescope=None, default_color='Green'):
    """Resolve the base directory for *this* telescope.

    sim  -> ``<sim_root>/<color>``
    real -> ``D:/``

    ``default_color`` is the fallback color when the telescope label is unknown
    (most stages used 'Green'; colibri_main historically used 'Red').
    """
    if IS_SIM:
        if telescope is None:
            telescope = current_telescope()
        return sim_root() / telescope_color(telescope, default_color)
    return Path('D:/')


def telescope_base_dirs(self_local=True):
    """Return ``{REDBIRD/GREENBIRD/BLUEBIRD: base Path}`` for all three scopes.

    sim  -> each maps to ``<sim_root>/<color>``.
    real -> R:/, G:/, B:/ network mounts; if ``self_local`` the running scope is
            replaced with the locally-mounted ``D:/`` (matches the historical
            cumulative_stats / simultaneous_occults / timeline behaviour).
    """
    if IS_SIM:
        root = sim_root()
        return {name: root / color for name, color in TELESCOPE_COLORS.items()}
    drives = {'REDBIRD': Path('R:/'), 'GREENBIRD': Path('G:/'), 'BLUEBIRD': Path('B:/')}
    if self_local:
        drives[current_telescope()] = Path('D:/')
    return drives


#----------------------------- datetime formats ------------------------------#

OBSDATE_FORMAT   = '%Y%m%d'
MINDIR_FORMAT    = '%Y%m%d_%H.%M.%S.%f'
TIMESTAMP_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'
BARE_FORMAT      = '%Y-%m-%d_%H%M%S_%f'
CLOCK_FORMAT     = '%H:%M:%S'
ACPLOG_STRP      = '%a %b %d %H:%M:%S %Z %Y'

# Minute-directory name regex (e.g. 20250830_01.54.12.147)
import re as _re
MINDIR_REGEX = _re.compile(r'\d{8}_\d{2}\.\d{2}\.\d{2}\.\d{3}')


#------------------------------ site location --------------------------------#

# Elginfield Observatory
SITE_LAT = 43.1933116667
SITE_LON = -81.3160233333
SITE_HGT = 224

_SITE_LOC = None


def site_location():
    """Lazily build and cache the astropy EarthLocation for the site."""
    global _SITE_LOC
    if _SITE_LOC is None:
        from astropy.coordinates import EarthLocation
        _SITE_LOC = EarthLocation(lat=SITE_LAT, lon=SITE_LON, height=SITE_HGT)
    return _SITE_LOC


#------------------------------ weather schema -------------------------------#

WEATHER_HEADERS = ['unix_time', 'temp', 'humidity', 'wind_speed', 'wind_direction',
                   'rain_value', 'sky_temp', 'ground_temp', 'alert', 'polaris_mag']
