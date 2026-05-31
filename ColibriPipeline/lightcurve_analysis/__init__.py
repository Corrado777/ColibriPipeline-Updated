#!/usr/bin/env python3
"""
ColibriPipeline.lightcurve_analysis
===================================

Night-long light-curve analysis for the Colibri telescope array.

Wave 1 (core data contracts) public API:
    - paths.resolve_archive_root, paths.night_dir, paths.TELESCOPE_COLORS
    - io.MinuteRef, io.discover_minutes
    - matching.GlobalMatchTable, matching.match_minutes_to_global
    - stitch.LightCurve and binning/time-axis helpers
    - catalog.NightCatalog, catalog.build_night_catalog,
      catalog.save_cache, catalog.load_cache

Later waves add differential.py, gaia.py, plotting.py, cli.py, widgets.py.
"""

try:
    from .paths import (
        TELESCOPE_COLORS,
        resolve_archive_root,
        night_dir,
        normalize_obsdate,
    )
    from .io import MinuteRef, discover_minutes, load_minute_full, minute_radec
    from .matching import GlobalMatchTable, match_minutes_to_global
    from .stitch import (
        LightCurve,
        build_time_axis,
        gap_mask,
        bin_lightcurve,
        bin_by_count,
    )
    from .catalog import (
        NightCatalog,
        build_night_catalog,
        save_cache,
        load_cache,
    )
except ImportError:
    # Allow flat-layout imports when run from inside ColibriPipeline/.
    from paths import (  # type: ignore
        TELESCOPE_COLORS,
        resolve_archive_root,
        night_dir,
        normalize_obsdate,
    )
    from io import MinuteRef, discover_minutes, load_minute_full, minute_radec  # type: ignore
    from matching import GlobalMatchTable, match_minutes_to_global  # type: ignore
    from stitch import (  # type: ignore
        LightCurve,
        build_time_axis,
        gap_mask,
        bin_lightcurve,
        bin_by_count,
    )
    from catalog import (  # type: ignore
        NightCatalog,
        build_night_catalog,
        save_cache,
        load_cache,
    )

__all__ = [
    'TELESCOPE_COLORS',
    'resolve_archive_root',
    'night_dir',
    'normalize_obsdate',
    'MinuteRef',
    'discover_minutes',
    'load_minute_full',
    'minute_radec',
    'GlobalMatchTable',
    'match_minutes_to_global',
    'LightCurve',
    'build_time_axis',
    'gap_mask',
    'bin_lightcurve',
    'bin_by_count',
    'NightCatalog',
    'build_night_catalog',
    'save_cache',
    'load_cache',
]
