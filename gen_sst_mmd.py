#!/usr/bin/env python3
"""
Script to generate L3+ SST validation MD files by simple direct matching
between the raw SIRDS files and CCI/C3S L3C/L4 SST products.
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

SIRDS_COLLECTION = {
    1: 'ICOADS2.5.1',
    2: 'EN4.2.0',
    3: 'MetDB',
    4: 'EN4.2.1',
    5: 'PMEL_GTMBA.2016',
    6: 'Unknown',
    7: 'CMEMS',
    8: 'EN4.2.2',
    9: 'ICOADS3.0.2',
    10: 'PMEL_GTMBA',
    }


def arraytostring(da):
    """
    Simple wrapper to convert character array to a string.

    This would normally be done automatically by xarray, but as we need to
    disable decode_cf we need to convert strings manually.
    """
    dtype = da.dtype.kind + str(da.shape[-1])
    out = xr.DataArray(da.values.view(dtype)[:, 0], dims=da.dims[:1])
    out.attrs = da.attrs
    return out


def read_sirds(filename):
    """
    Read a SIRDS in situ file
    """
    # Disable decoding as we don't want to convert everything to float64
    ds = xr.open_dataset(filename, decode_cf=False)
    # Add some useful metadata
    ds.attrs['source_file'] = os.path.basename(filename)
    ds.attrs['sirds_type'] = os.path.basename(filename)[16:-10]
    ds.COLLECTION.attrs['flag_values'] = np.array(list(SIRDS_COLLECTION), 'b')
    ds.COLLECTION.attrs['flag_meanings'] = ' '.join(SIRDS_COLLECTION.values())
    # Convert to standard datetime
    date = ['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND']
    ds['time'] = pd.to_datetime(ds[date].astype(int).to_dataframe(), errors='coerce')
    ds = ds.drop_vars(date)
    # Convert platform ID to a string
    ds['PLAT_ID'] = arraytostring(ds.PLAT_ID)
    # Rename variables to lower case and N_OBS to matchup
    ds = ds.rename({v: 'ins_'+v.lower() for v in ds}).rename(N_OBS='matchup')
    # Fix some CF attributes
    ds.ins_plat_id.attrs.update(standard_name='platform_id')
    ds.ins_longitude.attrs.update(standard_name='longitude')
    ds.ins_latitude.attrs.update(standard_name='latitude')
    ds.ins_depth.attrs.update(standard_name='depth')
    ds.ins_sst.attrs.update(standard_name='sea_water_temperature', units='celsius', units_metadata='temperature: on_scale')
    ds.ins_sst_type_corr.attrs.update(standard_name='sea_water_temperature_difference', units='celsius', units_metadata='temperature: difference')
    ds.ins_sst_plat_corr.attrs.update(standard_name='sea_water_temperature_difference', units='celsius', units_metadata='temperature: difference')
    ds.ins_sst_type_corr_unc.attrs.update(units='celsius', units_metadata='temperature: difference')
    ds.ins_sst_plat_corr_unc.attrs.update(units='celsius', units_metadata='temperature: difference')
    ds.ins_sst_rand_unc.attrs.update(units='celsius', units_metadata='temperature: difference')
    ds.ins_sst_comb_unc.attrs.update(units='celsius', units_metadata='temperature: difference')

    # Create a common quality_flag
    ds['ins_qc'] = ds.ins_qc1
    ds.ins_qc.attrs = {
        'long_name': 'In situ quality flag',
        'standard_name': 'quality_flag',
        'flag_values': np.array([0, 1], 'b'),
        'flag_meanings': 'pass fail',
        'comment': '',
        }
    pmel = ds.ins_collection.isin([5, 10]).values
    if pmel.any():
        # pmel qcs 1,2,3 considered pass (0), other values fail (1)
        newqc = xr.where((ds.ins_qc1 > 0) & (ds.ins_qc1 < 4), 0, 1)
        # And replace pmel values in ins_qc
        ds.ins_qc.loc[{'matchup': pmel}] = newqc.isel(matchup=pmel)
    # Need to manually apply masking to any floating point variables as xarray
    # is inconsistent in how SIRDS files are read (some have -9999 while others
    # have NaNs).
    ds = ds.load()
    for var in ds:
        fv = ds[var].attrs.get('_FillValue')
        if fv and ds[var].dtype.kind == 'f':
            ds[var][ds[var] == fv] = np.nan
            del ds[var].attrs['_FillValue']
    return ds


def read_ghrsst(filename, depth=True, full_uncertainties=True, skip_empty=True):
    """
    Read CCI/C3S L3 SST file
    """
    ignore = ['lat_bnds', 'lon_bnds', 'time_bnds', 'sses_bias']
    ds = xr.open_dataset(filename, decode_cf=False)
    # If the gbcs processor flagged this as empty then skip now
    if skip_empty and ds.attrs.get('file_quality_reason') == 'no data':
        return

    ds.attrs['product'] = ds.attrs['id'].split('-')[0]
    if 'gbcs' in ds.history.casefold():
        ds.attrs['gbcsver'] = ds.attrs['history'].split()[-1]
    else:
        ds.attrs['gbcsver'] = 'None'

    # Make sure we have processing_level attribute
    level = ds.attrs.get('processing_level')
    if not level:
        level = os.path.basename(filename).split('-')[2].replace('_GHRSST','')
        ds.attrs['processing_level'] = level

    # Level 4 files
    rename = {'analysis_error': 'unc',              # GDS
              'analysis_uncertainty': 'unc',        # CDR 2.0
              'analysed_sst_uncertainty': 'unc',    # CDR 2.1+
              'analysed_sst': 'sst',
              'sea_ice_fraction': 'ice',
              }

    # Level 3 files
    if depth:
        rename.update(
            {'sea_surface_temperature_depth': 'sst',
             'sst_depth_total_uncertainty': 'unc',  # CDR 2.0
             'sea_surface_temperature_depth_total_uncertainty': 'unc', # CDR2.1+
             'sst_depth_dtime': 'dtime',
            })
    else:
        rename.update(
            {'sea_surface_temperature': 'sst',
             'sst_dtime': 'dtime',
            })
        if 'sea_surface_temperature_total_uncertainty' in ds:
            rename['sea_surface_temperature_total_uncertainty'] = 'unc'
        else:
            rename['sses_standard_deviation'] = 'unc'

    ignore += ['sea_surface_temperature',
               'sea_surface_temperature_depth',
               'sea_surface_temperature_depth_anomaly',
               'sst_dtime',
               'sst_depth_dtime',
               'sses_standard_deviation',
               'sea_surface_temperature_total_uncertainty',
               'sst_depth_total_uncertainty',
               'sea_surface_temperature_depth_total_uncertainty',
               'depth_adjustment',
               'adjustment_alt',
               'sea_surface_temperature_retrieval_type',
               'alt_sst_retrieval_type',
               ]

    if not full_uncertainties:
        ignore += ['large_scale_correlated_uncertainty',
                   'synoptically_correlated_uncertainty',
                   'uncorrelated_uncertainty',
                   'adjustment_uncertainty',
                   'uncertainty_random',
                   'uncertainty_correlated',
                   'uncertainty_systematic',
                   'uncertainty_correlated_time_and_depth_adjustment',
                   'uncertainty_random_alt',
                   'uncertainty_correlated_alt',
                   'uncertainty_systematic_alt',
                   ]
    # Rename variables
    ds = ds.rename(dict(k for k in rename.items() if k[0] in ds))
    ds['time'] = xr.decode_cf(ds).time
    # Drop bounds variables and time dimension
    ds = ds.drop_vars([k for k in ignore if k in ds]).isel(time=0)
    for var in ds:
        try:
            del ds[var].attrs['coordinates']
        except KeyError:
            pass
    ds = ds.rename({v: 'sat_'+v for v in ds.variables})
    return ds


def earthdistance(lon1, lat1, lon2, lat2):
    """Calculate great circle distance between two points."""
    rlat1 = np.deg2rad(lat1)
    rlat2 = np.deg2rad(lat2)
    dlon = np.deg2rad(lon1-lon2)
    a = 6378.1370
    b = 6356.7523
    erad = np.sqrt(((a**2 * np.cos(rlat1))**2 + (b**2 * np.sin(rlat1))**2)
                   / ((a * np.cos(rlat1))**2 + (b * np.sin(rlat1))**2))
    c1 = np.sin((rlat1-rlat2) / 2)**2
    c2 = np.cos(rlat1) * np.cos(rlat2)
    return erad*2*np.arcsin(np.sqrt(c1 + c2 * np.sin(dlon/2)**2))


def calc_ll_inds(sirds, sdata):
    """Calculate the lat/lon indices"""
    nlon = len(sdata.sat_lon)
    nlat = len(sdata.sat_lat)
    assert nlon % 360 == 0, f"Unexpected longitude size: {nlon}"
    n = nlon // 360
    sirds['ilat'] = (n * (sirds.ins_latitude + 90)).astype(int).clip(0,nlat-1)
    sirds['ilon'] = (n * (sirds.ins_longitude + 180)).astype(int) % nlon


def find_matches_l3(sirds, sdata):
    """
    Find all the matches between given insitu and L3 SST data
    """
    tlimit = np.timedelta64(2, 'h')
    t6hour = np.timedelta64(6, 'h')  # Extra margin for DV adjustment
    t0 = pd.to_datetime(sdata.start_time).to_datetime64() - tlimit - t6hour
    t1 = pd.to_datetime(sdata.stop_time).to_datetime64() + tlimit + t6hour
    # Extract SIRDS data overlapping with the satellite file
    idata = sirds.isel(matchup=(sirds.ins_time >= t0) & (sirds.ins_time < t1))
    # If there are no possible matches with this file then abort now
    if len(idata.matchup) == 0:
        return
    # Extract SIRDS data corresponding to valid satellite data
    ql = sdata.sat_quality_level.load()[idata.ilat, idata.ilon]
    # If there are no possible matches then abort before reading the rest of the L3 file
    if ql.max() <= 1:
        return
    idata = idata.isel(matchup=ql > 1)
    # We need the dtime variables for matching, do copy now
    vars = ['sat_quality_level', 'sat_dtime']
    sdata.sat_dtime.load()
    idata = idata.merge(sdata[vars].isel(sat_lat=idata.ilat,
                                         sat_lon=idata.ilon))
    sdata = sdata.drop_vars(vars).load()
    # Loop over each unique callsign and find best match
    matches = []
    for cs in np.unique(idata.ins_plat_id):
        ds = idata.isel(matchup=idata.ins_plat_id == cs)
        tdiff = np.abs(sdata.sat_time
                       + ds.sat_dtime.astype('timedelta64[s]')
                       - ds.ins_time)
        imin = tdiff.argmin()
        if tdiff[imin] < tlimit:
            matches.append(ds.isel(matchup=imin))
    if not matches:
        return
    matches = xr.concat(matches, 'matchup')
    # Now add the satellite data (faster than doing it in the loop above)
    matches = matches.merge(sdata.isel(sat_lat=matches.ilat,
                                       sat_lon=matches.ilon))
    # Finalise output
    sat_time = matches.sat_time + matches.sat_dtime.astype('timedelta64[s]')
    matches = matches.drop_vars(['sat_time', 'sat_dtime', 'ilat', 'ilon'])
    matches['sat_time'] = sat_time
    matches['file_quality_level'] = 'matchup', np.full(len(matches.matchup), sdata.file_quality_level, 'i1')
    matches.attrs.update(
        sat_source = sdata.attrs.get('id', ''),
        institution = sdata.institution,
        spatial_resolution = sdata.spatial_resolution
    )
    return matches


def find_matches_l4(sirds, sdata, dailyavg=True):
    """
    Find all the matches between given insitu and L4 SST data
    """
    t0 = pd.to_datetime(sdata.start_time).to_datetime64()
    t1 = pd.to_datetime(sdata.stop_time).to_datetime64()
    # Extract SIRDS data overlapping with the satellite file
    idata = sirds.isel(matchup=(sirds.ins_time >= t0) & (sirds.ins_time < t1))
    # If there are no possible matches with this file then abort now
    if len(idata.matchup) == 0:
        return
    # Load the satellite data
    sdata = sdata.load()
    if dailyavg:
        # Floating point variables
        fvars = [v for v in idata if idata[v].dtype.kind == 'f']
        # Loop over each unique callsign and find best match
        matches = []
        for cs in np.unique(idata.ins_plat_id):
            ds = idata.isel(matchup=idata.ins_plat_id == cs).sortby('ins_time')
            m = ds.isel(matchup=0)
            m['ins_nobs'] = len(ds.matchup)
            if len(ds.matchup) > 1:
                m.update(ds[fvars].mean())
                # Must calculate time separately as it is a datetime
                m['ins_time'] = ds.ins_time.mean()
                dist = earthdistance(ds.ins_longitude[0], ds.ins_latitude[0], ds.ins_longitude[-1], ds.ins_latitude[-1])
                time = (ds.ins_time[-1] - ds.ins_time[0]).astype(float) / 3.6e12
                m['ins_speed'] = dist/time
                m['ins_movement'] = dist
                # Check if longitude has wrapped
                if np.ptp(ds.ins_longitude.values) > 180:
                    lon = (ds.ins_longitude % 360).mean()
                    if lon > 180:
                        lon = lon - 360
                    m['ins_longitude'] = lon
            else:
                m['ins_speed'] = np.nan
                m['ins_movement'] = 0
            matches.append(m)
        if not matches:
            return
        matches = xr.concat(matches, 'matchup')
        calc_ll_inds(matches, sdata)
    else:
        matches = idata
        matches['ins_speed'] = matches.ins_latitude * 0

    matches.ins_nobs.attrs.update(
        long_name='Number of observations used in daily average',
        standard_name='number_of_observations',
        units='1',
        )
    matches.ins_speed.attrs.update(
        long_name='Platform speed',
        standard_name='platform_speed_wrt_ground',
        units='km hr-1',
        comment='Based on first and last report in this day',
        )
    matches.ins_movement.attrs.update(
        long_name='Platform movement',
        units='km',
        comment='Based on first and last report in this day',
        )

    # Now add the satellite data (faster than doing it in the loop above)
    matches = matches.merge(sdata.isel(sat_lat=matches.ilat,
                                       sat_lon=matches.ilon))
    # Finalise output
    matches = matches.drop_vars(['ilat', 'ilon'])
    matches['file_quality_level'] = 'matchup', np.full(len(matches.matchup), sdata.file_quality_level, 'i1')
    matches.attrs.update(
        sat_source = sdata.attrs.get('id', ''),
        institution = sdata.institution,
        spatial_resolution = sdata.spatial_resolution
    )
    return matches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sirds", help="SIRDS file")
    parser.add_argument("satpath", help="Path to L3 data")
    parser.add_argument("-d", dest='outpath', metavar='DIR', default='./',
                        help="Output directory")
    parser.add_argument("--keepall", dest='dailyavg', action='store_false',
                        help='Keep all observations instead of daily average (l4 only)')
    parser.add_argument('--no-qc1', dest='filter_qc1', action='store_false',
                        help='Disable QC1 filtering (basic QC)')
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG,
                        datefmt='%y-%m-%d %H:%M',
                        stream=sys.stdout,
                        format='%(asctime)s %(levelname)-8s %(message)s')

    logging.info(f'Processing SIRDS: {args.sirds}')
    idata = read_sirds(args.sirds)
    logging.info(f'Read {len(idata.matchup)} matches')
    # Only use data which has passed QC checks
    if args.filter_qc1:
        idata = idata.isel(matchup=idata.ins_qc == 0)
    logging.info(f'Processing dir: {args.satpath}')
    datestr = os.path.basename(args.sirds)[-9:-3]
    matches = []
    product = None
    gbcsver = None
    for root, dirs, files in os.walk(args.satpath, followlinks=True):
        dirs.sort()
        for f in files:
            if not f.startswith(datestr):
                continue
            logging.info(f'Reading: {f}')
            try:
                sdata = read_ghrsst(os.path.join(root, f))
                if not sdata:
                    continue
                product = product or sdata.attrs['product']
                gbcsver = gbcsver or sdata.attrs['gbcsver']
                # For first file need to calc lat/lon indices
                calc_ll_inds(idata, sdata)
                logging.debug('Matching')
                if sdata.processing_level.startswith('L3'):
                    m = find_matches_l3(idata, sdata)
                else:
                    m = find_matches_l4(idata, sdata, dailyavg=args.dailyavg)
            except Exception as e:
                logging.error(e)
            else:
                if m:
                    logging.debug(f'Found {len(m.matchup)} matches')
                    matches.append(m)
                else:
                    logging.info('No matches found')
            try:
                sdata.close()
                del sdata
            except NameError:
                pass

    if matches:
        matches = xr.concat(matches, 'matchup')
        # Final modifications to output data
        matches.attrs['product'] = product
        matches.attrs['gbcsver'] = gbcsver
        matches['matchup.tdiff'] = matches.sat_time - matches.ins_time
        matches['matchup.tdiff'].attrs['comment'] = "Satellite - in situ time difference"
        # Now write it
        logging.info(f'Found {len(matches.matchup)} matches total')
        fname = os.path.basename(args.sirds).replace('SSTCCI2_refdata', product)
        os.makedirs(args.outpath, exist_ok=True)
        matches.to_netcdf(os.path.join(args.outpath, fname))
        logging.info(f'Wrote {fname}')
    else:
        logging.info('No matches found')
    print('DONE')
