#!/usr/bin/env python3
"""
Script to generate L4 SST validation MD files by simple matching between the
raw SIRDS files and CCI/C3S L4 SST products.
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr


def arraytostring(da):
    """
    Simple wrapper to convert character array to a string.

    This would normally be done automatically by xarray, but as we need to
    diable decode_cf we need to convert strings manually.
    """
    dtype = da.dtype.kind + str(da.shape[-1])
    out = xr.DataArray(da.data.view(dtype)[:, 0], dims=da.dims[:1])
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
    ds.COLLECTION.attrs['flag_values'] = np.array([1, 2, 3, 4, 5, 6, 7], 'b')
    ds.COLLECTION.attrs['flag_meanings'] = (
        'ICOADS EN4.2.0 MetDB EN4.2.1 PMEL_GTMBA Unknown CMEMS')
    # Convert to standard datetime
    date = ['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND']
    ds['time'] = pd.to_datetime(ds[date].astype(int).to_dataframe(), errors='coerce')
    ds = ds.drop(date)
    # Convert platform ID to a string
    ds['PLAT_ID'] = arraytostring(ds.PLAT_ID)
    # Rename variables to lower case and N_OBS to matchup
    ds = ds.rename({v: 'ins_'+v.lower() for v in ds}).rename(N_OBS='matchup')
    # Precalculate lat/lon indices
    ds['ilat'] = (20 * (ds.ins_latitude + 90)).astype(int).clip(0,3599)
    ds['ilon'] = (20 * (ds.ins_longitude + 180)).astype(int) % 7200
    pmel = (ds.ins_collection == 5).data
    if pmel.any():
        # Copy PMEL flag to ins_qc2 (overwrite the position qc)
        ds.ins_qc2.loc[{'matchup': pmel}] = ds.ins_qc1.isel(matchup=pmel)
        # pmel qcs 1,2,3 considered pass (0), other values fail (1)
        newqc = xr.where((ds.ins_qc1 > 0) & (ds.ins_qc1 < 4), 0, 1)
        # And replace pmel values in ins_qc1
        ds.ins_qc1.loc[{'matchup': pmel}] = newqc.isel(matchup=pmel)
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


def read_l4_cdr2(filename):
    """
    Read CCI/C3S L4 SST file
    """
    ignore = ['lat_bnds', 'lon_bnds', 'time_bnds']
    ds = xr.open_dataset(filename, decode_cf=False)
    ds.attrs['product'] = ds.attrs['id'].split('-')[0]
    ds.attrs['gbcsver'] = ds.attrs['history'].split()[-1]

    # Rename uncertainty variable to "unc"
    if 'analysis_error' in ds.data_vars:
        ds = ds.rename(analysis_error='unc')
    elif 'analysis_uncertainty' in ds.data_vars:
        ds = ds.rename(analysis_uncertainty='unc')
    elif 'analysed_sst_uncertainty' in ds.data_vars:
        ds = ds.rename(analysed_sst_uncertainty='unc')
    else:
        print("Error! Analysis variable not recognised.")

    ds = ds.rename(analysed_sst='sst',
                   sea_ice_fraction='ice')
    ds['time'] = xr.decode_cf(ds).time
    # Drop bounds variables and time dimension
    ds = ds.drop([k for k in ignore if k in ds]).isel(time=0)
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


def find_matches(sirds, sdata, dailyavg=True):
    """
    Find all the matches between given insitu and L4 SST data
    """
    t0 = pd.to_datetime(sdata.start_time).to_datetime64()
    t1 = pd.to_datetime(sdata.stop_time).to_datetime64()
    # Extract SIRDS data overlapping with the satellite file
    idata = sirds.isel(matchup=(sirds.ins_time >= t0) & (sirds.ins_time < t1))
    # Extract SIRDS data with QC=0
    idata = idata.isel(matchup=idata.ins_qc1 == 0)
    # If there are no possible matches with this file then abort now
    if len(idata.matchup) == 0:
        return
    # We need the dtime variables for matching, do copy now
    sdata = sdata.load()
    if dailyavg:
        # Floating point variables
        fvars = [v for v in idata if idata[v].dtype.kind == 'f']
        # Loop over each unique callsign and find best match
        matches = []
        for cs in np.unique(idata.ins_plat_id):
            ds = idata.isel(matchup=idata.ins_plat_id == cs).sortby('ins_time')
            m = ds.isel(matchup=0)
            if len(ds.matchup) > 1:
                m.update(ds[fvars].mean())
                dist = earthdistance(ds.ins_longitude[0], ds.ins_latitude[0], ds.ins_longitude[-1], ds.ins_latitude[-1])
                time = (ds.ins_time[-1] - ds.ins_time[0]).astype(float) / 3.6e12
                m['ins_speed'] = dist/time
                m['ins_movement'] = dist
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
        matches['ilat'] = (20 * (matches.ins_latitude + 90)).astype(int).clip(0, 3599)
        matches['ilon'] = (20 * (matches.ins_longitude + 180)).astype(int) % 7200
    else:
        matches = idata
        matches['ins_speed'] = matches.ins_latitude * 0

    # Now add the satellite data (faster than doing it in the loop above)
    matches = matches.merge(sdata.isel(sat_lat=matches.ilat,
                                       sat_lon=matches.ilon))
    # Finalise output
    matches = matches.drop(['ilat', 'ilon'])
    matches['file_quality_level'] = 'matchup', np.full(len(matches.matchup), sdata.file_quality_level, 'i1')
    return matches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sirds", help="SIRDS file")
    parser.add_argument("satpath", help="Path to L4 data")
    parser.add_argument("-d", dest='outpath', metavar='DIR', default='./',
                        help="Output directory")
    parser.add_argument("--keepall", dest='dailyavg', action='store_false',
                        help='Keep all observations instead of daily average')
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG,
                        datefmt='%y-%m-%d %H:%M',
                        stream=sys.stdout,
                        format='%(asctime)s %(levelname)-8s %(message)s')

    logging.info('Processing SIRDS: {0}'.format(args.sirds))
    idata = read_sirds(args.sirds)
    logging.info('Read {0} matches'.format(len(idata.matchup)))
    logging.info('Processing dir: {0}'.format(args.satpath))
    datestr = os.path.basename(args.sirds)[-9:-3]
    matches = []
    product = None
    gbcsver = None
    for root, dirs, files in os.walk(args.satpath, followlinks=True):
        dirs.sort()
        for f in files:
            if not f.startswith(datestr):
                continue
            logging.info('Reading: {0}'.format(f))
            try:
                sdata = read_l4_cdr2(os.path.join(root, f))
                if not sdata:
                    continue
                product = product or sdata.attrs['product']
                gbcsver = gbcsver or sdata.attrs['gbcsver']
                logging.debug('Matching')
                m = find_matches(idata, sdata, dailyavg=args.dailyavg)
            except Exception as e:
                logging.error(e)
            else:
                if m:
                    logging.debug('Found {0} matches'.format(len(m.matchup)))
                    matches.append(m)
                else:
                    logging.info('No matches found')
    if matches:
        matches = xr.concat(matches, 'matchup')
        # Final modifications to output data
        matches.attrs['product'] = product
        matches.attrs['gbcsver'] = gbcsver
        matches['matchup.tdiff'] = matches.sat_time - matches.ins_time
        matches['matchup.tdiff'].attrs['comment'] = "Satellite - in situ time difference"
        # Now write it
        logging.info('Found {0} matches total'.format(len(matches.matchup)))
        fname = os.path.basename(args.sirds).replace('SSTCCI2_refdata', product)
        os.makedirs(args.outpath, exist_ok=True)
        matches.to_netcdf(os.path.join(args.outpath, fname))
        logging.info('Wrote {0}'.format(fname))
    else:
        logging.info('No matches found')
    print('DONE')
