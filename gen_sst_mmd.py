#!/usr/bin/env python3
"""
Script to generate L3+ SST validation MD files by simple matching between the
raw SIRDS files and CCI/C3S L3C/L4 SST products.
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
    return ds.load()


def read_l3_cdr2(filename, depth=True, full_uncertainties=True, skip_empty=True):
    """
    Read CCI/C3S L3 SST file
    """
    ignore = ['lat_bnds', 'lon_bnds', 'time_bnds', 'sses_bias']
    ds = xr.open_dataset(filename, decode_cf=False)
    try:
        # If the gbcs processor flagged this as empty then skip now
        if skip_empty and ds.attrs['file_quality_reason'] == 'no data':
            return
    except KeyError:
        pass
    ds.attrs['product'] = ds.attrs['id'].split('-')[0]
    ds.attrs['gbcsver'] = ds.attrs['history'].split()[-1]
    if depth:
        rename = {'sea_surface_temperature_depth': 'sst',
                  'sst_depth_total_uncertainty': 'unc',  # CDR 2.0
                  'sea_surface_temperature_depth_total_uncertainty': 'unc', # CDR2.1+
                  'sst_depth_dtime': 'dtime',
                  }
    else:
        rename = {'sea_surface_temperature': 'sst',
                  'sst_dtime': 'dtime',
                  }
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
    ds = ds.drop([k for k in ignore if k in ds]).isel(time=0)
    for var in ds:
        try:
            del ds[var].attrs['coordinates']
        except KeyError:
            pass
    ds = ds.rename({v: 'sat_'+v for v in ds.variables})
    return ds


def find_matches(sirds, sdata):
    """
    Find all the matches between given insitu and L3 SST data
    """
    tlimit = np.timedelta64(2, 'h')
    t6hour = np.timedelta64(6, 'h')  # Extra margin for DV adjustment
    t0 = pd.to_datetime(sdata.start_time).to_datetime64() - tlimit - t6hour
    t1 = pd.to_datetime(sdata.stop_time).to_datetime64() + tlimit + t6hour
    # Extract SIRDS data overlapping with the satellite file
    idata = sirds.isel(matchup=(sirds.ins_time > t0) & (sirds.ins_time < t1))
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
    sdata = sdata.drop(vars).load()
    # Loop over each unique callsign and find best match
    matches = []
    for cs in np.unique(idata.ins_plat_id):
        ds = idata.isel(matchup=idata.ins_plat_id == cs)
        tdiff = np.abs(+ sdata.sat_time
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
    matches = matches.drop(['sat_time', 'sat_dtime', 'ilat', 'ilon'])
    matches['sat_time'] = sat_time
    matches['file_quality_level'] = 'matchup', np.full(len(matches.matchup), sdata.file_quality_level, 'i1')
    return matches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sirds", help="SIRDS file")
    parser.add_argument("satpath", help="Path to L3 data")
    parser.add_argument("-d", dest='outpath', metavar='DIR', default='./',
                        help="Output directory")
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
                sdata = read_l3_cdr2(os.path.join(root, f))
                if not sdata:
                    continue
                product = product or sdata.attrs['product']
                gbcsver = gbcsver or sdata.attrs['gbcsver']
                logging.debug('Matching')
                m = find_matches(idata, sdata)
            except Exception as e:
                logging.error(e)
            else:
                if m:
                    logging.debug('Found {0} matches'.format(len(m.matchup)))
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
        logging.info('Found {0} matches total'.format(len(matches.matchup)))
        fname = os.path.basename(args.sirds).replace('SSTCCI2_refdata', product)
        os.makedirs(args.outpath, exist_ok=True)
        matches.to_netcdf(os.path.join(args.outpath, fname))
        logging.info('Wrote {0}'.format(fname))
    else:
        logging.info('No matches found')
    print('DONE')
