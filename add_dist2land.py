#!/usr/bin/env python

import os
import argparse
import netCDF4


lutfile = 'Distance_to_land-GloboLakes-ESACCI_LC-5km.nc'


def read_lut(lutfile):
    """Read the distance to land LUT"""
    print(f"Reading {lutfile}")
    with netCDF4.Dataset(lutfile) as nc:
        nc.set_auto_mask(False)
        dlut = nc.variables['distance_to_land'][:]
        dlat = nc.variables['lat'][:]
        dlon = nc.variables['lon'][:]
    return dlut, dlat, dlon


def add_distance(filename, dlut):
    """Add distance to land data to an existing MD file"""
    print(f"Processing {filename}")
    with netCDF4.Dataset(filename, 'a') as nc:
        lat = nc.variables['sat_lat'][:]
        lon = nc.variables['sat_lon'][:]
        ilat = (20 * (90 - lat)).astype(int).clip(0, 3599)
        ilon = (20 * (180 + lon)).astype(int) % 7200
        if 'sat_land' in nc.variables:
            dist = nc.variables['sat_land']
        else:
            dist = nc.createVariable('sat_land', 'f4', nc.variables['sat_lat'].dimensions)
            dist.long_name = 'Distance to land'
            dist.units = 'km'
        dist[:] = dlut[ilat, ilon]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='MMD directory to process')
    args = parser.parse_args()

    dlut, dlat, dlon = read_lut(lutfile)
    for root, dirs, files in os.walk(args.path):
        for f in files:
            add_distance(os.path.join(root, f), dlut)
