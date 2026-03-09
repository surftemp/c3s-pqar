#!/usr/bin/env python

import argparse
import datetime
import glob
import os
import pathlib
import sys
import subprocess

import pandas as pd
import pyjob

# List of known products and available date range
known_products = {
    'AVHRR06_G': ('197906', '198203'),
    'AVHRR07_G': ('198108', '198502'),
    'AVHRR08_G': ('198305', '198510'),
    'AVHRR09_G': ('198501', '199201'),
    'AVHRR10_G': ('198611', '199109'),
    'AVHRR11_G': ('198809', '199503'),
    'AVHRR12_G': ('199108', '199812'),
    'AVHRR14_G': ('199501', '200210'),
    'AVHRR15_G': ('199805', '201012'),
    'AVHRR16_G': ('200010', '201012'),
    'AVHRR17_G': ('200206', '201012'),
    'AVHRR18_G': ('200505', None),
    'AVHRR19_G': ('200902', None),
    'AVHRRMTA_G': ('200610', '202112'),
    'AVHRRMTB_G': ('201301', None),
    'AVHRRMTC_G': ('201811', None),
    'AVHRRMTA': ('200610', '202112'),
    'AVHRRMTB': ('201301', None),
    'AVHRRMTC': ('201811', None),
    'ATSR1': ('199108', '199712'),
    'ATSR2': ('199506', '200801'),
    'AATSR': ('200205', '201204'),
    'SLSTRA': ('201605', None),
    'SLSTRB': ('201805', None),
    'OSTIA_CMEMS': ('198201', '202206'),  # OSITA Reprocessing
    'l4': ('197907', None),               # Special case for l4 (internal)
    'L4': ('198001', None),               # Special case for l4 (archived)
    }


def tidyup_job(command):
    """Removes certain characters from job description.
    Args:
        command (list): Command line arguments
    Returns:
        command (dateTimeObject): Same as input without unwanted characters.
    """
    return str(command).replace("'", "").replace(",", "")[1:-1]


def scanpath(path, levels={'l3u', 'l3c', 'l4'}):
    """Search the supplied path for directories containing known SST-CCI
    products. To be recognised the path must:
    * contain a standard product level: l3u, l3c, or l4
    * end with a known SST-CCI product string"""
    satlist = []
    for root, dirs, files in os.walk(path, followlinks=True):
        if 'segregated' in root or 'climatology' in root or 'Climatology' in root:
            dirs[:] = []    # Ignore this directory
            continue
        # Don't walk into year/month directories
        dirs[:] = [d for d in dirs if not d.isdigit()]
        level = levels.intersection(p.casefold() for p in pathlib.Path(root).parts)
        if len(level) == 0:
            continue
        elif len(level) == 1:
            prod = os.path.basename(root)
            if prod in known_products:
                satlist.append((root, level.pop(), prod))
                dirs[:] = []  # Don't look any deeper
        else:
            print(f'Error: multiple level strings found in path {level}')
            dirs[:] = []  # Don't look any deeper
    return satlist


def datetype(string):
    """Used to check that input arguments are valid dates"""
    return pd.Period(string, 'M')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("startdate", type=datetype, help="Start date (YYYYMM)")
    parser.add_argument("enddate", type=datetype, help="End date (YYYYMM)")
    parser.add_argument("satpath", help="Location of L3 / L4 data")
    parser.add_argument("-p", "--product", help="Product code (will disable directory scan)")
    parser.add_argument("-l", "--level", default='l4', help="Product level (if not scanning)")
    parser.add_argument("-o", "--outpath", help="Output path (required if using -p)")
    parser.add_argument("--levels", nargs='+', default=['l3c', 'l4'],
                        help="Specify product levels to allow in directory scan")
    parser.add_argument("-s", "--sirds", help="Location of SIRDS data", metavar='PATH',
                        default="/gws/nopw/j04/cds_c3s_sst/input/refdata/raw/sirds")
    parser.add_argument("-i", "--insitu", default="drifter_cmems",
                        help="Type of insitu observation to use (default drifter_cmems)")
    parser.add_argument("-n", "--dry-run", action='store_true',
                        help="Perform a trial run without submitting any jobs")
    parser.add_argument("--interactive", action='store_true',
                        help="Run tasks immediately (default is to submit to batch system)")
    args = parser.parse_args()

    if args.product and not args.outpath:
        print('submit_gen_mmd.py: error: outpath is required when specifing explicit product')
        sys.exit(1)

    sirdspath = args.sirds
    sirdsname = 'SSTCCI2_refdata_{type}_{month}.nc'

    # Check and see if SIRDs path and files exist.
    if not os.path.exists(sirdspath):
        print(f"ERROR! SIRDS directory does not exist: {sirdspath}")
        sys.exit(1)

    sirds_files = glob.glob(os.path.join(
                            sirdspath,
                            sirdsname.format(month="??????", type=args.insitu)))
    if not sirds_files:
        print(f"ERROR! No SIRDS files for data type: {args.insitu} found in directory: {sirdspath}")
        sys.exit(1)


    if not args.interactive:
        # Create a directory for the batch system logs
        today = datetime.datetime.now()
        logpath = f'./log/{today:%Y/%m/%d}'
        print(f'Creating log directory: {logpath}')
        if not args.dry_run:
            os.makedirs(logpath, exist_ok=True)

    if args.product:
        satlist = [(args.satpath, args.level, args.product)]
    else:
        satlist = scanpath(args.satpath, levels=set(args.levels))
        if not satlist:
            print(f"ERROR! Exiting - no data folders found in directory: {args.satpath}\n"
                  f"for levels: {args.levels}")
            sys.exit(1)

    for path, lev, prod in satlist:
        print(f'Found {path} {lev} {prod}')

    for month in pd.period_range(args.startdate, args.enddate, freq='M'):
        month = month.to_timestamp()
        sirdsfile = os.path.join(sirdspath, sirdsname.format(month=month.strftime("%Y%m"), type=args.insitu))
        if not os.path.exists(sirdsfile):
            continue
        print(f'Insitu file: {sirdsfile}')

        for path, lev, prod in satlist:
            try:
                sat_range = [d or datetime.datetime.today() for d in known_products[prod]]
                if month not in pd.period_range(*sat_range, freq='M'):
                    continue
            except KeyError:
                pass

            print(f'  {lev} {prod}')
            if args.outpath:
                outpath = os.path.join(args.outpath, 'mmd', lev, prod)
            else:
                outpath = os.path.join(path.split(lev)[0], 'mmd', lev, prod)

            if not args.dry_run:
                os.makedirs(outpath, exist_ok=True)

            # Specify processing code depending on dataset.
            cmd = f'./gen_sst_mmd.py {sirdsfile} {path} -d={outpath}'

            if lev == 'l4' and args.insitu not in ['drifter', 'drifter_cmems', 'gtmba', 'gtmba2', 'mooring']:
                cmd += ' --keepall'

            # Specify job command depending on which run mode the code is using.
            if args.interactive:
                if args.dry_run:
                    print(cmd)
                else:
                    subprocess.run(cmd, shell=True)
            else:
                jobopts = {
                    'runtime': '02:00',
                    'memlimit': '2000',
                    'name': f'md_gen_{args.insitu}',
                    'logpath': logpath,
                    'logname': f'md_gen_{month:%Y%m}_{prod}_{lev}_{args.insitu}_{{jobid}}'
                    }
                j = pyjob.Job(cmd, options=jobopts)
                pyjob.cluster.submit(j, args.dry_run)
