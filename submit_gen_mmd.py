#!/usr/bin/env python

import argparse
import datetime
import logging
import os
import sys
import subprocess

import pandas as pd

import glob

def get_levels_and_product(data_type):
    
    if data_type == 'l3':
        levels = ['l3u', 'l3c']
        products = {'AVHRR06_G': ('197906', '198203'),
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
                    'AVHRRMTA_G': ('200610', None),
                    'AVHRRMTB_G': ('201301', None),
                    'AVHRRMTC_G': ('201811', None),
                    'AVHRRMTA': ('200610', None),
                    'AVHRRMTB': ('201301', None),
                    'AVHRRMTC': ('201811', None),
                    'ATSR1': ('199108', '199712'),
                    'ATSR2': ('199506', '200801'),
                    'AATSR': ('200205', '201204'),
                    'SLSTRA': ('201605', None),
                    'SLSTRB': ('201805', None),
                    }
    elif data_type == 'l4':
        levels = ['l4']
        products = {'OSTIA_CMEMS': ('198201', '202001'),
                    'l4': ('197907', None),
                    }
    else:
        print("Error: data_type={} not recognised. Exiting.".format(data_type))
        sys.exit(1)

    return levels, products


def tidyup_job(command):
    """Removes certain characters from job description.
    Args:
        command (list): Command line arguments
    Returns:
        command (dateTimeObject): Same as input without unwanted characters.
    """
    return str(command).replace("'", "").replace(",", "")[1:-1]


def scanpath(path, levels, products):
    satlist = []
    for root, dirs, files in os.walk(path, followlinks=True):
        # Don't walk into year/month directories
        dirs[:] = [d for d in dirs if not d.isdigit()]
        for lev in levels:
            if lev in root:
                prod = os.path.basename(root)
                if prod in products:
                    satlist.append((root, lev, prod))
    return satlist


logging.basicConfig(level=logging.DEBUG,
                    datefmt='%y-%m-%d %H:%M',
                    stream=sys.stdout,
                    format='%(asctime)s %(levelname)-8s %(message)s')


parser = argparse.ArgumentParser()
parser.add_argument("startdate", help="Start date - format: 'YYYYMM'")
parser.add_argument("enddate", help="End date - format: 'YYYYMM'")
parser.add_argument("satpath", help="Location of L3 / L4 data")
parser.add_argument("--sirdspath", help="Location of SIRDS data files", metavar='PATH',
                    default="/gws/nopw/j04/cds_c3s_sst/input/refdata/raw/sirds")
parser.add_argument("--data_type", help="Satellite data type: l3 (default) or l4",
                    default="l3", choices=['l3', 'l4'])
parser.add_argument("--itype", default="drifter_cmems",
                    help="Type of insitu observation to use (default drifter_cmems)")
parser.add_argument("--interactive", action='store_true',
                    help="Run tasks immediately (default is to submit to batch system)")
args = parser.parse_args()

sirdspath = args.sirdspath
sirdsname = 'SSTCCI2_refdata_{type}_{month}.nc'

# Print input arguments to output.
msg = f"""Running submit_gen_mmd.py. Runtime variables:
startdate={args.startdate}, enddate={args.enddate},
satpath={args.satpath}, sirdspath={sirdspath},
data_type={args.data_type}, itype={args.itype}, interactive={args.interactive}"""

logging.info(msg)

# Check and see if SIRDs path and files exist.
if not os.path.exists(sirdspath):
    print("ERROR! SIRDS dir path does not exist: {}".format(sirdspath))
    sys.exit(1)

sirds_files = glob.glob(os.path.join(
                        sirdspath,
                        sirdsname.format(month="??????", type=args.itype)))
if not sirds_files:
    print("ERROR! No SIRDS files for data type: {} found in "
          "directory: {}".format(args.itype, sirdspath))
    sys.exit(1)

levels, products = get_levels_and_product(args.data_type)

today = datetime.datetime.now()
logpath = f'./log/{today:%Y/%m/%d}'
os.makedirs(logpath, exist_ok=True)

satlist = scanpath(args.satpath, levels, products)
if not satlist:
    print("ERROR! Exiting - no data folders found in directory: {}\n"
          "for levels: {}\nand products: {}.".format(args.satpath, levels, products))
    sys.exit(1)

for path, lev, prod in satlist:
    logging.info(f'Found {path}')

for month in pd.period_range(args.startdate, args.enddate, freq='M'):
    month = month.to_timestamp()
    sirdsfile = os.path.join(sirdspath, sirdsname.format(month=month.strftime("%Y%m"), type=args.itype))
    if not os.path.exists(sirdsfile):
        continue
    logging.info(f'Insitu file: {sirdsfile}')

    for path, lev, prod in satlist:
        sat_range = [d or datetime.datetime.today() for d in products[prod]]
        if month not in pd.period_range(*sat_range, freq='M'):
            continue
        outpath = os.path.join(path.split(lev)[0], 'mmd', lev, prod)
        os.makedirs(outpath, exist_ok=True)

        # Specify processing code depending on dataset.
        if lev == 'l4':
            pycommand = f'./gen_l4_mmd.py -d={outpath} {sirdsfile} {path}'
            if args.itype not in ['drifter', 'drifter_cmems', 'gtmba', 'mooring']:
                pycommand += ' --keepall'
        else:
            pycommand = f'./gen_sst_mmd.py -d={outpath} {sirdsfile} {path}'

        logging.debug(pycommand)

        # Specify job command depending on which run mode the code is using.
        if args.interactive:
            job_command = pycommand
        else:
            job = [
                'sbatch',
                '--account=short4hr',
                '-p', 'short-serial-4hr',
                '-t', '2:00:00',
                '-o', os.path.join(logpath, f'md_gen_{month:%Y%m}_{prod}_{lev}_{args.itype}_%j.out'),
                '-e', os.path.join(logpath, f'md_gen_{month:%Y%m}_{prod}_{lev}_{args.itype}_%j.err'),
                '--mem=2000',
                f'--job-name=gen_mmd_{args.itype}',
                f'--wrap="{pycommand}"']
            job_command = tidyup_job(job)

        result = subprocess.check_output(job_command, shell=True)
        logging.debug(result)
