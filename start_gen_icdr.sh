#!/usr/bin/env bash

base=/gws/nopw/j04/cds_c3s_sst/public/data/ICDR_v3

startdate=202501   # Read 2024 onwards from c3s public directory
enddate=202701

outpath=/gws/ssde/j25b/esacci_sst/validation/cdr3-pqar

for itype in drifter_cmems gtmba2
do
  ./submit_gen_mmd.py $startdate $enddate $base -i $itype -o $outpath
done
