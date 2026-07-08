#!/usr/bin/env bash

base=${C3S}/public/data/ICDR_v3

startdate=202601   # Read 2024 onwards from c3s public directory
enddate=202701

outpath=${CCI}/validation/cdr3-pqar

for itype in drifter_cmems gtmba2
do
  ./submit_gen_mmd.py $startdate $enddate $base -i $itype -o $outpath
done
