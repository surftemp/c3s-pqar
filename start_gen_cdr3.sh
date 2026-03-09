#!/usr/bin/env bash

base=/neodc/eocis/data/global_and_regional/sea_surface_temperature/CDR_v3

enddate=202312   # Read 2024 onwards from c3s public directory

outpath=/gws/ssde/j25b/esacci_sst/validation/cdr3-pqar

# Level 3C
for itype in drifter drifter_cmems gtmba2 argo argosurf
do
  ./submit_gen_mmd.py 201501 $enddate $base --levels l3c -i $itype -o $outpath
done

# Level 4
for itype in drifter_cmems gtmba2
do
  ./submit_gen_mmd.py 198001 $enddate $base/Analysis/L4 -i $itype -o $outpath
done
