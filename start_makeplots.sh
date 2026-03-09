#!/usr/bin/env bash

./add_dist2land.py mmd

./c3s_pvir_plots.py config/cdr3.yaml --title="{region}Level 4 - {sirds_name}"
./c3s_pvir_plots.py config/cdr3.yaml --files mmd/l3c/AVHRRMTB/*drifter_cmems*nc
./c3s_pvir_plots.py config/cdr3.yaml --files mmd/l3c/SLSTRA/*drifter_cmems*nc
./c3s_pvir_plots.py config/cdr3.yaml --files mmd/l3c/SLSTRB/*drifter_cmems*nc
