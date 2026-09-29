#!/bin/sh
# Create the pinned environment and run the three demos.
set -e

conda env create -f environment.yml

conda run --no-capture-output -n xai-id python demo_gbsg2.py
conda run --no-capture-output -n xai-id python demo_act.py
conda run --no-capture-output -n xai-id python demo_peak.py
