# Explainable AI for Data-Driven Design of High-Dimensional Predictive Studies
This code repository can be used to replicate numerical experiments performed on open data sets (GBSG2 and ACT).

## One-click Test
Run the following to install necessary packages and execute a demo in a single step:
```
sh setup-run.sh
```

## Manual Approach
Run the following steps in your command line:
```
conda env create -f environment.yml
```

```
conda activate xai-id
```

```
python demo_gbsg2.py
python demo_act.py
```

## Settings and outputs
All analysis settings (split, seeds, forest hyperparameters, bootstrap replicates, subgroup margins, thresholds and calibration horizons) are defined once in [config.yml](./config.yml). Each demo run writes the settings it used, with package versions, to `results/<dataset>_run_config.json`, and the strata thresholds with their stratum sizes to `results/<dataset>_strata_thresholds.csv`. Plots are saved to `plots/`.

## Interactive Test
Once you have installed necessary packages (see steps above), you can also try an interactive [demo](./demo.ipynb).
