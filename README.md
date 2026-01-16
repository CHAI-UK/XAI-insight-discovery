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

## Interactive Test
Once you have installed necessary packages (see steps above), you can also try an interactive [demo](./demo.ipynb).
