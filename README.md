# Explainable AI for Data-Driven Design of High-Dimensional Predictive Studies
This code repository can be used to replicate numerical experiments performed on open data sets (GBSG2, ACT and the National Wilms Tumor Study).

Each demo follows the same steps as the analysis of the main cohort, using the same code (`recommender.py` for the recommendation rules, the evaluation tools in `utils.py`):
1. An 80/20 train-test split; a random survival forest (the exploratory model) fitted on the training set.
2. Feature attributions (KernelSHAP) of the forest's log risk score in a low-risk and a high-risk subcohort, around the average patient without and with the event. The subgroup margin is chosen on the training data from feature values alone: the smallest margin at which at least 90% of the candidate pairs that can be tested at any margin are testable (enough patients in each stratum and at each level of a binary partner). The table of testable pairs by margin is saved for each data set.
3. Recommendations from the training data: features to exclude, features needing non-linear terms, and candidate interactions, screened with within-stratum contrasts (binary and nominal partners; one-hot encoded nominal features are recombined and tested with one contrast per level) and product-term regressions (continuous and ordinal partners), with false discovery rate control over unique pairs, then screened against nonlinear main effects in a reference Cox model, with the number of interaction terms limited by a sample-size criterion (Riley et al).
4. The Cox model fitted without recommendations, with each recommendation and with all of them, each evaluated once on the test set: Harrell's C with bootstrap confidence interval, Uno's C, time-dependent AUC, integrated Brier score, calibration with bootstrap intervals and calibration slope, and the paired difference in C from the model without recommendations. Two comparators are fitted: a Cox model with restricted cubic splines and a LASSO Cox model over all pairwise interactions.

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
python demo_nwtco.py
```

Results are written to `results/` (model comparison, margin testability, interaction tests, target-model tests, summary counts, final Cox coefficients and proportional hazards tests for each data set) and figures to `plots/`. Set `XAI_N_JOBS` to limit the number of parallel workers (default: all cores).

## Interactive Test
Once you have installed necessary packages (see steps above), you can also try an interactive [demo](./demo.ipynb).

## Tests
```
pytest test_recommender.py -q
```

## Data
GBSG2 and ACT are loaded from scikit-survival. `data/nwtco.csv` is the `nwtco` data set of the R package survival, exported with `write.table(survival::nwtco, "nwtco.csv", sep = ",", row.names = FALSE)`.
