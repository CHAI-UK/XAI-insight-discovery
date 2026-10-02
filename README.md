# Explainable AI to Guide the Design of Interpretable Cox Models, Applied to Falls and Related Injuries in Adults Aged 50 Years or Older: Algorithm Development and Validation
This code repository can be used to replicate the numerical experiments performed on open data sets (GBSG2, ACT and peakVO2).

The recommendation rules (`recommender.py`) and the evaluation tools (`utils.py`) are the same code as used for the analysis of the main cohort (DataLoch), and each demo calls them through the same entry points (`recommend` and `evaluate_recommendations`). `public_analyses.py` adds outputs for the open data sets without changing that code: `run_recommend` calls `recommend` unchanged and also saves the tables it computes but does not return, `final_cox_table` adds hazard ratios with confidence intervals, and `write_run_record` saves the settings of each run. `public_extras.py` holds additional analyses requested in review, run on the open data sets only (see below). The data processing specific to DataLoch is not released, for data security reasons. The version of the code that produced the results in the paper is tagged `dataloch-run`.

Each demo follows the same steps:
1. An 80/20 train-test split; a random survival forest (the exploratory model) fitted on the training set.
2. Feature attributions (KernelSHAP) of the forest's log risk score in a low-risk and a high-risk subcohort: training patients whose log risk lies within a margin of that of the average patient without and with the event. The margin is chosen on the training data from a grid (0.05, 0.1, 0.2, 0.3, 0.5 and 1.0 times the SD of the log risk). It is the smaller of the first two consecutive margins at which each subcohort has at least 100 patients, every feature that needs an interaction screen can be screened, and the exclusion, non-linearity and interaction recommendations agree (Jaccard similarity of at least 0.9 for each). If no two margins agree, the largest margin that could be evaluated is used and the run is flagged as unstable.
3. Recommendations from the training data: features to exclude, features needing non-linear terms, and candidate interactions. Interactions are screened with within-stratum contrasts (binary and nominal partners; one-hot encoded nominal features are recombined and tested with one contrast per level) and product-term regressions (continuous and ordinal partners). False discovery rate control is applied over unique pairs, and the surviving pairs are then tested against nonlinear main effects in a reference Cox model, with the number of interaction terms limited by a sample-size criterion (Riley et al).
4. The Cox model is fitted without recommendations, with each recommendation and with all of them, and each is evaluated once on the test set. Metrics: Harrell's C with bootstrap confidence interval, Uno's C, time-dependent AUC, integrated Brier score, calibration with bootstrap intervals and calibration slope, and the paired difference in C from the model without recommendations. Two comparators are fitted: a Cox model with restricted cubic splines and a LASSO Cox model over all pairwise interactions.
5. Additional analyses on the open data sets only (`public_extras.run_extras`; not part of the DataLoch analysis). They call the shared code unchanged and do not alter steps 1–4:
   - an interaction sensitivity model: the final model plus every pair confirmed in the reference Cox model, ignoring the parameter budget;
   - calibration-in-the-large at t0 (log observed/expected events);
   - global and per-term Schoenfeld tests, with residual plots, for the models without and with recommendations;
   - events per split, subcohort and parameter;
   - a bootstrap of the final model's coefficients;
   - bootstrap selection frequencies of each recommendation, resampling the two subcohorts with the forest and attributions fixed;
   - an ablation over every subset and ordering of the three recommendation types;
   - the rules re-run on attributions of the predicted risk by t0, 1 − S(t0), instead of the log risk score.

## One-click Test
Run the following to create the pinned environment and execute the three demos in a single step:
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
python demo_peak.py
```

The environment pins Python 3.12 and every package version (`requirements.txt`); `pip install -r requirements.txt` in a Python 3.12 virtual environment works as well. A full run of one demo is slow, because KernelSHAP is computed for each margin evaluated. Set `XAI_N_JOBS` to limit the number of parallel workers (default: all cores). On machines without a display, set `MPLBACKEND=Agg`.

## Outputs
Each demo writes the following, where `<data>` is `gbsg2`, `act` or `peak`:

| File | Content |
|---|---|
| `results/<data>_run_settings.json` | Every setting the run used (split, forest, attributions, recommendation rules, evaluation), package versions and git commit |
| `results/<data>_rsf_evaluation.csv` | Test-set metrics of the exploratory random survival forest |
| `results/<data>_model_comparison.csv` | Test-set metrics for every model: without recommendations, with each recommendation, with all of them, and the two comparators |
| `results/<data>_margin_stability.csv` | The margin search: subcohort sizes, recommendation counts and agreement at each margin, and the chosen margin |
| `results/<data>_exclusion_tests.csv` | Exclusion rule for every feature in each subcohort: mean absolute attribution, upper confidence bound and threshold |
| `results/<data>_nonlinear_tests.csv` | Non-linearity rule for every candidate in each subcohort: correlation, gain in R² of the flexible fit, raw and FDR-adjusted P |
| `results/<data>_feature_screen.csv` | Within-value dispersion of each feature's attributions, its pattern, cut point, and whether it was screened for interactions |
| `results/<data>_interaction_tests.csv` | Every interaction test: stratifying feature, partner, method, cut point, effect size with 95% CI, raw, pair-level and FDR-adjusted P, and the reference Cox model result |
| `results/<data>_target_model_tests.csv` | Score tests of the screened pairs in the reference Cox model |
| `results/<data>_budget_log.csv` | Pairs offered to the parameter budget, their cost in columns, and whether each entered the model |
| `results/<data>_summary.csv` | Chosen margin, subcohort sizes, numbers of tests and pairs, parameter budget and number of interaction pairs entered |
| `results/<data>_final_cox_coefficients.csv` | Coefficients of the Cox model with all recommendations |
| `results/<data>_final_cox_hr.tsv` | Hazard ratios with 95% CIs and P values for the same model |
| `results/<data>_ph_test_final.csv` | Schoenfeld residual tests for the Cox model with all recommendations |
| `results/<data>_shap_values_<low\|high>.csv`, `results/<data>_shap_data_<low\|high>.csv` | Attributions and feature values of the two subcohorts at the chosen margin |
| `plots/<data>_shap_low.png`, `plots/<data>_shap_high.png` | Feature attributions in the low- and high-risk subcohorts |
| Calibration plots and bins | `plots/` (GBSG2), `plots/aids/` (ACT), `plots/peak/` (peakVO2) |

Additional analyses (step 5, open data only):

| File | Content |
|---|---|
| `results/<data>_interaction_sensitivity.csv`, `..._hr.tsv` | The final model and the model with every confirmed pair (budget ignored): test-set metrics, paired differences in C from the baseline and final models, training likelihood ratio test of the added terms, and hazard ratios. Only a note when no confirmed pair was left out |
| `results/<data>_calibration_intercept.csv` | Observed and expected events up to t0, log(O/E) with 95% CI, Kaplan-Meier and mean predicted risk at t0, for the forest and each Cox model |
| `results/<data>_ph_global.csv`, `results/<data>_ph_terms.csv` | Global and per-term Schoenfeld tests; residual plots `ph_residuals_<data>_<model>.pdf` in the plot folder |
| `results/<data>_events_per_parameter.csv` | Patients and events per split and subcohort; parameters and events per parameter of each model |
| `results/<data>_coefficient_stability.csv` | Bootstrap SD, percentile interval and sign agreement of each coefficient of the final model |
| `results/<data>_selection_frequency.csv`, `..._summary.csv` | Share of bootstrap resamples of the subcohorts in which each recommendation is made, and agreement with the main run |
| `results/<data>_ablation_subsets.csv`, `results/<data>_ablation_orderings.csv` | C for every subset of the three recommendation types, and the gain from each under every ordering, with paired bootstrap intervals |
| `results/<data>_survival_target_recommendations.csv`, `..._models.csv` | Recommendations from attributions of 1 − S(t0) against the main run, and the resulting final model on the test set; attributions in `results/<data>_shap_values_surv_<low\|high>.csv` |
| `results/<data>_extras_settings.json` | Settings of the additional analyses |

## Simulations
`simulations.py` checks the interaction screen and the non-linearity rule on simulated data, calling the shared code unchanged:
```
python simulations.py            # full run, several hours; resumes if interrupted
python simulations.py --quick    # a few replicates, for checking
```
- **Attribution level** (the setting of reviewer comment 1): two strata of 2,362 and 4,566 patients, a binary comorbidity at 5% and 25% prevalence, and a strictly additive log hazard, with exact attributions plus noise. It compares the published rule (a stratum-specific reference and a Wilcoxon rank-sum test) with the current within-stratum contrast, for type I error and for power when the log hazard includes a product term.
- **Whole pipeline:** survival times from a Cox model with known main effects and interactions; a random survival forest (300 trees) or the true risk function as the exploratory model; then KernelSHAP, the margin search, the three rules, the reference Cox model check and the parameter budget, as `recommend` runs them. Scenarios: an additive model, three interactions (binary × binary, continuous × continuous, binary × continuous), and one feature per shape for the non-linearity rule (no effect, linear, quadratic, hinge, sine, a nominal feature coded 0–3, an ordinal feature with a linear effect).

| File | Content |
|---|---|
| `results/sim_attribution_level.csv` | Rejection rate of each design and test, by interaction size and noise |
| `results/sim_interaction_summary.csv` | Per scenario and exploratory model: share of replicates with a false pair screened, confirmed in the Cox model, and entered; share finding the true pair |
| `results/sim_nonlinear_summary.csv` | Per feature: share of replicates flagged as non-linear by the current rule and by the published rule (\|r\| < 0.1). The no-effect feature is named `null`; read the summaries with `pd.read_csv(..., keep_default_na=False)` or it becomes NaN |
| `results/sim_exclusion_summary.csv` | Per feature: share of replicates in which it is excluded |
| `results/sim_pipeline_replicates.jsonl` | One record per replicate |
| `results/sim_settings.json` | Settings |

## Interactive Test
Once you have installed necessary packages (see steps above), you can also try an interactive [demo](./demo.ipynb).

## Tests
```
pytest test_recommender.py -q
```
The tests also run on every push (GitHub Actions, `.github/workflows/tests.yml`).

## Data
GBSG2 and ACT are loaded from scikit-survival. `data/peakvo2.csv` is the `peakVO2` data set of the R package randomForestSRC (2,231 patients with systolic heart failure, 39 predictors, all-cause death); `data/peakvo2_source.txt` describes its source and conversion.

## Citation
Citation metadata (authors, affiliations, licence) is in [`CITATION.cff`](CITATION.cff); GitHub shows it under "Cite this repository".

## Licence
MIT; see [`LICENSE`](LICENSE). `public_analyses.export_coefficients` is copied from survival-model-toolkit 0.3.0 (MIT), whose notice is included in the same file.
