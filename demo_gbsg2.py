import os
import sys
import numpy as np
import pandas as pd
from sksurv.datasets import load_gbsg2
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sksurv.preprocessing import OneHotEncoder

from utils import exclusion_analysis, load_config, save_run_config, save_recommendations, cox_coefficient_table, recommended_exclusions, recommended_nonlinear, recommended_interactions, save_model_spec, screen_skips, MetricEval, CalibrationPerform, get_explanations, make_plot, ShapleyAnalysis, sign_balance_test, strata_generate, stratify_shap_analysis, nonlinear_analysis, get_model, interaction_analysis, check_interaction_spec

# All settings come from config.yml; the ones used are saved to results/gbsg2_run_config.json.
# Run with --quick for a fast debugging run (results go to results_quick/, not for reporting).
cfg = load_config('gbsg2', quick='--quick' in sys.argv)
np.random.seed(cfg['seed'])   # KernelExplainer sampling and plot jitter
save_run_config(cfg)

## Load the dataset and train the naive model and the recommendar.
X, y = load_gbsg2()

grade_str = X.loc[:, "tgrade"].astype(object).values[:, np.newaxis]
grade_num = OrdinalEncoder(categories=[["I", "II", "III"]]).fit_transform(grade_str)

X_no_grade = X.drop("tgrade", axis=1)
Xt = OneHotEncoder().fit_transform(X_no_grade)
Xt.loc[:, "tgrade"] = grade_num
con_cols = ['age', 'estrec', 'progrec', 'tsize']

Xt.rename(columns={'horTh=yes': 'horTh', 'menostat=Post': 'menostat'}, inplace=True)

# The encodings above use fixed categories; the scaler is fitted on the training split only
X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=cfg['test_size'], random_state=cfg['seed'])
pre = ColumnTransformer([('con', StandardScaler(), con_cols)],
                        remainder='passthrough', verbose_feature_names_out=False).set_output(transform='pandas')
X_train = pre.fit_transform(X_train)
X_test = pre.transform(X_test)

ex_model = get_model('rf', cfg['seed'], **cfg['rsf']).fit(X_train, y_train)
org_model = get_model('cox', cfg['seed']).fit(X_train, y_train)

## Evaluate original model
evaluator = MetricEval(cfg['n_bootstrap'], seed=cfg['seed'])
calib = CalibrationPerform(t0=cfg['calibration_t0'], n_bins=cfg['calibration_bins'], kind='survival', random_state=cfg['seed'], save_folder=cfg['plots_dir'], model_name=['Cox_org'])
calib.calib_plot([org_model], [[X_test, y_test]])
calib.calib_estimate(org_model, X_test, y_test)

## Evaluate rsf model
evaluator = MetricEval(cfg['n_bootstrap'], seed=cfg['seed'])
calib = CalibrationPerform(t0=cfg['calibration_t0'], n_bins=cfg['calibration_bins'], kind='survival', random_state=cfg['seed'], save_folder=cfg['plots_dir'], model_name=['rsf'])
calib.calib_plot([ex_model], [[X_test, y_test]])
calib.calib_estimate(ex_model, X_test, y_test)

## Get feature attributions
org_X = X_train.copy()
org_y = y_train.copy()
df_shap_low, sel_data_low = get_explanations(ex_model, org_X, org_y, eps=cfg['subgroup_margin'], risk_level='low',
                                               max_rows=cfg['shap_max_rows'], seed=cfg['seed'])
make_plot(df_shap_low, sel_data_low, 'GBSG2_org_shap_low', plot_type='violin', xlabel='SHAP value', figsize=(8, 6), folder=cfg['plots_dir'])
df_shap_high, sel_data_high = get_explanations(ex_model, org_X, org_y, eps=cfg['subgroup_margin'], risk_level='high',
                                               max_rows=cfg['shap_max_rows'], seed=cfg['seed'])
make_plot(df_shap_high, sel_data_high, 'GBSG2_org_shap_high', plot_type='violin', xlabel='SHAP value', figsize=(8, 6), folder=cfg['plots_dir'])

# outcomes of the selected training patients, aligned with sel_data
y_sel_low = org_y[org_X.index.get_indexer(sel_data_low.index)]
y_sel_high = org_y[org_X.index.get_indexer(sel_data_high.index)]

strata_log = []
exclusion_tests, nonlinear_tests, interaction_tests = [], [], []
shapana = ShapleyAnalysis(cfg['exclusion_threshold'], None, cfg['interaction_p_threshold'], random_state=cfg['seed'],
                          r_thresh=cfg['nonlinear_r_threshold'], n_boot=cfg['n_bootstrap'])
for sd_definition in ['across_features', 'whole_matrix']:
    print(f'====== Exclusion (SD {sd_definition}) ======')
    exclusion_tests.append(shapana.inclu_exclu_var(df_shap_low, sd_definition=sd_definition).assign(cohort='low', sd_definition=sd_definition))
    exclusion_tests.append(shapana.inclu_exclu_var(df_shap_high, sd_definition=sd_definition).assign(cohort='high', sd_definition=sd_definition))
print('======non-linear======')
nonlinear_tests.append(shapana.non_linear_test(df_shap_low, sel_data_low).assign(cohort='low'))
nonlinear_tests.append(shapana.non_linear_test(df_shap_high, sel_data_high).assign(cohort='high'))

# Excluded features are not tested as interaction partners, nor used in interaction terms
excluded = recommended_exclusions(exclusion_tests)
print('Recommended for exclusion:', excluded)
nonlinear_feature = recommended_nonlinear(nonlinear_tests, X_train, excluded)
print('Recommended squared terms:', nonlinear_feature)

print('====== Low risk cohort ======')
sign_balance_test(df_shap_low)
print('====== High risk cohort ======')
sign_balance_test(df_shap_high)

strata = {'age':0, 'estrec':0, 'progrec':6, 'horTh':-2}
# strata = {'pnodes':-1, 'tsize':0}
for variable, thresh in strata.items():
    print(f'The stratified variable is {variable}')
    X_test_list, y_test_list, n_at_threshold = strata_generate(df_shap_low, variable, sel_data_low.reset_index(drop=True), y_sel_low, thresh)
    print(f'{n_at_threshold} observations at the threshold')
    strata_log.append(dict(cohort='low', variable=variable, split_on='SHAP value', threshold=thresh,
                           n_at_or_below=len(X_test_list[0]), n_above=len(X_test_list[1]), n_at_threshold=n_at_threshold))
    shap_file = stratify_shap_analysis(ex_model, X_test_list, y_test_list, variable, risk_level=None, margin=cfg['subgroup_margin'],
                                       max_rows=cfg['shap_max_rows'], seed=cfg['seed'])
    df1, df2 = shap_file[0], shap_file[1]
    interaction_tests.append(shapana.wilcoxon_rank_sum_test(df1, df2, skip=screen_skips(variable, excluded)).assign(
        cohort='low', stratifying_variable=variable, split_on='SHAP value', threshold=thresh))

strata = {'age':0, 'tsize':0, 'tgrade':0, 'horTh':-1, 'estrec':0, 'progrec':20}

for variable, thresh in strata.items():
    print(f'The stratified variable is {variable}')
    X_test_list, y_test_list, n_at_threshold = strata_generate(df_shap_high, variable, sel_data_high.reset_index(drop=True), y_sel_high, thresh)
    print(f'{n_at_threshold} observations at the threshold')
    strata_log.append(dict(cohort='high', variable=variable, split_on='SHAP value', threshold=thresh,
                           n_at_or_below=len(X_test_list[0]), n_above=len(X_test_list[1]), n_at_threshold=n_at_threshold))
    shap_file = stratify_shap_analysis(ex_model, X_test_list, y_test_list, variable, risk_level=None, margin=cfg['subgroup_margin'],
                                       max_rows=cfg['shap_max_rows'], seed=cfg['seed'])
    df1, df2 = shap_file[0], shap_file[1]
    interaction_tests.append(shapana.wilcoxon_rank_sum_test(df1, df2, skip=screen_skips(variable, excluded)).assign(
        cohort='high', stratifying_variable=variable, split_on='SHAP value', threshold=thresh))

# Record the analyst-chosen strata thresholds with the results
pd.DataFrame(strata_log).to_csv(os.path.join(cfg['results_dir'], f"{cfg['dataset']}_strata_thresholds.csv"), index=False)
# Every exclusion, non-linearity and interaction test, as machine-readable tables
rec_tables = save_recommendations(cfg, exclusion_tests, nonlinear_tests, interaction_tests)

# The model specification comes from the screen output, not from hand-typed lists
inter_feat_total, interaction_list_total = recommended_interactions(rec_tables['interactions'])
save_model_spec(cfg, excluded, nonlinear_feature, inter_feat_total, interaction_list_total)

nonlinear_train = X_train.copy()
nonlinear_test = X_test.copy()
X_train_nonlinear, centre = nonlinear_analysis(nonlinear_train, nonlinear_feature, nonlinear_type='quadratic')
X_test_nonlinear, _ = nonlinear_analysis(nonlinear_test, nonlinear_feature, nonlinear_type='quadratic', centre=centre)
cox_new_model = get_model('cox', cfg['seed'])
cox_new_model.fit(X_train_nonlinear, y_train)

evaluator.cal_metric_CI(cox_new_model, X_test_nonlinear, y_test,'c-index')
calib = CalibrationPerform(t0=cfg['calibration_t0'], n_bins=cfg['calibration_bins'], kind='survival', random_state=cfg['seed'], save_folder=cfg['plots_dir'], model_name=['Cox_org','Cox_nonlinear'])
calib.calib_plot([org_model, cox_new_model], [[X_test, y_test], [X_test_nonlinear,y_test]])
calib.calib_estimate(cox_new_model, X_test_nonlinear, y_test)

non_linear_list = []
X_test_interact = X_test.copy()
X_train_interact = X_train.copy()
check_interaction_spec(inter_feat_total, interaction_list_total, X_train_interact.columns, excluded=excluded)
for i in range(len(inter_feat_total)):
    inter_feat = inter_feat_total[i]
    interaction_list = interaction_list_total[i]
    X_train_interact, inter_centre = interaction_analysis(X_train_interact,inter_feat, interaction_list, non_linear_list)
    X_test_interact, _ = interaction_analysis(X_test_interact,inter_feat, interaction_list, non_linear_list, centre=inter_centre, verbose=False)

cox_new_model = get_model('cox', cfg['seed'])
cox_new_model.fit(X_train_interact, y_train)

evaluator.cal_metric_CI(cox_new_model, X_test_interact, y_test,'c-index')
calib = CalibrationPerform(t0=cfg['calibration_t0'], n_bins=cfg['calibration_bins'], kind='survival', random_state=cfg['seed'], save_folder=cfg['plots_dir'], model_name=['Cox_org','Cox_inter'])
calib.calib_plot([org_model, cox_new_model], [[X_test, y_test], [X_test_interact,y_test]])
calib.calib_estimate(cox_new_model, X_test_interact, y_test)

non_linear_list = nonlinear_feature
X_test_interact = exclusion_analysis(X_test_nonlinear, excluded)
X_train_interact = exclusion_analysis(X_train_nonlinear, excluded)
check_interaction_spec(inter_feat_total, interaction_list_total, X_train_interact.columns, excluded=excluded)
for i in range(len(inter_feat_total)):
  inter_feat = inter_feat_total[i]
  interaction_list = interaction_list_total[i]
  X_train_interact, inter_centre = interaction_analysis(X_train_interact,inter_feat, interaction_list, non_linear_list)
  X_test_interact, _ = interaction_analysis(X_test_interact,inter_feat, interaction_list, non_linear_list, centre=inter_centre, verbose=False)

cox_new_model = get_model('cox', cfg['seed'])
cox_new_model.fit(X_train_interact, y_train)

# Record the final design matrix and the coefficient table (HR, 95% CI) with the results
print('FINAL DESIGN MATRIX COLUMNS:', list(X_train_interact.columns))
print(f"{sum('_x_' in c for c in X_train_interact.columns)} interaction terms in the final model")
cox_coefficient_table(cox_new_model, X_train_interact, y_train).to_csv(
    os.path.join(cfg['results_dir'], f"{cfg['dataset']}_final_cox_coefficients.csv"), index=False)

evaluator.cal_metric_CI(cox_new_model, X_test_interact, y_test,'c-index')
calib = CalibrationPerform(t0=cfg['calibration_t0'], n_bins=cfg['calibration_bins'], kind='survival', random_state=cfg['seed'], save_folder=cfg['plots_dir'], model_name=['Cox_org','Cox_all'])
calib.calib_plot([org_model, cox_new_model], [[X_test, y_test], [X_test_interact,y_test]])
calib.calib_estimate(cox_new_model, X_test_interact, y_test)
