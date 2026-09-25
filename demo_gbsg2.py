import numpy as np
from sksurv.datasets import load_gbsg2
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sksurv.preprocessing import OneHotEncoder

from utils import MetricEval, CalibrationPerform, get_explanations, make_plot, ShapleyAnalysis, sign_balance_test, strata_generate, stratify_shap_analysis, nonlinear_analysis, get_model, interaction_analysis, check_interaction_spec

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
X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.20, random_state=20)
pre = ColumnTransformer([('con', StandardScaler(), con_cols)],
                        remainder='passthrough', verbose_feature_names_out=False).set_output(transform='pandas')
X_train = pre.fit_transform(X_train)
X_test = pre.transform(X_test)

ex_model = get_model('rf', 20).fit(X_train, y_train)
org_model = get_model('cox', 20).fit(X_train, y_train)

## Evaluate original model
evaluator = MetricEval(1000)
calib = CalibrationPerform(t0=1500, n_bins=5, kind='survival', random_state=0, save_folder='plots/', model_name=['Cox_org'])
calib.calib_plot([org_model], [[X_test, y_test]])
calib.calib_estimate(org_model, X_test, y_test)

## Evaluate rsf model
evaluator = MetricEval(1000)
calib = CalibrationPerform(t0=1500, n_bins=5, kind='survival', random_state=0, save_folder='plots/', model_name=['rsf'])
calib.calib_plot([ex_model], [[X_test, y_test]])
calib.calib_estimate(ex_model, X_test, y_test)

## Get feature attributions
org_X = X_train.copy()
org_y = y_train.copy()
df_shap_low, sel_data_low = get_explanations(ex_model, org_X, org_y, eps=0.2, risk_level='low')
make_plot(df_shap_low, sel_data_low, 'GBSG2_org_shap_low', plot_type='violin', xlabel='SHAP value', figsize=(8, 6))
df_shap_high, sel_data_high = get_explanations(ex_model, org_X, org_y, eps=0.2, risk_level='high')
make_plot(df_shap_high, sel_data_high, 'GBSG2_org_shap_high', plot_type='violin', xlabel='SHAP value', figsize=(8, 6))

# outcomes of the selected training patients, aligned with sel_data
y_sel_low = org_y[org_X.index.get_indexer(sel_data_low.index)]
y_sel_high = org_y[org_X.index.get_indexer(sel_data_high.index)]

shapana = ShapleyAnalysis(0.05, 0.05, 0.05, random_state=20)
for sd_definition in ['across_features', 'whole_matrix']:
    print(f'====== Exclusion (SD {sd_definition}) ======')
    shapana.inclu_exclu_var(df_shap_low, sd_definition=sd_definition)
    shapana.inclu_exclu_var(df_shap_high, sd_definition=sd_definition)
print('======non-linear======')
shapana.non_linear_test(df_shap_low, sel_data_low)
shapana.non_linear_test(df_shap_high, sel_data_high)

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
    shap_file = stratify_shap_analysis(ex_model, X_test_list, y_test_list, variable, risk_level=None)
    df1, df2 = shap_file[0], shap_file[1]
    shapana.wilcoxon_rank_sum_test(df1, df2)

strata = {'age':0, 'tsize':0, 'tgrade':0, 'horTh':-1, 'estrec':0, 'progrec':20}

for variable, thresh in strata.items():
    print(f'The stratified variable is {variable}')
    X_test_list, y_test_list, n_at_threshold = strata_generate(df_shap_high, variable, sel_data_high.reset_index(drop=True), y_sel_high, thresh)
    print(f'{n_at_threshold} observations at the threshold')
    shap_file = stratify_shap_analysis(ex_model, X_test_list, y_test_list, variable, risk_level=None)
    df1, df2 = shap_file[0], shap_file[1]
    shapana.wilcoxon_rank_sum_test(df1, df2)

nonlinear_feature = ['age']
nonlinear_train = X_train.copy()
nonlinear_test = X_test.copy()
X_train_nonlinear, centre = nonlinear_analysis(nonlinear_train, nonlinear_feature, nonlinear_type='quadratic')
X_test_nonlinear, _ = nonlinear_analysis(nonlinear_test, nonlinear_feature, nonlinear_type='quadratic', centre=centre)
cox_new_model = get_model('cox', 20)
cox_new_model.fit(X_train_nonlinear, y_train)

evaluator.cal_metric_CI(cox_new_model, X_test_nonlinear, y_test,'c-index')
calib = CalibrationPerform(t0=1500, kind='survival', random_state=0, save_folder='plots/', model_name=['Cox_org','Cox_nonlinear'])
calib.calib_plot([org_model, cox_new_model], [[X_test, y_test], [X_test_nonlinear,y_test]])
calib.calib_estimate(cox_new_model, X_test_nonlinear, y_test)

inter_feat_total = ['age','estrec','progrec', 'horTh']
interaction_list_total = [['tsize'], ['pnodes', 'age'],
 ['age', 'pnodes', 'estrec', 'horTh', 'tgrade', 'menostat'],['menostat','tsize', 'tgrade','pnodes', 'age']]
non_linear_list = []
# non_linear_list = ['age', 'tsize']
X_test_interact = X_test.copy()
X_train_interact = X_train.copy()
check_interaction_spec(inter_feat_total, interaction_list_total, X_train_interact.columns)
for i in range(len(inter_feat_total)):
    inter_feat = inter_feat_total[i]
    interaction_list = interaction_list_total[i]
    X_train_interact = interaction_analysis(X_train_interact,inter_feat, interaction_list, non_linear_list)
    X_test_interact = interaction_analysis(X_test_interact,inter_feat, interaction_list, non_linear_list)

cox_new_model = get_model('cox', 20)
cox_new_model.fit(X_train_interact, y_train)

evaluator.cal_metric_CI(cox_new_model, X_test_interact, y_test,'c-index')
calib = CalibrationPerform(t0=1500, kind='survival', random_state=0, save_folder='plots/', model_name=['Cox_org','Cox_inter'])
calib.calib_plot([org_model, cox_new_model], [[X_test, y_test], [X_test_interact,y_test]])
calib.calib_estimate(cox_new_model, X_test_interact, y_test)

inter_feat_total = ['age','estrec','progrec', 'horTh']
interaction_list_total = [['tsize'], ['pnodes', 'age'],
 ['age', 'pnodes', 'estrec', 'horTh', 'tgrade', 'menostat'],['menostat','tsize', 'tgrade','pnodes', 'age']]
non_linear_list = ['age']
X_test_interact = X_test_nonlinear.copy()
X_train_interact = X_train_nonlinear.copy()
check_interaction_spec(inter_feat_total, interaction_list_total, X_train_interact.columns)
for i in range(len(inter_feat_total)):
  inter_feat = inter_feat_total[i]
  interaction_list = interaction_list_total[i]
  X_train_interact = interaction_analysis(X_train_interact,inter_feat, interaction_list, non_linear_list)
  X_test_interact = interaction_analysis(X_test_interact,inter_feat, interaction_list, non_linear_list)

cox_new_model = get_model('cox', 20)
cox_new_model.fit(X_train_interact, y_train)

evaluator.cal_metric_CI(cox_new_model, X_test_interact, y_test,'c-index')
calib = CalibrationPerform(t0=1500, kind='survival', random_state=0, save_folder='plots/', model_name=['Cox_org','Cox_all'])
calib.calib_plot([org_model, cox_new_model], [[X_test, y_test], [X_test_interact,y_test]])
calib.calib_estimate(cox_new_model, X_test_interact, y_test)
