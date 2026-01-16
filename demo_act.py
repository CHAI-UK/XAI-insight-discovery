from sksurv.datasets import load_aids
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sksurv.preprocessing import OneHotEncoder

from utils import split_and_train, MetricEval, CalibrationPerform, get_explanations, make_plot, ShapleyAnalysis, sign_balance_test, strata_generate, stratify_shap_analysis, nonlinear_analysis, get_model, interaction_analysis

X, y = load_aids()

cat_cols = ['hemophil', 'ivdrug', 'karnof', 'raceth', 'sex', 'strat2', 'tx', 'txgrp']
X[cat_cols] = OrdinalEncoder().fit_transform(X[cat_cols])
con_cols = ['age', 'cd4', 'priorzdv']
X[con_cols] = StandardScaler().fit_transform(X[con_cols])

ex_model, (X_train, y_train), (X_test, y_test) = split_and_train(X, y, 'rf')
org_model, (X_train, y_train), (X_test, y_test) = split_and_train(X, y, 'cox')

## Evaluate original model
evaluator = MetricEval(1000)
evaluator.cal_metric_CI(org_model, X_test, y_test,'c-index')
calib = CalibrationPerform(t0=320, kind='survival', random_state=0, save_folder='plots/aids/', model_name=['Cox_org'])
calib.calib_plot([org_model], [[X_test, y_test]])
calib.calib_estimate(org_model, X_test, y_test)

## Evaluate rsf model
evaluator = MetricEval(1000)
evaluator.cal_metric_CI(ex_model, X_test, y_test,'c-index')
calib = CalibrationPerform(t0=320, kind='survival', random_state=0, save_folder='plots/aids/', model_name=['rsf'])
calib.calib_plot([ex_model], [[X_test, y_test]])
calib.calib_estimate(ex_model, X_test, y_test)

## Get feature interaction
org_X = X_train.copy()
org_y = y_train.copy()
df_shap_low, sel_data_low = get_explanations(ex_model, org_X, org_y, eps=0.1, risk_level='low')
make_plot(df_shap_low, sel_data_low, 'aids_org_shap_low', plot_type='violin', xlabel='SHAP value', figsize=(8, 6))
df_shap_high, sel_data_high = get_explanations(ex_model, org_X, org_y, eps=0.1, risk_level='high')
make_plot(df_shap_high, sel_data_high, 'aids_org_shap_high', plot_type='violin', xlabel='SHAP value', figsize=(8, 6))

shapana = ShapleyAnalysis(0.05, 0.05, 0.05, random_state=20)
shapana.inclu_exclu_var(df_shap_low)
shapana.inclu_exclu_var(df_shap_high)
shapana.non_linear_test(df_shap_low, sel_data_low)
shapana.non_linear_test(df_shap_high, sel_data_high)

exclu_cols = ['hemophil']
exclu_train = X_train.copy()
exclu_test = X_test.copy()
X_train_exclu = exclusion_analysis(exclu_train, exclu_cols)
X_test_exclu = exclusion_analysis(exclu_test, exclu_cols)
cox_new_model = get_model('cox', 20)
cox_new_model.fit(X_train_exclu, y_train)

evaluator.cal_metric_CI(cox_new_model, X_test_exclu, y_test,'c-index')
calib = CalibrationPerform(t0=320, kind='survival', random_state=0, save_folder='plots/aids/', model_name=['Cox_org','Cox_exclu'])
calib.calib_plot([org_model, cox_new_model], [[X_test, y_test], [X_test_exclu,y_test]])
calib.calib_estimate(cox_new_model, X_test_exclu, y_test)

nonlinear_feature = ['age', 'karnof']
nonlinear_train = X_train.copy()
nonlinear_test = X_test.copy()
X_train_nonlinear = nonlinear_analysis(nonlinear_train, nonlinear_feature, nonlinear_type='quadratic')
X_test_nonlinear = nonlinear_analysis(nonlinear_test, nonlinear_feature, nonlinear_type='quadratic')
cox_new_model = get_model('cox', 20)
cox_new_model.fit(X_train_nonlinear, y_train)

evaluator.cal_metric_CI(cox_new_model, X_test_nonlinear, y_test,'c-index')
calib = CalibrationPerform(t0=320, kind='survival', random_state=0, save_folder='plots/aids/', model_name=['Cox_org','Cox_nonlinear'])
calib.calib_plot([org_model, cox_new_model], [[X_test, y_test], [X_test_nonlinear,y_test]])
calib.calib_estimate(cox_new_model, X_test_nonlinear, y_test)

print('====== Low risk cohort ======')
sign_balance_test(df_shap_low)
print('====== High risk cohort ======')
sign_balance_test(df_shap_high)

# strata = {'sex':0.01, 'ivdrug':0, 'raceth':0.01, 'priorzdv': -0.1, 'cd4':-0.01}
strata = {'tx':0.1, 'txgrp':0.1}
for variable, thresh in strata.items():
    print(f'The stratified variable is {variable}')
    X_test_list, y_test_list =strata_generate(df_shap_low, variable, sel_data_low.reset_index(drop=True), y_test, thresh)
    shap_file = stratify_shap_analysis(ex_model, X_test_list, y_test_list, variable, risk_level=None)
    df1, df2 = shap_file[0], shap_file[1]
    shapana.wilcoxon_rank_sum_test(df1, df2)

strata = {'age':0, 'karnof':-1, 'cd4':1,'tx':-0.01,'txgrp': -0.01}
# strata = {'tx':-0.01}
for variable, thresh in strata.items():
    print(f'The stratified variable is {variable}')
    X_test_list, y_test_list =strata_generate(df_shap_high, variable, sel_data_high.reset_index(drop=True), y_test, thresh)
    shap_file = stratify_shap_analysis(ex_model, X_test_list, y_test_list, variable, risk_level=None)
    df1, df2 = shap_file[0], shap_file[1]
    shapana.wilcoxon_rank_sum_test(df1, df2)

inter_feat_total = ['ivdrug', 'cd4', 'priorzdv' 'raceth', 'age']
interaction_list_total = [['karnof'], ['karnof','raceth'],
                          ['ivdrug', 'priorzdv'], ['ivdrug', 'priorzdv', 'sex'],
                          ['strat2','priorzdv','karnof','raceth']]
non_linear_list = []
X_test_interact = X_test.copy()
X_train_interact = X_train.copy()
for i in range(len(inter_feat_total)):
  inter_feat = inter_feat_total[i]
  interaction_list = interaction_list_total[i]
  X_train_interact = interaction_analysis(X_train_interact,inter_feat, interaction_list, non_linear_list)
  X_test_interact = interaction_analysis(X_test_interact,inter_feat, interaction_list, non_linear_list)

cox_new_model = get_model('cox', 20)
cox_new_model.fit(X_train_interact, y_train)

evaluator.cal_metric_CI(cox_new_model, X_test_interact, y_test,'c-index')
calib = CalibrationPerform(t0=320, kind='survival', random_state=0, save_folder='plots/aids/', model_name=['Cox_org','Cox_inter'])
calib.calib_plot([org_model, cox_new_model], [[X_test, y_test], [X_test_interact,y_test]])
calib.calib_estimate(cox_new_model, X_test_interact, y_test)

X_train_final = X_train_exclu.copy()
X_test_final = X_test_exclu.copy()

X_train_final = nonlinear_analysis(X_train_final, nonlinear_feature, nonlinear_type='quadratic')
X_test_final = nonlinear_analysis(X_test_final, nonlinear_feature, nonlinear_type='quadratic')

inter_feat_total = ['ivdrug', 'cd4', 'priorzdv' 'raceth', 'age']
interaction_list_total = [['karnof'], ['karnof','raceth'],
                          ['ivdrug', 'priorzdv'], ['ivdrug', 'priorzdv', 'sex'],
                          ['strat2','priorzdv','karnof','raceth']]
non_linear_list = ['age','karnof']

X_test_interact = X_test_final.copy()
X_train_interact = X_train_final.copy()
for i in range(len(inter_feat_total)):
  inter_feat = inter_feat_total[i]
  interaction_list = interaction_list_total[i]
  X_train_interact = interaction_analysis(X_train_interact,inter_feat, interaction_list, non_linear_list)
  X_test_interact = interaction_analysis(X_test_interact,inter_feat, interaction_list, non_linear_list)
cox_new_model = get_model('cox', 20)
cox_new_model.fit(X_train_interact, y_train)

evaluator.cal_metric_CI(cox_new_model, X_test_interact, y_test,'c-index')
calib = CalibrationPerform(t0=320, kind='survival', random_state=0, save_folder='plots/aids/', model_name=['Cox_org','Cox_all'])
calib.calib_plot([org_model, cox_new_model], [[X_test, y_test], [X_test_interact,y_test]])
calib.calib_estimate(cox_new_model, X_test_interact, y_test)