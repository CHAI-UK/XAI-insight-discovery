import os
import pandas as pd
from sksurv.datasets import load_aids
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

from utils import (MetricEval, CalibrationPerform, get_model, as_surv, evaluation_times,
                   evaluate_recommendations)
from public_analyses import run_recommend, write_run_record, final_cox_table
from public_extras import run_extras

os.makedirs('plots/aids', exist_ok=True)
os.makedirs('results', exist_ok=True)

SEED, TEST_SIZE = 20, 0.20   # train/test split and random survival forest
T0 = 320                     # calibration horizon, days

## Load the dataset (AIDS Clinical Trials Group 320, endpoint AIDS or death)
X, y = load_aids()
y = as_surv(y)

cat_cols = ['hemophil', 'karnof', 'sex', 'strat2', 'tx']     # binary, and karnof (ordinal)
nom_cols = ['ivdrug', 'raceth']                              # nominal, more than 2 categories
# txgrp is left out: tx (indinavir or not) is the randomised treatment, and
# txgrp differs from it only in levels 3 and 4 (3 patients, the stavudine
# arms), so the two are collinear and the Cox coefficients are not identified
con_cols = ['age', 'cd4', 'priorzdv']
# karnof is ordinal; its levels are given in order, since sorting the strings
# would put '100' before '70'
categories = [['70', '80', '90', '100'] if c == 'karnof' else sorted(X[c].astype(str).unique())
              for c in cat_cols]
X[cat_cols + nom_cols] = X[cat_cols + nom_cols].astype(str)

# Fit the encoders and scaler on the training split only. Nominal features are
# one-hot encoded and recombined for the interaction screen through `groups`.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
pre = ColumnTransformer([('cat', OrdinalEncoder(categories=categories), cat_cols),
                         ('nom', OneHotEncoder(drop='first', sparse_output=False,
                                               handle_unknown='ignore'), nom_cols),
                         ('con', StandardScaler(), con_cols)],
                        verbose_feature_names_out=False).set_output(transform='pandas')
X_train = pre.fit_transform(X_train)
X_test = pre.transform(X_test)
# levels with fewer than 5 training patients are merged into the reference level
rare = [c for c in X_train.columns
        if c.startswith(tuple(f'{g}_' for g in nom_cols)) and X_train[c].sum() < 5]
X_train, X_test = X_train.drop(columns=rare), X_test.drop(columns=rare)
groups = {g: [c for c in X_train.columns if c.startswith(f'{g}_')] for g in nom_cols}

ex_model = get_model('rf', SEED).fit(X_train, y_train)

## Evaluate the exploratory model (random survival forest), as in the DataLoch analysis
evaluator = MetricEval(times=evaluation_times(y_train))
report = evaluator.full_report(ex_model, X_train, y_train, X_test, y_test, label='rsf')
calib = CalibrationPerform(t0=T0, n_bins=5, save_folder='plots/aids/')
report.update(calib.report(ex_model, X_test, y_test, label='rsf'))
calib.calib_plot([ex_model], [(X_test, y_test)], model_labels=['rsf'],
                 filename='calibration_act_rsf.pdf')
print(pd.Series(report))
pd.Series(report).to_csv('results/act_rsf_evaluation.csv')

## Generate recommendations from the training data
# Cut points for stratifying continuous features, prespecified in original units
# and converted to the standardized scale of the model
scaler = pre.named_transformers_['con']
cutpoints = {}
rec, info = run_recommend(ex_model, X_train, y_train, continuous=con_cols, ordinal=['karnof'],
                          cutpoints=cutpoints, groups=groups, tag='act')
info['tests'].to_csv('results/act_interaction_tests.csv', index=False)
info['target'].to_csv('results/act_target_model_tests.csv', index=False)
pd.Series(info['summary']).to_csv('results/act_summary.csv')

## Integrate the recommendations and evaluate each once on the test set
evaluate_recommendations(rec, X_train, X_test, y_train, y_test, t0=T0, tag='act',
                         continuous=con_cols, groups=groups, save_folder='plots/aids/')

## Hazard ratios of the final model, and the settings this run used
final_cox_table(rec, X_train, y_train, 'act', groups=groups)
write_run_record('act', split=dict(test_size=TEST_SIZE, random_state=SEED, stratified=False),
                 ex_model=ex_model, t0=T0, info=info, eval_times=evaluation_times(y_train))

## Additional analyses (open data only; not part of the DataLoch analysis)
# Interaction sensitivity model, calibration intercept, global PH tests,
# events per parameter, coefficient and selection stability, ablation, and
# attributions of 1 - S(t0); see public_extras.py
run_extras(rec, info, ex_model, X_train, X_test, y_train, y_test, t0=T0, tag='act',
           groups=groups, plot_dir='plots/aids/')
