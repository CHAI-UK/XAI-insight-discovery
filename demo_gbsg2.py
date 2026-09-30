import os
import numpy as np
import pandas as pd
from sksurv.datasets import load_gbsg2
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sksurv.preprocessing import OneHotEncoder

from utils import (MetricEval, CalibrationPerform, get_model, as_surv, evaluation_times,
                   evaluate_recommendations)
from public_analyses import run_recommend, write_run_record, final_cox_table
from public_extras import run_extras

os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

SEED, TEST_SIZE = 20, 0.20   # train/test split and random survival forest
T0 = 1500                    # calibration horizon, days

## Load the dataset and train the naive model and the recommender.
X, y = load_gbsg2()
y = as_surv(y)

grade_str = X.loc[:, "tgrade"].astype(object).values[:, np.newaxis]
grade_num = OrdinalEncoder(categories=[["I", "II", "III"]]).fit_transform(grade_str)

X_no_grade = X.drop("tgrade", axis=1)
Xt = OneHotEncoder().fit_transform(X_no_grade)
Xt.loc[:, "tgrade"] = grade_num
con_cols = ['age', 'estrec', 'progrec', 'tsize']

Xt.rename(columns={'horTh=yes': 'horTh', 'menostat=Post': 'menostat'}, inplace=True)

# The encodings above use fixed categories; the scaler is fitted on the training split only
X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=TEST_SIZE, random_state=SEED)
pre = ColumnTransformer([('con', StandardScaler(), con_cols)],
                        remainder='passthrough', verbose_feature_names_out=False).set_output(transform='pandas')
X_train = pre.fit_transform(X_train)
X_test = pre.transform(X_test)

ex_model = get_model('rf', SEED).fit(X_train, y_train)

## Evaluate the exploratory model (random survival forest), as in the DataLoch analysis
evaluator = MetricEval(times=evaluation_times(y_train))
report = evaluator.full_report(ex_model, X_train, y_train, X_test, y_test, label='rsf')
calib = CalibrationPerform(t0=T0, n_bins=5, save_folder='plots/')
report.update(calib.report(ex_model, X_test, y_test, label='rsf'))
calib.calib_plot([ex_model], [(X_test, y_test)], model_labels=['rsf'],
                 filename='calibration_gbsg2_rsf.pdf')
print(pd.Series(report))
pd.Series(report).to_csv('results/gbsg2_rsf_evaluation.csv')

## Generate recommendations from the training data
scaler = pre.named_transformers_['con']
rec, info = run_recommend(ex_model, X_train, y_train, continuous=con_cols + ['pnodes'],
                          ordinal=['tgrade'], cutpoints={}, tag='gbsg2')
info['tests'].to_csv('results/gbsg2_interaction_tests.csv', index=False)
info['target'].to_csv('results/gbsg2_target_model_tests.csv', index=False)
pd.Series(info['summary']).to_csv('results/gbsg2_summary.csv')

## Integrate the recommendations and evaluate each once on the test set
evaluate_recommendations(rec, X_train, X_test, y_train, y_test, t0=T0, tag='gbsg2',
                         continuous=con_cols + ['pnodes'], save_folder='plots/')

## Hazard ratios of the final model, and the settings this run used
final_cox_table(rec, X_train, y_train, 'gbsg2')
write_run_record('gbsg2', split=dict(test_size=TEST_SIZE, random_state=SEED, stratified=False),
                 ex_model=ex_model, t0=T0, info=info, eval_times=evaluation_times(y_train))

## Additional analyses (open data only; not part of the DataLoch analysis)
# Interaction sensitivity model, calibration intercept, global PH tests,
# events per parameter, coefficient and selection stability, ablation, and
# attributions of 1 - S(t0); see public_extras.py
run_extras(rec, info, ex_model, X_train, X_test, y_train, y_test, t0=T0, tag='gbsg2',
           plot_dir='plots/')
