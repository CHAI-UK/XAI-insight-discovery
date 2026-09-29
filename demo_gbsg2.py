import os
import numpy as np
import pandas as pd
from sksurv.datasets import load_gbsg2
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sksurv.preprocessing import OneHotEncoder

from utils import (MetricEval, CalibrationPerform, get_model, as_surv, evaluation_times,
                   recommend, evaluate_recommendations)

os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

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
X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.20, random_state=20)
pre = ColumnTransformer([('con', StandardScaler(), con_cols)],
                        remainder='passthrough', verbose_feature_names_out=False).set_output(transform='pandas')
X_train = pre.fit_transform(X_train)
X_test = pre.transform(X_test)

ex_model = get_model('rf', 20).fit(X_train, y_train)

## Evaluate the exploratory model (random survival forest), as in the DataLoch analysis
evaluator = MetricEval(times=evaluation_times(y_train))
report = evaluator.full_report(ex_model, X_train, y_train, X_test, y_test, label='rsf')
calib = CalibrationPerform(t0=1500, n_bins=5, save_folder='plots/')
report.update(calib.report(ex_model, X_test, y_test, label='rsf'))
calib.calib_plot([ex_model], [(X_test, y_test)], model_labels=['rsf'],
                 filename='calibration_gbsg2_rsf.pdf')
print(pd.Series(report))

## Generate recommendations from the training data
scaler = pre.named_transformers_['con']
rec, info = recommend(ex_model, X_train, y_train, continuous=con_cols + ['pnodes'],
                      ordinal=['tgrade'], cutpoints={}, tag='gbsg2')
info['tests'].to_csv('results/gbsg2_interaction_tests.csv', index=False)
info['target'].to_csv('results/gbsg2_target_model_tests.csv', index=False)
pd.Series(info['summary']).to_csv('results/gbsg2_summary.csv')

## Integrate the recommendations and evaluate each once on the test set
evaluate_recommendations(rec, X_train, X_test, y_train, y_test, t0=1500, tag='gbsg2',
                         continuous=con_cols + ['pnodes'], save_folder='plots/')
