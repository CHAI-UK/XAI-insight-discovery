import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv

from utils import (MetricEval, CalibrationPerform, get_model, evaluation_times,
                   evaluate_recommendations)
from public_analyses import run_recommend, write_run_record, final_cox_table
from public_extras import run_extras

os.makedirs('plots/peak', exist_ok=True)
os.makedirs('results', exist_ok=True)

SEED, TEST_SIZE = 20, 0.20   # train/test split and random survival forest
T0 = 3.0                     # three-year calibration horizon; ttodead is measured in years

## Load the dataset (randomForestSRC::peakVO2; systolic heart failure)
# All-cause death, with follow-up time in years. The CSV preserves all 39
# predictors from the original R dataset; died and ttodead define the outcome.
df = pd.read_csv('data/peakvo2.csv')
y = Surv.from_arrays(event=df['died'].astype(bool), time=df['ttodead'].astype(float))
X = df.drop(columns=['died', 'ttodead']).astype(float)
con_cols = ['age', 'resting.systolic.bp', 'resting.hr', 'bmi', 'lvef.metabl',
            'peak.rer', 'peak.vo2', 'interval', 'bun', 'sodium', 'hgb',
            'glucose', 'crcl']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
pre = ColumnTransformer([('con', StandardScaler(), con_cols)],
                        remainder='passthrough', verbose_feature_names_out=False).set_output(transform='pandas')
X_train = pre.fit_transform(X_train)
X_test = pre.transform(X_test)

ex_model = get_model('rf', SEED).fit(X_train, y_train)

## Evaluate the exploratory model (random survival forest)
evaluator = MetricEval(times=evaluation_times(y_train))
report = evaluator.full_report(ex_model, X_train, y_train, X_test, y_test, label='rsf')
calib = CalibrationPerform(t0=T0, n_bins=5, save_folder='plots/peak/')
report.update(calib.report(ex_model, X_test, y_test, label='rsf'))
calib.calib_plot([ex_model], [(X_test, y_test)], model_labels=['rsf'],
                 filename='calibration_peak_rsf.pdf')
print(pd.Series(report))
pd.Series(report).to_csv('results/peak_rsf_evaluation.csv')

## Generate recommendations from the training data
scaler = pre.named_transformers_['con']
cutpoints = {}
rec, info = run_recommend(ex_model, X_train, y_train, continuous=con_cols, ordinal=[],
                          cutpoints=cutpoints, tag='peak')
info['tests'].to_csv('results/peak_interaction_tests.csv', index=False)
info['target'].to_csv('results/peak_target_model_tests.csv', index=False)
pd.Series(info['summary']).to_csv('results/peak_summary.csv')

## Integrate the recommendations and evaluate each once on the test set
evaluate_recommendations(rec, X_train, X_test, y_train, y_test, t0=T0, tag='peak',
                         continuous=con_cols, save_folder='plots/peak/')

## Hazard ratios of the final model, and the settings this run used
final_cox_table(rec, X_train, y_train, 'peak')
write_run_record('peak', split=dict(test_size=TEST_SIZE, random_state=SEED, stratified=False),
                 ex_model=ex_model, t0=T0, info=info, eval_times=evaluation_times(y_train))

## Additional analyses (open data only; not part of the DataLoch analysis)
# Interaction sensitivity model, calibration intercept, global PH tests,
# events per parameter, coefficient and selection stability, ablation, and
# attributions of 1 - S(t0); see public_extras.py
run_extras(rec, info, ex_model, X_train, X_test, y_train, y_test, t0=T0, tag='peak',
           plot_dir='plots/peak/')
