"""
utils.py: shared tools for the demonstrations on open data sets.

Evaluation (MetricEval, CalibrationPerform, fit_and_score, comparator_models)
and the recommendation rules (recommender.py) are the same code as used for
the main cohort (DataLoch), so that every data set is analysed identically.
"""
import itertools
import json
import re
import os

import numpy as np
import pandas as pd
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.util import Surv

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from scipy.stats import gaussian_kde
from sksurv.linear_model import CoxPHSurvivalAnalysis
from recommender import Recommender

RANDOM_STATE = 20
N_BOOT = 1000
N_JOBS = int(os.environ.get("XAI_N_JOBS", -1))    # workers for the forest, SHAP and bootstraps
RESULT_DIR = "results"


def read_tsv(path, **kw):
    return pd.read_csv(path, sep="\t", **kw)


def write_tsv(df, path, index=False, **kw):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, sep="\t", index=index, **kw)
    return path


# =============================================================================
# Models and attributions
# =============================================================================
# Split the dataset and fit the naive model and the recommender.
def split_and_train(X, y, model_name, seed=20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

    model = get_model(model_name, seed)
    model.fit(X_train, y_train)

    return model, (X_train, y_train), (X_test, y_test)


def get_model(name, seed=20, low_memory=False):
    if name == 'rf':
        # low_memory keeps only what predict() needs; use it when the data have
        # many distinct event times, where a full forest stores a survival curve
        # at every node and becomes very large
        model = RandomSurvivalForest(n_estimators=1000, min_samples_split=10, min_samples_leaf=15,
                                     low_memory=low_memory, n_jobs=-1, random_state=seed)
    elif name == 'cox':
        # a negligible ridge penalty keeps the fit stable when interaction and
        # indicator columns make the design nearly collinear
        model = CoxPHSurvivalAnalysis(alpha=1e-6, ties='efron')
    else:
        raise ValueError(f"Unrecognised model name '{name}'!")

    return model


## Generate shap analysis results and shap plots for the Recommendar
def _extract_time_event(y):
    """
    Return (time, event) as 1D numpy arrays.

    Supports:
        * sksurv structured arrays from load_gbsg2, load_aids, ...
        * pandas DataFrame with usual column names.
    """

    # Case 1: structured array (what sksurv datasets return)
    if isinstance(y, np.ndarray) and y.dtype.names is not None:
        names = list(y.dtype.names)
        if len(names) != 2:
            raise ValueError(f"Structured array y must have exactly 2 fields, got {names}")

        # By sksurv convention: first field = event indicator (bool),
        # second field = time (float).
        event = y[names[0]].astype(bool)
        time = y[names[1]].astype(float)

        return time, event

    # Case 2: pandas DataFrame
    if isinstance(y, pd.DataFrame):
        # possible event column names across datasets
        event_candidates = [
            "event", "Status", "status", "death", "fstat",
            "cens", "censor", "censor_d"
        ]

        time_candidates = [
            "time", "Survival_in_days", "lenfol", "time_d", "futime"
        ]

        event_col = next((c for c in event_candidates if c in y.columns), None)
        time_col = next((c for c in time_candidates if c in y.columns), None)

        if event_col is None or time_col is None:
            raise ValueError(
                "Could not infer event/time columns in DataFrame y; "
                "got columns: " + ", ".join(y.columns)
            )

        event = y[event_col].astype(bool).to_numpy()
        time = y[time_col].astype(float).to_numpy()

        return time, event

    raise TypeError("Unsupported type for y. Use a sksurv structured array or a pandas DataFrame.")


def get_explanations(model, X_test, y_test, eps, risk_level='low', nsamples='auto', groups=None,
                     selected_index=None, seed=RANDOM_STATE):
    """
    KernelSHAP attributions of the log risk score, against a single reference
    row: the mean patient of the event group (risk_level='high') or of the
    non-event group ('low'). Patients whose log risk lies within eps SDs of the
    reference's are explained.

    The log scale matters: a Cox model without interactions is additive in log
    hazard, so its risk score C*exp(x'b) is multiplicative, and explaining the
    raw score makes any two features with main effects look as if they
    interact. The returned attributions are indexed like the selected rows;
    with groups ({feature: [one-hot columns]}) the attributions of each
    feature's dummies are summed onto the feature.
    """
    time, event = _extract_time_event(y_test)
    if risk_level == 'high':
        risk_data = X_test[event]
    elif risk_level == 'low':
        risk_data = X_test[~event]
    else:
        risk_data = X_test
    if risk_data.empty or not X_test.index.is_unique:
        raise ValueError('SHAP needs a nonempty reference group and unique patient indices')

    cols = list(X_test.columns)

    def log_risk(Z):
        if not isinstance(Z, pd.DataFrame):
            Z = pd.DataFrame(Z, columns=cols)
        return np.log(np.maximum(model.predict(Z), 1e-12))

    # compute the reference point (average patient)
    X_mean = risk_data.mean().to_frame().T

    if selected_index is not None:
        sel_data = X_test.loc[selected_index]
    elif (risk_level == 'high') | (risk_level == 'low'):
        y_mean = float(log_risk(X_mean)[0])
        margin = np.std(log_risk(risk_data)) * eps
        # select only those individuals close to the reference risk
        sel_mask = np.abs(log_risk(X_test) - y_mean) < margin
        sel_data = X_test[sel_mask]
    else:
        sel_data = X_test
    if sel_data.empty:
        return aggregate_shap(pd.DataFrame(columns=cols, index=sel_data.index, dtype=float), groups), sel_data

    # patients are explained in parallel chunks; the forest is set to one
    # thread meanwhile so that workers do not oversubscribe the cores
    from joblib import Parallel, delayed
    n_workers = (os.cpu_count() or 1) if N_JOBS in (-1, None) else N_JOBS
    positions = X_test.index.get_indexer(sel_data.index)
    prev = getattr(model, 'n_jobs', None)
    if prev is not None:
        model.set_params(n_jobs=1)

    def one(rows):
        state = np.random.get_state()
        try:
            ex = shap.KernelExplainer(log_risk, X_mean)
            values = []
            for row in rows:
                # Reproducible per patient/reference, independent of batch size.
                np.random.seed(int(np.random.SeedSequence(
                    [seed, int(positions[row]), int(risk_level == 'high')]
                ).generate_state(1)[0]))
                v = ex.shap_values(sel_data.iloc[[row]], nsamples=nsamples, silent=True)
                values.append(np.asarray(v).reshape(1, len(cols)))
            return np.vstack(values)
        finally:
            np.random.set_state(state)

    splits = [s for s in np.array_split(np.arange(len(sel_data)), max(1, n_workers)) if len(s)]
    try:
        values = np.vstack(Parallel(n_jobs=len(splits))(delayed(one)(s) for s in splits))
    finally:
        if prev is not None:
            model.set_params(n_jobs=prev)
    df_shap = pd.DataFrame(values, columns=cols, index=sel_data.index)
    return aggregate_shap(df_shap, groups), sel_data


def aggregate_shap(shap_df, onehot_groups):
    """Sum dummy-level attributions back onto the parent feature."""
    shap_df = shap_df.copy()
    for feature, dummies in (onehot_groups or {}).items():
        cols = [c for c in dummies if c in shap_df.columns]
        if not cols:
            continue
        shap_df[feature] = shap_df[cols].sum(axis=1)
        shap_df = shap_df.drop(columns=cols)
    return shap_df


def make_plot(df_shap, org_df, filename, plot_type='scatter', xlabel='SHAP value', figsize=(4, 3)):
    features = df_shap.columns.to_list()
    n_feats = len(features)

    fig, ax = plt.subplots(figsize=figsize)

    if plot_type == 'violin':
        if org_df is not None:
            shap_long = df_shap.melt(var_name='Feature', value_name='SHAP Value')
            feat_long = org_df.melt(var_name='Feature', value_name='Feature Value')
            data = pd.concat([shap_long, feat_long['Feature Value']], axis=1)
        else:
            raise ValueError('No original faeture values provided')

        for i, feature in enumerate(features):
            df_feat = data[data['Feature'] == feature]

            try:
                kde = gaussian_kde(df_feat['SHAP Value'].values)
                x_vals = df_feat['SHAP Value'].values
                density_vals = kde(x_vals)

                density_vals = density_vals/density_vals.max()*0.4

                y_center = i
                y_vals = np.random.uniform(low=i-density_vals, high=i+density_vals)

            except np.linalg.LinAlgError:
                print(f'KDE failed for feature {feature}, using scatter fallback')
                x_vals = df_feat['SHAP Value'].values
                y_vals = np.full_like(df_feat['SHAP Value'], fill_value=i, dtype=float)

            val = df_feat['Feature Value'].values

            if (val.max() - val.min()) == 0.0:
                val_norm = np.full_like(val, fill_value=val.max(), dtype=float)
            else:
                val_norm = (val - val.min()) / (val.max() - val.min())

            sc = ax.scatter(
                x_vals, y_vals,
                c=val_norm,
                cmap='coolwarm', edgecolors='None', s=20, alpha=0.8
            )

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Feature Value')
        cbar.set_ticks([])
        cbar.ax.text(1.05, 0.05, 'Low', transform=cbar.ax.transAxes, va='center')
        cbar.ax.text(1.05, 0.95, 'High', transform=cbar.ax.transAxes, va='center')
    elif plot_type == 'bar':
        for i, feature in enumerate(features):
            x = df_shap[feature].values
            y = np.full_like(x, fill_value=i, dtype=float)
            plt.barh(y, x, align='center')

    ax.axvline(0.0, c='k', linewidth=0.5)
    ax.set_yticks(np.arange(n_feats))
    ax.set_yticklabels(features)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Feature')
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.grid(True, axis='both', linestyle='--', alpha=0.5)
    plt.savefig(f'plots/{filename}.png', dpi=1000)
    plt.tight_layout()
    plt.show()


# =============================================================================
# Evaluation (as in the DataLoch analysis)
# =============================================================================
class NonlinearTransform:
    """Quadratic or restricted-cubic-spline expansion fitted on training data.
    
    Reuse training means and spline design information when transforming new data."""

    def __init__(self, features, kind="quadratic", df_spline=4):
        self.features = list(features)
        self.kind = kind
        self.df_spline = df_spline
        self.centers_ = {}
        self.design_info_ = {}
        self.output_names_ = []

    def fit(self, X):
        self.centers_, self.design_info_, self.output_names_ = {}, {}, []
        for f in self.features:
            if f not in X.columns:
                continue
            x = X[f].to_numpy(float)
            if self.kind == "quadratic":
                self.centers_[f] = float(np.nanmean(x))
                self.output_names_.append(f"{f}_quadratic")
            elif self.kind == "spline":
                from patsy import dmatrix
                d = dmatrix(f"0 + cr(x, df={self.df_spline})",
                            {"x": x}, return_type="dataframe")
                self.design_info_[f] = d.design_info
                self.output_names_ += [f"{f}_spline{j}" for j in range(d.shape[1])]
            else:
                raise ValueError("kind must be 'quadratic' or 'spline'")
        return self

    def transform(self, X):
        X = X.copy()
        new = {}
        for f in self.features:
            if f not in X.columns:
                continue
            if self.kind == "quadratic":
                X[f] = X[f] - self.centers_[f]           # training mean
                new[f"{f}_quadratic"] = X[f].to_numpy(float) ** 2
            else:
                from patsy import build_design_matrices
                d = build_design_matrices([self.design_info_[f]],
                                          {"x": X[f].to_numpy(float)})[0]
                arr = np.asarray(d)
                for j in range(arr.shape[1]):
                    new[f"{f}_spline{j}"] = arr[:, j]
        if not new:
            return X
        return pd.concat([X, pd.DataFrame(new, index=X.index)], axis=1)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def cox_risk_score(model, x):
    """
    Scalar risk ordering used for discrimination and for SHAP.

    For CoxPHSurvivalAnalysis this is the linear predictor. For
    RandomSurvivalForest, model.predict returns the sum of the ensemble
    cumulative hazard function over the distinct training event times: an
    unbounded, monotone ordering, not a probability and not tied to a horizon.
    """
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x)).ravel()
    if hasattr(model, "predict"):
        return np.asarray(model.predict(x)).ravel()
    if hasattr(model, "predict_partial_hazard"):
        return np.asarray(model.predict_partial_hazard(x)).ravel()
    raise AttributeError("model has no usable scoring method")


def batch_risk_scores(model, X, batch_size=1024):
    n = X.shape[0]
    out = np.empty(n, dtype=float)
    for start in range(0, n, batch_size):
        stop = start + batch_size
        out[start:stop] = cox_risk_score(model, X.iloc[start:stop])
    return out


class MetricEval:
    """Discrimination metrics with bootstrap confidence intervals.
    
    Each bootstrap replicate scores the resampled observations."""

    def __init__(self, num_of_b=N_BOOT, seed=RANDOM_STATE, n_jobs=N_JOBS, times=None):
        self.num_of_b = num_of_b
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.times = times      # evaluation times for time-dependent AUC and Brier score
        self.n_jobs = n_jobs
        self._seeds = None      # fixed resamples, drawn on first use
        self._cache = {}        # C-index replicates per score, keyed by content

    # ---- point estimates -------------------------------------------------- #

    @staticmethod
    def cindex(event, time, s):
        from sksurv.metrics import concordance_index_censored
        c, *_ = concordance_index_censored(np.asarray(event, bool),
                                           np.asarray(time, float),
                                           np.asarray(s, float))
        return float(c)

    @staticmethod
    def uno_cindex(y_train, y_test, s, tau=None):
        """Uno's C: inverse-probability-of-censoring weighted (comment 27)."""
        from sksurv.metrics import concordance_index_ipcw
        if tau is None:
            tau = float(np.percentile(np.asarray(y_test["time"], float), 90))
        c, *_ = concordance_index_ipcw(y_train, y_test, np.asarray(s, float), tau=tau)
        return float(c), tau

    @staticmethod
    def time_dependent_auc(y_train, y_test, s, times):
        from sksurv.metrics import cumulative_dynamic_auc
        auc, mean_auc = cumulative_dynamic_auc(y_train, y_test,
                                               np.asarray(s, float), times)
        return dict(zip([float(t) for t in times], [float(a) for a in auc])), float(mean_auc)

    @staticmethod
    def brier(model, y_train, y_test, X_test, times):
        from sksurv.metrics import brier_score, integrated_brier_score
        surv = np.vstack([[float(fn(t)) for t in times]
                             for fn in model.predict_survival_function(X_test)])
        _, bs = brier_score(y_train, y_test, surv, times)
        ibs = integrated_brier_score(y_train, y_test, surv, times)
        return dict(zip([float(t) for t in times], [float(b) for b in bs])), float(ibs)

    # ---- bootstrap -------------------------------------------------------- #

    def _boot_seeds(self):
        """One fixed set of resamples for every model this evaluator scores.

        Drawn once and reused: every model is then evaluated on the same
        resampled patients, which is what makes a comparison between two models
        paired, and it lets each model's replicates be computed once and reused
        (see boot_replicates). Workers receive seeds rather than sharing
        self.rng, so the result depends on self.seed and not on scheduling.
        """
        if self._seeds is None or len(self._seeds) != self.num_of_b:
            self._seeds = self.rng.integers(0, 2 ** 31 - 1, size=self.num_of_b)
            self._cache = {}
        return self._seeds

    def boot_replicates(self, y_test, s):
        """C-index of score s on each fixed resample, NaN where a resample has
        no events. Cached by content, so a model scored in full_report and again
        in boot_cindex_diff, or a baseline compared against several models, is
        computed once. At n = 44,000 one C-index takes about 0.6 s, and before
        caching each augmented variant needed 3,000 of them."""
        event = np.asarray(y_test["event"], bool)
        time = np.asarray(y_test["time"], float)
        s = np.asarray(s, float)
        seeds = self._boot_seeds()
        key = (hash(event.tobytes()), hash(time.tobytes()), hash(s.tobytes()), len(seeds))
        if key in self._cache:
            return self._cache[key]
        n = len(time)

        def one(sd):
            idx = np.random.default_rng(sd).integers(0, n, size=n)
            if event[idx].sum() == 0:
                return np.nan
            return self.cindex(event[idx], time[idx], s[idx])

        from joblib import Parallel, delayed
        boot = np.asarray(Parallel(n_jobs=self.n_jobs, batch_size=16)(
            delayed(one)(sd) for sd in seeds), dtype=float)
        self._cache[key] = boot
        return boot

    def boot_metric(self, y_test, s, metric_type="c-index"):
        if metric_type != "c-index":
            raise ValueError(f"unsupported metric: {metric_type}")
        event = np.asarray(y_test["event"], bool)
        time = np.asarray(y_test["time"], float)
        s = np.asarray(s, float)
        val_obs = self.cindex(event, time, s)
        boot = self.boot_replicates(y_test, s)
        boot = boot[~np.isnan(boot)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return val_obs, (float(lo), float(hi))

    # backwards-compatible alias
    boot_matric = boot_metric

    def boot_cindex_diff(self, y_test, s1, s2):
        """Paired bootstrap for c-index(s2) - c-index(s1) (comments 28, 29).

        Paired: both scores are evaluated on the same resampled patients within
        a replicate, so the shared sampling variation cancels and the interval
        is for the difference, not the difference of two independent intervals.
        """
        event = np.asarray(y_test["event"], bool)
        time = np.asarray(y_test["time"], float)
        s1, s2 = np.asarray(s1, float), np.asarray(s2, float)
        d_obs = self.cindex(event, time, s2) - self.cindex(event, time, s1)
        # same resamples for both scores, so the difference is paired; each
        # score's replicates come from the cache when it has been seen before
        diffs = self.boot_replicates(y_test, s2) - self.boot_replicates(y_test, s1)
        diffs = diffs[~np.isnan(diffs)]
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        p_two = min(1.0, 2 * min((diffs >= 0).mean(), (diffs <= 0).mean()))
        return float(d_obs), (float(lo), float(hi)), float(p_two)

    def cal_metric_CI(self, model, data, y_test, metric="c-index", label=""):
        s = cox_risk_score(model, data)
        d, (lo, hi) = self.boot_metric(y_test, s, metric)
        print(f"{label}{metric} = {d:.3f} (95% CI {lo:.3f}-{hi:.3f})")
        return d, (lo, hi)

    def full_report(self, model, X_train, y_train, X_test, y_test,
                    times=None, label=""):
        """Harrell's C with CI, Uno's C, time-dependent AUC, Brier/IBS. times
        defaults to the evaluator's times, else to 90-330 days (DataLoch)."""
        if times is None:
            times = self.times if self.times is not None else (90.0, 180.0, 270.0, 330.0)
        s = cox_risk_score(model, X_test)
        c, (lo, hi) = self.boot_metric(y_test, s)
        times = [t for t in times
                 if t > float(np.min(y_test["time"])) and t < float(np.max(y_test["time"]))]
        rep = dict(model=label, harrell_c=c, harrell_ci_low=lo, harrell_ci_high=hi)
        try:
            uno, tau = self.uno_cindex(y_train, y_test, s)
            rep.update(uno_c=uno, uno_tau=tau)
        except Exception as exc:                     # noqa: BLE001
            rep.update(uno_c=np.nan, uno_note=str(exc)[:120])
        if times:
            try:
                auc, mean_auc = self.time_dependent_auc(y_train, y_test, s, times)
                rep.update(mean_auc=mean_auc, auc_by_time=json.dumps(auc))
            except Exception as exc:                 # noqa: BLE001
                rep.update(mean_auc=np.nan, auc_note=str(exc)[:120])
            if hasattr(model, "predict_survival_function"):
                try:
                    bs, ibs = self.brier(model, y_train, y_test, X_test, times)
                    rep.update(ibs=ibs, brier_by_time=json.dumps(bs))
                except Exception as exc:             # noqa: BLE001
                    rep.update(ibs=np.nan, brier_note=str(exc)[:120])
        return rep


class CalibrationPerform:
    """Calibration at a fixed horizon t0 with bin-level bootstrap intervals.
    
    The conventional slope is estimated by refitting the linear predictor on
    test data. The binned weighted least squares slope is reported separately."""

    def __init__(self, t0, n_bins=10, kind="survival", n_boot=500,
                 random_state=0, save_folder=RESULT_DIR, model_name="model",
                 min_bin_n=20):
        self.t0 = t0
        self.n_bins = n_bins
        self.kind = kind
        self.n_boot = n_boot
        self.random_state = random_state
        self.save_folder = save_folder
        self.model_name = model_name
        self.min_bin_n = min_bin_n

    # ---- helpers ---------------------------------------------------------- #

    @staticmethod
    def extract_survival_estimator(model):
        if hasattr(model, "predict_survival_function"):
            return model
        if hasattr(model, "named_steps"):
            for _, step in model.named_steps.items():
                if hasattr(step, "predict_survival_function"):
                    return step
        raise AttributeError("no estimator with predict_survival_function")

    def quantile_bins(self, x):
        edges = np.unique(np.quantile(x, np.linspace(0, 1, self.n_bins + 1)))
        return edges if edges.size >= 2 else np.array([x.min(), x.max()])

    def km_s_at(self, time, event):
        from sksurv.nonparametric import kaplan_meier_estimator
        if time.size == 0:
            return np.nan
        t, s = kaplan_meier_estimator(event, time)
        if self.t0 <= t.min():
            return 1.0
        if self.t0 > t.max():
            return np.nan
        return float(np.interp(self.t0 - 1e-12, t, s))

    # ---- binned observed vs predicted ------------------------------------- #

    def survival_calibration(self, X, y, model):
        rng = np.random.default_rng(self.random_state)
        est = self.extract_survival_estimator(model)
        pred_surv = np.array([float(sf(self.t0))
                              for sf in est.predict_survival_function(X, return_array=False)])
        pred_cal = pred_surv if self.kind == "survival" else 1.0 - pred_surv

        edges = self.quantile_bins(pred_cal)
        bin_idx = np.digitize(pred_cal, edges[1:-1], right=True)
        time = np.asarray(y["time"], float)
        event = np.asarray(y["event"], bool)

        rows = []
        for b in range(edges.size - 1):
            mask = bin_idx == b
            if mask.sum() < self.min_bin_n:
                continue
            s_hat = self.km_s_at(time[mask], event[mask])
            if np.isnan(s_hat):
                continue
            obs = s_hat if self.kind == "survival" else 1.0 - s_hat
            row = dict(bin=b, n=int(mask.sum()), pred=float(pred_cal[mask].mean()),
                       obs=float(obs))
            if self.n_boot > 0:
                boot = []
                m = int(mask.sum())
                for _ in range(self.n_boot):
                    idx = rng.integers(0, m, m)
                    sb = self.km_s_at(time[mask][idx], event[mask][idx])
                    if not np.isnan(sb):
                        boot.append(sb if self.kind == "survival" else 1.0 - sb)
                if boot:
                    row["obs_ci_low"], row["obs_ci_high"] = np.percentile(boot, [2.5, 97.5])
            rows.append(row)
        return pd.DataFrame(rows)

    def binned_slope(self, X, y, model):
        """Weighted least squares through the bin means (the earlier estimator)."""
        b = self.survival_calibration(X, y, model)
        if len(b) < 2:
            return dict(intercept=np.nan, slope=np.nan, n_bins_eff=len(b))
        w = b["n"].to_numpy(float)
        A = np.c_[np.ones(len(b)), b["pred"].to_numpy(float)]
        beta = np.linalg.inv(A.T @ (w[:, None] * A)) @ (A.T @ (w * b["obs"].to_numpy(float)))
        return dict(intercept=float(beta[0]), slope=float(beta[1]),
                    n_bins_eff=int(len(b)), min_bin_n=int(b["n"].min()),
                    bins=b)

    @staticmethod
    def conventional_slope(model, X_test, y_test, t0):
        """
        Calibration slope on a scale comparable across model types: refit a
        Cox model on the test data with the complementary log-log of predicted
        survival at t0 as the only covariate. Slope 1 is ideal.

        The complementary log-log transformation gives forest and Cox
        predictions a common scale. For Cox models, it is linear in the
        linear predictor.
        """
        from lifelines import CoxPHFitter
        s_t0 = np.array([float(fn(t0)) for fn in model.predict_survival_function(X_test)])
        s_t0 = np.clip(s_t0, 1e-6, 1 - 1e-6)
        d = pd.DataFrame({"time": np.asarray(y_test["time"], float),
                          "event": np.asarray(y_test["event"], int),
                          "cll": np.log(-np.log(s_t0))})
        s = CoxPHFitter().fit(d, "time", "event").summary.loc["cll"]
        return dict(slope=float(s["coef"]),
                    slope_ci_low=float(s["coef lower 95%"]),
                    slope_ci_high=float(s["coef upper 95%"]),
                    slope_p=float(s["p"]))

    def report(self, model, X_test, y_test, label=""):
        out = {"model": label, "t0": self.t0, "kind": self.kind}
        binned = self.binned_slope(X_test, y_test, model)
        bins = binned.pop("bins", None)
        out.update({f"binned_{k}": v for k, v in binned.items()})
        try:
            out.update({f"conventional_{k}": v
                        for k, v in self.conventional_slope(model, X_test, y_test, self.t0).items()})
        except Exception as exc:                     # noqa: BLE001
            out["conventional_note"] = str(exc)[:120]
        if bins is not None:
            write_tsv(bins, os.path.join(self.save_folder,
                                         f"calibration_bins_{label or self.model_name}.tsv"))
        return out

    def calib_plot(self, model_lst, data_lst, model_labels=None, ax=None, title=None,
                  filename=None):
        """
        Overlay the empirical calibration curve (KM-observed vs. mean-predicted,
        joined across bins) for one or more models on a single axis, each with
        its own CI whiskers, plus the diagonal reference line.

        model_lst / data_lst: parallel lists, data_lst[i] = (X_i, y_i).
        model_labels: names for the legend; defaults to self.model_name if it
                      is a list of the same length, else "model 0", "model 1", ...
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=(5, 5))

        if model_labels is None:
            names = self.model_name if isinstance(self.model_name, (list, tuple)) else None
            model_labels = (list(names) if names and len(names) == len(model_lst)
                            else [f"model {i}" for i in range(len(model_lst))])

        all_pred, all_obs = [], []
        for i, mdl in enumerate(model_lst):
            X_i, y_i = data_lst[i]
            b = self.survival_calibration(X_i, y_i, mdl)
            if not len(b):
                print(f"[calib_plot] {model_labels[i]}: no bin had >= {self.min_bin_n} "
                     "patients, nothing to plot")
                continue
            pred, obs = b["pred"].to_numpy(float), b["obs"].to_numpy(float)
            all_pred.append(pred)
            all_obs.append(obs)
            line, = ax.plot(pred, obs, marker="o", linestyle="-",
                            label=f"Observed (KM)_{model_labels[i]}")
            if "obs_ci_low" in b.columns:                          # per-model CI, inside the loop
                ax.vlines(pred, b["obs_ci_low"].to_numpy(float),
                          b["obs_ci_high"].to_numpy(float),
                          color=line.get_color(), alpha=0.6, linewidth=1.2)

        y_label = "Observed survival" if self.kind == "survival" else "Observed risk"
        x_label = "Predicted survival" if self.kind == "survival" else "Predicted risk"

        # Set axis limits from all plotted models.
        pooled = np.concatenate(all_pred + all_obs) if all_pred else np.array([0.0, 1.0])
        lo_d, hi_d = float(pooled.min()), float(pooled.max())
        pad = max(0.02, 0.08 * (hi_d - lo_d))
        minv, maxv = max(0.0, lo_d - pad), min(1.0, hi_d + pad)
        ax.plot([minv, maxv], [minv, maxv], color="black", linestyle="--",
                linewidth=1.2, label="Perfect calibration")
        ax.set_xlim(minv, maxv)
        ax.set_ylim(minv, maxv)
        ax.set_xlabel(f"{x_label} at t0={self.t0:g}")
        ax.set_ylabel(y_label)
        ax.set_title(title or f"Calibration plot ({self.kind})")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

        if own_fig:
            fig.tight_layout()
            safe_names = "_".join(model_labels)
            path = os.path.join(self.save_folder,
                                filename or f"Calibration_plot_{safe_names}.pdf")
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return path
        return ax

    def is_stable(self, model, X_test, y_test):
        """Return whether the conventional slope's 95% confidence interval contains 1."""
        c = self.conventional_slope(model, X_test, y_test, self.t0)
        return bool(c["slope_ci_low"] <= 1.0 <= c["slope_ci_high"]), c


def ph_assumption_report(X_train, y_train, out=None, penalizer=0.01,
                         time_transform="km"):
    """Run Schoenfeld residual tests for proportional hazards.
    
    A small penalizer stabilizes fitting with wide design matrices."""
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test

    d = X_train.copy()
    d["time"] = np.asarray(y_train["time"], float)
    d["event"] = np.asarray(y_train["event"], int)
    fitter = CoxPHFitter(penalizer=penalizer).fit(d, "time", "event")
    res = proportional_hazard_test(fitter, d, time_transform=time_transform)
    tab = res.summary.reset_index().rename(columns={"index": "term"})
    n_bad = int((tab["p"] < 0.05).sum())
    print(f"[PH] {n_bad}/{len(tab)} terms with p<.05; smallest p = {tab['p'].min():.2e}")
    if out:
        write_tsv(tab, out)
    return fitter, tab


def fit_and_score(X_train, y_train, X_test, y_test, evaluator, calib, label,
                  alpha=1e-6):
    # Match the tie handling used by lifelines for coefficient intervals.
    model = CoxPHSurvivalAnalysis(alpha=alpha, ties="efron").fit(X_train, y_train)
    rep = evaluator.full_report(model, X_train, y_train, X_test, y_test, label=label)
    rep.update({f"calib_{k}": v for k, v in
                calib.report(model, X_test, y_test, label=label).items()
                if k not in ("model",)})
    rep["n_parameters"] = X_train.shape[1]
    return model, rep


def partial_loglik_contrib(time, event, lp):
    """Per-event Cox partial log-likelihood contributions on held-out data,
    Breslow ties: lp_i - log sum_{t_j >= t_i} exp(lp_j) for each event i.
    Used to choose a penalty: it uses every event's information, so it is far
    more stable than a C-index computed on the same validation set."""
    time, event, lp = np.asarray(time, float), np.asarray(event, bool), np.asarray(lp, float)
    o = np.argsort(-time, kind="mergesort")
    t, l = time[o], lp[o]
    m = l.max()
    cs = np.cumsum(np.exp(l - m))
    ut, first = np.unique(-t, return_index=True)
    last = np.r_[first[1:], len(t)] - 1
    logrisk = np.log(cs[last[np.searchsorted(ut, -t)]]) + m
    return (l - logrisk)[event[o]]


def _coxnet_cv(Ptr, Pte, names, y_train, y_test, evaluator, calib, label, n_folds=10,
               return_scores=False):
    """LASSO Cox as cv.glmnet(family = "cox") is used in most clinical
    applications: LASSO penalty on every coefficient, standardized columns, and
    the penalty chosen by 10-fold cross-validated partial likelihood
    (lambda.min), using the grouped cross-validated partial likelihood of
    Verweij and van Houwelingen that glmnet uses by default. The test set is not
    used. The penalty grid has 50 values down to 1% of the largest, shorter than
    glmnet's default, to keep 10 path fits feasible on a large all-pairs design;
    a warning is printed if the chosen value lies on the edge of the grid.
    Ptr and Pte are the training and test designs, names their column names
    (interactions contain '__x__')."""
    from sklearn.model_selection import StratifiedKFold
    ok = Ptr.std(axis=0) > 0
    Ptr, Pte, names = Ptr[:, ok], Pte[:, ok], [nm for nm, k in zip(names, ok) if k]
    mu, sd = Ptr.mean(axis=0), Ptr.std(axis=0)
    Ptr = (Ptr - mu) / sd
    Pte = (Pte - mu) / sd
    is_int = np.array(["__x__" in nm for nm in names])
    print(f"[{label}] design: {Ptr.shape[1]} columns "
          f"({int((~is_int).sum())} main and non-linear terms, {int(is_int.sum())} interactions)")
 
    lasso = dict(l1_ratio=1.0, max_iter=100000)
    full = CoxnetSurvivalAnalysis(n_alphas=50, alpha_min_ratio=0.01,
                                  fit_baseline_model=True, **lasso).fit(Ptr, y_train)
    alphas = full.alphas_
 
    def pl(time, event, lp):
        return partial_loglik_contrib(time, event, lp).sum()
 
    cvpl = np.zeros(len(alphas))
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    for itr, _ in folds.split(Ptr, np.asarray(y_train["event"])):
        net_k = CoxnetSurvivalAnalysis(alphas=alphas, **lasso).fit(Ptr[itr], y_train[itr])
        lp_all = Ptr @ net_k.coef_                       # (n, n_alphas)
        lp_tr = lp_all[itr]
        for j in range(net_k.coef_.shape[1]):
            cvpl[j] += (pl(y_train["time"], y_train["event"], lp_all[:, j])
                        - pl(y_train["time"][itr], y_train["event"][itr], lp_tr[:, j]))
    k = int(np.argmax(cvpl))
    if k in (0, len(alphas) - 1):
        print(f"[{label}] WARNING: chosen penalty is at the edge of the grid")
 
    class _AtPenalty:
        """The full-path fit, evaluated at the cross-validated penalty."""
        def __init__(self, net, a):
            self.net, self.a = net, a
        def predict(self, X):
            return self.net.predict(X, alpha=self.a)
        def predict_survival_function(self, X, return_array=False):
            return self.net.predict_survival_function(X, alpha=self.a, return_array=return_array)
 
    beta = full.coef_[:, k]
    n_main = int(np.sum((np.abs(beta) > 1e-8) & ~is_int))
    n_int = int(np.sum((np.abs(beta) > 1e-8) & is_int))
    print(f"[{label}] lambda.min at step {k + 1}/{len(alphas)} ({n_folds}-fold CV partial likelihood); "
          f"non-zero: {n_main} main and non-linear terms, {n_int} interactions")
    model = _AtPenalty(full, alphas[k])
    rep = evaluator.full_report(model, Ptr, y_train, Pte, y_test, label=label)
    rep.update({f"calib_{key}": value for key, value in
                calib.report(model, Pte, y_test, label=label).items()
                if key != "model"})
    rep.update(n_parameters=Ptr.shape[1], n_main_nonzero=n_main, n_interactions_nonzero=n_int,
               alpha=float(alphas[k]), alpha_step=k + 1, cv_folds=n_folds)
    if return_scores:
        return rep, np.asarray(model.predict(Pte), dtype=float)
    return rep
 
 
def _pairwise(X, cols):
    """Products of every pair of columns, with names a__x__b."""
    A = X[cols].to_numpy(float)
    pairs = list(itertools.combinations(range(len(cols)), 2))
    prods = np.column_stack([A[:, i] * A[:, j] for i, j in pairs]) if pairs else np.empty((len(X), 0))
    return prods, [f"{cols[i]}__x__{cols[j]}" for i, j in pairs]
 
 
def comparator_models(X_train, X_test, y_train, y_test, contin_cols,
                      evaluator, calib, max_pairs=None, n_folds=10, baseline_scores=None):
    """Evaluate a spline Cox model and a penalised Cox model.
 
    The spline model expands continuous predictors. The penalised model includes
    main effects and all pairwise interactions. Paired C-index differences use
    Cox_org as the reference; positive differences favour the comparator."""
    out = []
    if baseline_scores is None:
        baseline = CoxPHSurvivalAnalysis(alpha=1e-6, ties="efron").fit(X_train, y_train)
        baseline_scores = cox_risk_score(baseline, X_test)

    def add_difference(report, scores):
        d, (lo, hi), p = evaluator.boot_cindex_diff(y_test, baseline_scores, scores)
        report.update(delta_c_vs_baseline=d, delta_ci_low=lo, delta_ci_high=hi, delta_p=p)
 
    rcs = NonlinearTransform(contin_cols, kind="spline", df_spline=4).fit(X_train)
    Xtr = rcs.transform(X_train).drop(columns=list(contin_cols), errors="ignore")
    Xte = rcs.transform(X_test).drop(columns=list(contin_cols), errors="ignore")
    model, rep = fit_and_score(Xtr, y_train, Xte, y_test, evaluator, calib, "cox_rcs")
    add_difference(rep, cox_risk_score(model, Xte))
    out.append(rep)
 
    cols = list(X_train.columns)
    if max_pairs is not None and len(cols) * (len(cols) - 1) // 2 > max_pairs:
        keep = (X_train.std().sort_values(ascending=False)
                .head(int(np.sqrt(2 * max_pairs))).index.tolist())
        cols = keep
        print(f"[coxnet] restricting all-pairs design to {len(cols)} features")
 
    # Penalised Cox over main effects and all pairwise interactions, fitted
    # without reference to the attributions
    Qtr, pnames = _pairwise(X_train, cols)
    Qte, _ = _pairwise(X_test, cols)
    Ptr = np.column_stack([X_train[cols].to_numpy(float), Qtr])
    Pte = np.column_stack([X_test[cols].to_numpy(float), Qte])
    rep, scores = _coxnet_cv(Ptr, Pte, cols + pnames, y_train, y_test, evaluator, calib,
                             "coxnet_all_pairwise", n_folds, return_scores=True)
    add_difference(rep, scores)
    out.append(rep)
    return out
 
 
def coxnet_augmented(X_train, X_test, y_train, y_test, rec, evaluator, calib,
                     groups=None, n_folds=10, return_scores=False):
    """
    The final augmented Cox model fitted with a LASSO penalty. The design is
    exactly that of Cox_all (recommended exclusion, non-linear terms and the
    interactions that entered the model); only the fitting differs, with the
    penalty chosen as for coxnet_all_pairwise (10-fold cross-validated partial
    likelihood, lambda.min).
    """
    Atr, Ate = Recommender.apply(X_train, X_test, rec, variant="all", groups=groups)
    return _coxnet_cv(Atr.to_numpy(float), Ate.to_numpy(float), list(Atr.columns),
                      y_train, y_test, evaluator, calib, "Cox_all_lasso", n_folds,
                      return_scores=return_scores)

# =============================================================================
# Recommendation pipeline, identical to the DataLoch analysis
# =============================================================================
MARGIN_GRID = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)


def as_surv(y):
    """Outcome as a structured array with fields 'event' and 'time'."""
    time, event = _extract_time_event(y)
    return Surv.from_arrays(event=event, time=time)


def _subcohorts(model, X, y, eps):
    """Boolean masks of the low- and high-risk subcohorts at margin eps, on the
    log risk scale used by get_explanations."""
    time, event = _extract_time_event(y)
    lg = lambda s: np.log(np.maximum(np.asarray(s, dtype=float), 1e-12))
    pred = lg(model.predict(X))
    masks = []
    for ev in (False, True):
        ref = X[event == ev]
        centre = float(lg(model.predict(ref.mean().to_frame().T))[0])
        masks.append(np.abs(pred - centre) < np.std(lg(model.predict(ref))) * eps)
    return masks


# Share screening between the margin search and the final run.
def _screen_recommendations(X, y, shap_sel, continuous, ordinal, cutpoints,
                            groups, min_cell, seed, features=None, score_cache=None):
    rg = Recommender(min_cell=min_cell, groups=groups, seed=seed)
    excl = {lv: rg.exclusion(shap_sel[lv][0])[0] for lv in shap_sel}
    exclusion = sorted(set(excl['low']) & set(excl['high']))
    cand_nl = list(continuous) + list(ordinal)
    nonl = {lv: rg.nonlinear(shap_sel[lv][0], shap_sel[lv][1], cand_nl)[0] for lv in shap_sel}
    nonlinear = sorted(set(nonl['low']) | set(nonl['high']))
    dropped = set(exclusion) | {d for f in exclusion for d in groups.get(f, [])}
    nonlinear = [f for f in nonlinear if f not in dropped]
    wanted = set(rg._to_features(X.columns if features is None else features))
    screen = [c for c in X if c not in dropped and rg._to_features([c])[0] in wanted]
    tabs, diags = [], []
    if screen:
        for lv, (sh, sel) in shap_sel.items():
            t, diag = rg.interaction(sh, sel[screen], cutpoints=cutpoints,
                                    candidates=screen, subcohort=lv)
            diags.append(diag)
            if len(t):
                tabs.append(t)
    tests = pd.concat(tabs, ignore_index=True) if tabs else pd.DataFrame()
    _, tests = rg.merge_interaction_candidates(tests)
    n_screened = int(tests.loc[tests['selected'], 'pair'].nunique()) if len(tests) else 0
    tests, target = rg.target_model_filter(tests, X, y, list(continuous), cache=score_cache)
    selected = tests.loc[tests['selected']].drop_duplicates('pair') if len(tests) else tests
    spec = {}
    for _, r in selected.iterrows():
        spec.setdefault(r['strat'], []).append(r['partner'])
    diag = pd.concat(diags, ignore_index=True) if diags else pd.DataFrame()
    # An A/C feature for which no comparison could be run is not a negative
    # finding. Such a margin can be a fallback, but cannot establish stability.
    unassessed = int(((diag['pattern'] == 'A/C') & ~diag['screened']).sum()) if len(diag) else 0
    info = dict(tests=tests, target=target, feature_screen=diag,
                n_pairs_screened=n_screened, n_pairs_held=len(selected),
                n_focal_unassessed=unassessed)
    return dict(exclusion=exclusion, nonlinear=nonlinear, interaction=spec), info


def _recommendation_sets(rec):
    return dict(exclusion=set(rec['exclusion']), nonlinear=set(rec['nonlinear']),
                interaction={tuple(sorted((a, b))) for a, bs in rec['interaction'].items()
                             for b in bs})


def _window_similarity(window):
    """Minimum Jaccard across ALL pairs of margins in a consecutive window.
    Two empty sets agree, provided screening was possible (checked separately).
    """
    return {kind: min((len(a['sets'][kind] & b['sets'][kind]) /
                       len(a['sets'][kind] | b['sets'][kind])
                       if a['sets'][kind] | b['sets'][kind] else 1.)
                      for a, b in itertools.combinations(window, 2))
            for kind in ('exclusion', 'nonlinear', 'interaction')}


def choose_margin(model, X, y, cutpoints, features, min_cell=20, target=0.9, min_n=100,
                  grid=MARGIN_GRID, groups=None, *, continuous=(), ordinal=(),
                  stable_steps=2, nsamples='auto', seed=RANDOM_STATE,
                  result_cache=None, out=None):
    """Training-only choice by recommendation stability.

    Evaluate increasing margins with one fixed model and one fixed reference
    per subcohort. Compare exclusion, nonlinearity and post-Cox interaction
    candidates separately, before applying the Riley budget. Choose the smallest margin
    in the first consecutive `stable_steps` window whose pairwise Jaccards are
    all >= target. Defaults (3 margins, 0.90) are pragmatic, not statistical
    guarantees. If no stable window exists, use the largest successfully
    evaluated margin and report stable=False. No test-set input is accepted.

    `features` defines the candidate interaction features; continuous/ordinal
    must match the final recommendation call. Returns the margin, subcohort
    size, and diagnostics. result_cache is an optional output dictionary used by recommend
    to avoid recomputing the selected result. SHAP caches are local to this call.
    """
    groups = dict(groups or {})
    grid = sorted(set(map(float, grid)))
    if (not grid or not np.isfinite(grid).all() or grid[0] <= 0
            or not 0 < target <= 1 or int(stable_steps) != stable_steps or stable_steps < 2):
        raise ValueError('Use positive finite margins, 0 < target <= 1 and stable_steps >= 2')
    stable_steps = int(stable_steps)
    if not X.index.is_unique or len(X) != len(y):
        raise ValueError('Training data need unique patient indices and matching outcomes')
    yy = as_surv(y)
    pred = np.log(np.maximum(np.asarray(model.predict(X), float), 1e-12))
    if not np.isfinite(pred).all():
        raise ValueError('Non-finite training risk predictions')
    required = max(2 * min_cell, int(min_n))
    profiles = {}
    for level, event in [('low', False), ('high', True)]:
        mask = yy['event'] == event
        if not mask.any():
            raise ValueError(f'No training reference patients for {level}')
        ref = X.loc[mask].mean().to_frame().T
        centre = float(np.log(np.maximum(model.predict(ref), 1e-12))[0])
        profiles[level] = (np.abs(pred - centre), float(pred[mask].std()))
    cache = {lv: pd.DataFrame() for lv in profiles}
    score_cache = {}                  # fixed training Cox null; raw pair scores
    if result_cache is not None:
        result_cache.clear()
    rows, window = [], []
    chosen = last_available = None
    stable, chosen_scores = False, {}
    for eps in grid:
        masks = {lv: distance < eps * sd for lv, (distance, sd) in profiles.items()}
        row = dict(margin=eps, subcohort_low=int(masks['low'].sum()),
                   subcohort_high=int(masks['high'].sum()), status='insufficient_subcohort',
                   stability_eligible=False, n_exclusion=np.nan, n_nonlinear=np.nan,
                   n_interaction=np.nan, n_tests=0, n_pairs_tested=0,
                   n_focal_unassessed=np.nan, all_recommendations_empty=False,
                   n_new_shap_low=0, n_new_shap_high=0,
                   jaccard_exclusion=np.nan, jaccard_nonlinear=np.nan, jaccard_interaction=np.nan)
        print(f'[margin] eps={eps:g}: low={row["subcohort_low"]}, high={row["subcohort_high"]}', flush=True)
        if min(row['subcohort_low'], row['subcohort_high']) < required:
            window.clear()
        else:
            try:
                shap_sel = {}
                for level in ('low', 'high'):
                    index = X.index[masks[level]]
                    new = index[~index.isin(cache[level].index)]
                    row[f'n_new_shap_{level}'] = len(new)
                    if len(new):
                        print(f'[margin/{level}] SHAP for {len(new)} new patients', flush=True)
                        sh, _ = get_explanations(model, X, yy, eps, risk_level=level,
                                                nsamples=nsamples, groups=groups,
                                                selected_index=new, seed=seed)
                        cache[level] = sh if cache[level].empty else pd.concat([cache[level], sh])
                    shap_sel[level] = (cache[level].loc[index], X.loc[index])
                rec, info = _screen_recommendations(X, yy, shap_sel, continuous, ordinal,
                                                    cutpoints, groups, min_cell, seed,
                                                    features=features, score_cache=score_cache)
                sets = _recommendation_sets(rec)
                tests = info['tests']
                row.update(status='ok', stability_eligible=info['n_focal_unassessed'] == 0,
                        n_focal_unassessed=info['n_focal_unassessed'], n_tests=len(tests),
                        n_pairs_tested=int(tests['pair'].nunique()) if len(tests) else 0,
                        all_recommendations_empty=not any(sets.values()),
                        **{f'n_{kind}': len(v) for kind, v in sets.items()})
                entry = dict(margin=eps, rec=rec, info=info, sets=sets,
                            ready=row['stability_eligible'])
                last_available = entry
                window.append(entry)
                window = window[-stable_steps:]
                print(f'[margin] recommendations: exclusion={len(sets["exclusion"])}, '
                    f'nonlinear={len(sets["nonlinear"])}, interaction={len(sets["interaction"])} '
                    '(before parameter budget)', flush=True)
                if len(window) == stable_steps:
                    scores = _window_similarity(window)
                    row.update({f'jaccard_{kind}': v for kind, v in scores.items()})
                    if all(e['ready'] for e in window) and min(scores.values()) >= target:
                        chosen, stable, chosen_scores = window[0], True, scores
            except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                row.update(status='failed', error=str(exc))
                window.clear()       # failed comparisons do not count as empty agreement
                print(f'[margin] eps={eps:g} failed: {exc}', flush=True)
        rows.append(row)
        tab = pd.DataFrame(rows)
        if out:
            os.makedirs(os.path.dirname(os.fspath(out)) or '.', exist_ok=True)
            tab.to_csv(out, index=False)
        if stable:
            break
    if chosen is None:
        chosen = last_available
    if chosen is None:
        raise ValueError('No margin could be evaluated; inspect subgroup counts and margin errors')
    eps = chosen['margin']
    tab['selected'] = tab['margin'] == eps
    tab['stable'] = stable
    tab['selection_rule'] = ('stable_recommendations' if stable else 'largest_available_unstable')
    tab['similarity_threshold'], tab['stable_steps'] = target, stable_steps
    tab['last_evaluated_margin'] = rows[-1]['margin']
    tab['stability_window_end'] = window[-1]['margin'] if stable else np.nan
    for kind, value in chosen_scores.items():
        tab.loc[tab['selected'], f'jaccard_{kind}'] = value
    if out:
        tab.to_csv(out, index=False)
    if result_cache is not None:
        result_cache.update(rec=chosen['rec'], info=chosen['info'],
                            shap_sel={lv: (cache[lv].loc[X.index[d < eps * sd]], X.loc[d < eps * sd])
                                      for lv, (d, sd) in profiles.items()})
    selected = tab.loc[tab['selected']].iloc[0]
    print(f'[margin] chosen eps={eps:g}; ' + ('stable window found' if stable else
          'NOT STABLE: using the largest available margin'), flush=True)
    return eps, [int(selected['subcohort_low']), int(selected['subcohort_high'])], tab


def recommend(ex_model, X_train, y_train, continuous, ordinal, cutpoints, groups=None,
              min_cell=20, shrinkage=0.9, min_subcohort=100, tag='data', *,
              margin_grid=MARGIN_GRID, stability_target=0.9, stable_steps=2,
              nsamples='auto', seed=RANDOM_STATE):
    """Select candidates by stability, then apply the Riley parameter budget.
    """
    groups = dict(groups or {})
    y_s = as_surv(y_train)
    selected = {}
    eps, n_sub, margin_tab = choose_margin(
        ex_model, X_train, y_s, cutpoints, list(X_train.columns), min_cell=min_cell,
        min_n=min_subcohort, grid=margin_grid, groups=groups, continuous=continuous,
        ordinal=ordinal, target=stability_target, stable_steps=stable_steps,
        nsamples=nsamples, seed=seed, result_cache=selected,
        out=os.path.join(RESULT_DIR, f'{tag}_margin_stability.csv'))
    rec, info = selected['rec'], selected['info']
    rg = Recommender(min_cell=min_cell, groups=groups, seed=seed)
    for level, (df_shap, sel) in selected['shap_sel'].items():
        values, _ = rg._feature_frame(sel)
        make_plot(df_shap, values[df_shap.columns], f'{tag}_shap_{level}', plot_type='violin',
                  xlabel='SHAP value (log risk)', figsize=(8, 6))
    exclusion, nonlinear = rec['exclusion'], rec['nonlinear']
    tests, target = info['tests'], info['target']
    base_frame, _ = Recommender.apply(X_train, X_train, dict(exclusion=exclusion, nonlinear=nonlinear,
                                                             interaction={}), variant='all',
                                      groups=groups)
    riley = Recommender.riley_parameter_budget(base_frame, y_s, shrinkage=shrinkage)
    budget = max(0, riley['p_max'] - base_frame.shape[1])
    spec, budget_log = Recommender.spec_within_budget(tests, base_frame, budget, groups)
    rec = dict(exclusion=exclusion, nonlinear=nonlinear, interaction=spec)
    summary = dict(margin=eps, margin_stable=bool(margin_tab['stable'].iloc[0]),
                   margin_selection_rule=margin_tab['selection_rule'].iloc[0],
                   subcohort_low=n_sub[0], subcohort_high=n_sub[1],
                   n_tests=len(tests), n_pairs_tested=int(tests['pair'].nunique()) if len(tests) else 0,
                   n_pairs_screened=info['n_pairs_screened'], n_pairs_held=info['n_pairs_held'],
                   riley_p_max=riley['p_max'], interaction_budget=budget,
                   n_pairs_entered=sum(len(v) for v in spec.values()))
    print('RECOMMENDATIONS:', rec)
    print('SUMMARY:', summary)
    return rec, dict(summary=summary, tests=tests, target=target, budget_log=budget_log,
                     margin=margin_tab)


def evaluation_times(y):
    """Quartiles of the event times: where time-dependent AUC and Brier score
    are evaluated, so that each data set is assessed within its own follow-up."""
    t = np.asarray(y['time'])[np.asarray(y['event'], dtype=bool)]
    return tuple(float(v) for v in np.quantile(t, [0.25, 0.5, 0.75]))


def evaluate_recommendations(rec, X_train, X_test, y_train, y_test, t0, tag, continuous,
                             groups=None, times=None, n_bins=5, comparators=True,
                             save_folder='plots/', results_folder='results/'):
    """
    Fit the Cox model without recommendations, with each component and with all
    of them, and evaluate each once on the test set, as in the DataLoch
    analysis: Harrell's C with bootstrap CI, Uno's C, time-dependent AUC, Brier
    score and integrated Brier score, calibration at t0 (binned and
    conventional slope), and the paired bootstrap difference in C from the
    model without recommendations. The comparators are a Cox model with
    restricted cubic splines for continuous features and a LASSO Cox model over
    all pairwise interactions, plus LASSO models on the original and recommended
    feature designs. times defaults to the quartiles of the event
    times in the training data.
    """
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)
    evaluator = MetricEval(times=times if times is not None else evaluation_times(y_train))
    calib = CalibrationPerform(t0=t0, n_bins=n_bins, kind='survival', save_folder=save_folder)
    base, rep = fit_and_score(X_train, y_train, X_test, y_test, evaluator, calib, 'Cox_org')
    calib.calib_plot([base], [(X_test, y_test)], model_labels=['Cox_org'],
                     filename=f'calibration_{tag}_Cox_org.pdf')
    reports = [rep]
    s_base = cox_risk_score(base, X_test)
    final = None
    for variant, name in [('exclusion', 'Cox_exclu'), ('nonlinear', 'Cox_nonlinear'),
                          ('interaction', 'Cox_inter'), ('all', 'Cox_all')]:
        Xtr, Xte = Recommender.apply(X_train, X_test, rec, variant=variant, groups=groups)
        model, rep = fit_and_score(Xtr, y_train, Xte, y_test, evaluator, calib, name)
        d, (lo, hi), p = evaluator.boot_cindex_diff(y_test, s_base, cox_risk_score(model, Xte))
        rep.update(delta_c_vs_baseline=d, delta_ci_low=lo, delta_ci_high=hi, delta_p=p)
        reports.append(rep)
        calib.calib_plot([base, model], [(X_test, y_test), (Xte, y_test)],
                         model_labels=['Cox_org', name],
                         filename=f'calibration_{tag}_Cox_org_vs_{name}.pdf')
        if variant == 'all':
            final = (model, Xtr)
            s_all = cox_risk_score(model, Xte)
    if comparators:
        reports += comparator_models(X_train, X_test, y_train, y_test, list(continuous),
                                     evaluator, calib, baseline_scores=s_base)
    rep, s_lasso = coxnet_augmented(X_train, X_test, y_train, y_test, rec, evaluator, calib,
                                    groups=groups, return_scores=True)
    d, (lo, hi), p = evaluator.boot_cindex_diff(y_test, s_base, s_lasso)
    rep.update(delta_c_vs_baseline=d, delta_ci_low=lo, delta_ci_high=hi, delta_p=p)
    d, (lo, hi), p = evaluator.boot_cindex_diff(y_test, s_all, s_lasso)
    rep.update(delta_c_vs_cox_all=d, delta_vs_cox_all_ci_low=lo, delta_vs_cox_all_ci_high=hi,
               delta_vs_cox_all_p=p)
    reports.append(rep)
    tab = pd.DataFrame(reports)
    tab.to_csv(os.path.join(results_folder, f'{tag}_model_comparison.csv'), index=False)
    model, Xtr = final
    pd.Series(model.coef_, index=Xtr.columns, name='coef').to_csv(
        os.path.join(results_folder, f'{tag}_final_cox_coefficients.csv'))
    ph_assumption_report(Xtr, y_train, out=os.path.join(results_folder, f'{tag}_ph_test_final.csv'))
    cols = ['model', 'harrell_c', 'harrell_ci_low', 'harrell_ci_high', 'uno_c', 'delta_c_vs_baseline',
            'delta_ci_low', 'delta_ci_high', 'calib_conventional_slope', 'n_parameters', 'n_nonzero']
    print(tab[[c for c in cols if c in tab.columns]].round(4).to_string(index=False))
    return tab
