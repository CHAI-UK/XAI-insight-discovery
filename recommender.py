"""
recommender.py — exclusion / non-linearity / interaction recommendations from
saved SHAP files, and application of them to the design matrix.

One shared idea runs through two of the three rules. _basis(x) represents a
feature's own contribution flexibly: indicator columns when x takes few values,
a restricted cubic spline when it is continuous. Comparing a model built on
that basis against a straight line in x answers both questions:

  non-linearity  does phi_x deviate from a linear function of x?
                 (replaces |r| < 0.1, which conflates "no effect" with
                 "non-linear effect": on simulated data a monotone quadratic
                 gave |r| = 0.98 and was missed, pure noise gave |r| = 0.004
                 and was flagged)
  interaction    does phi_x vary among patients who share a value of x?
                 Under additivity phi_x = h(x) - E[h(x)] is a deterministic
                 function of x, so residual dispersion implies effect
                 modification (Patterns A and C); none implies Pattern B.

Stratification is always applied to the observed feature value at a
pre-specified cut point, never to the attribution: phi_x is a function of the
whole feature vector, so splitting on it conditions on the candidate modifier
itself. Within strata the contrast E[phi_y|y=1] - E[phi_y|y=0] is compared,
which is invariant to stratum composition under additivity, unlike the
marginal attribution distribution.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import chi2, f as fdist, norm
from statsmodels.stats.multitest import multipletests


class CoxScoreTester:
    """Score test for adding columns Z to a fitted Cox model (Breslow ties).

    The base model is fitted once. Each candidate then needs only grouped sums
    over the distinct event times, so hundreds of candidates can be tested on a
    large cohort without refitting. The information is the efficient one,
    I_zz - I_zb I_bb^-1 I_bz, which accounts for the base covariates. On
    Rotterdam its P values agreed with refitted likelihood ratio tests to a
    correlation of 0.998 on the log scale.
    """

    def __init__(self, B, lp, time, event):
        from scipy.sparse import csr_matrix
        time = np.asarray(time, float)
        self.event = np.asarray(event, bool)
        ut, self.bin = np.unique(time, return_inverse=True)
        self.K, n = len(ut), len(time)
        self.Ind = csr_matrix((np.ones(n), (self.bin, np.arange(n))), shape=(self.K, n))
        self.IndE = csr_matrix((np.ones(self.event.sum()),
                                (self.bin[self.event], np.flatnonzero(self.event))),
                               shape=(self.K, n))
        self.w = np.exp(np.asarray(lp, float) - np.max(lp))
        self.B = np.asarray(B, float)
        self.d = np.asarray(self.IndE.sum(axis=1)).ravel()
        self.S0 = self._risk(self.w[:, None])[:, 0]
        self.mb = self._risk(self.w[:, None] * self.B) / self.S0[:, None]
        p = self.B.shape[1]
        Ibb = np.zeros((p, p))
        for j in range(p):
            S2 = self._risk(self.w[:, None] * self.B[:, [j]] * self.B) / self.S0[:, None]
            Ibb[j] = (self.d[:, None] * (S2 - self.mb[:, [j]] * self.mb)).sum(0)
        self.Ibb_inv = np.linalg.pinv(Ibb)

    def _risk(self, a):
        """Sums over the risk set (all subjects with t >= t_k) at each distinct time."""
        g = np.asarray(self.Ind @ a)
        return np.cumsum(g[::-1], axis=0)[::-1]

    def test(self, Z):
        Z = np.asarray(Z, float).reshape(len(self.w), -1)
        k = Z.shape[1]
        wz = self.w[:, None] * Z
        mz = self._risk(wz) / self.S0[:, None]
        U = (np.asarray(self.IndE @ Z) - self.d[:, None] * mz).sum(0)
        Izz = np.zeros((k, k))
        Izb = np.zeros((k, self.B.shape[1]))
        for j in range(k):
            S2zz = self._risk(wz[:, [j]] * Z) / self.S0[:, None]
            Izz[j] = (self.d[:, None] * (S2zz - mz[:, [j]] * mz)).sum(0)
            S2zb = self._risk(wz[:, [j]] * self.B) / self.S0[:, None]
            Izb[j] = (self.d[:, None] * (S2zb - mz[:, [j]] * self.mb)).sum(0)
        info = Izz - Izb @ self.Ibb_inv @ Izb.T
        stat = float(U @ np.linalg.pinv(info) @ U)
        return stat, float(chi2.sf(stat, k))


class Recommender:

    def __init__(self, res_dir='data/result', frac=0.05, alpha=0.05,
                 n_boot=500, min_delta_r2=0.01, min_cell=100, n_bins=10,
                 df_spline=4, residual_thresh=0.05, min_abs_effect=0.10,
                 n_boot_screen=300, se_method='sandwich', seed=1,
                 dispersion_thresh=0.10,
                 groups=None, min_level=5, df_screen=8):
        self.res, self.frac, self.alpha = res_dir, frac, alpha
        self.n_boot, self.min_delta_r2 = n_boot, min_delta_r2
        self.min_cell, self.n_bins, self.df_spline = min_cell, n_bins, df_spline
        self.residual_thresh, self.min_abs_effect = residual_thresh, min_abs_effect
        # Gate interactions by within-value attribution dispersion.
        # residual_thresh is retained for API compatibility and is unused.
        # one-hot groups {parent: [dummy columns]}: nominal features are
        # recombined from their dummies and screened as one feature
        self.groups = dict(groups or {})
        self.min_level = min_level
        self.df_screen = df_screen
        self.dispersion_thresh = dispersion_thresh
        self.n_boot_screen = n_boot_screen
        if se_method not in ('sandwich', 'bootstrap'):
            raise ValueError("se_method must be 'sandwich' or 'bootstrap'")
        self.se_method = se_method
        self.rng = np.random.default_rng(seed)

    def load(self, tag):
        s = pd.read_csv(f'{self.res}/shap_values_{tag}.tsv', sep='\t', index_col=0)
        x = pd.read_csv(f'{self.res}/sel_data_{tag}.tsv', sep='\t', index_col=0)
        return s, x.loc[s.index]

    # ------------------------------------------------------ shared basis --
    def _basis_screen(self, x):
        """Flexible function of a feature's own value for the interaction screen
        (within-value dispersion and product-term regression): indicators for
        few-valued x, otherwise a natural cubic spline with df_screen (8) degrees
        of freedom. """
        x = np.asarray(x, float)
        u = np.unique(x[np.isfinite(x)])
        if u.size < 2:
            return np.empty((x.size, 0))
        if u.size <= self.n_bins:
            return np.column_stack([(x == v).astype(float) for v in u[1:]])
        from patsy import dmatrix
        return np.asarray(dmatrix(f'0 + cr(a, df={self.df_screen})', {'a': x},
                                  return_type='dataframe'))

    def _basis(self, x):
        """Indicator columns for few-valued x, restricted cubic spline for
        continuous x. Binning a continuous variable and using bin means would
        leave h(x) varying inside each bin and count that as residual
        dispersion; on simulated additive curvature that inflated the residual
        fraction from 0.006 to 0.29."""
        x = np.asarray(x, float)
        u = np.unique(x[np.isfinite(x)])
        if u.size < 2:
            return np.empty((x.size, 0))
        if u.size <= self.n_bins:
            return np.column_stack([(x == v).astype(float) for v in u[1:]])
        from patsy import dmatrix
        return np.asarray(dmatrix(f'0 + cr(a, df={self.df_spline})', {'a': x},
                                  return_type='dataframe'))

    def _fit_r2(self, A, phi, tss):
        beta = np.linalg.lstsq(A, phi, rcond=None)[0]
        return 1 - float(((phi - A @ beta) ** 2).sum()) / tss

    # ---------------------------------------------------- rule 1: exclude --
    def exclusion(self, shap_df, frac=None, out=None):
        """Exclude when the upper bound of the 95% CI of the mean absolute
        attribution falls below frac x SD(per-feature mean|FA|) — the SD across
        features, as the Methods describe, not of the whole matrix."""
        frac = self.frac if frac is None else frac
        ma = shap_df.abs().mean(axis=0)
        thr = frac * float(ma.std(ddof=1))
        rows = []
        for c in shap_df.columns:
            v = shap_df[c].abs().to_numpy(float)
            b = np.array([np.nanmean(v[self.rng.integers(0, v.size, v.size)])
                          for _ in range(self.n_boot)])
            hi = float(np.percentile(b, 97.5))
            rows.append(dict(feature=c, mean_abs=float(ma[c]), ci_high=hi,
                             threshold=thr, excluded=bool(hi < thr)))
        t = pd.DataFrame(rows).sort_values('mean_abs')
        if out:
            t.to_csv(out, sep='\t', index=False)
        excl = t.loc[t.excluded, 'feature'].tolist()
        print(f'[exclusion] threshold={thr:.4g}; {len(excl)}/{len(t)} excluded')
        return excl, t

    def exclusion_sensitivity(self, shap_df, fracs=(0.01, 0.05, 0.10, 0.20), out=None):
        rows = [dict(frac=f, n_excluded=len(e), features='; '.join(sorted(e))[:1000])
                for f, e in ((f, self.exclusion(shap_df, frac=f)[0]) for f in fracs)]
        t = pd.DataFrame(rows)
        if out:
            t.to_csv(out, sep='\t', index=False)
        return t

    # -------------------------------------------------- rule 2: nonlinear --
    def nonlinear(self, shap_df, sel_df, candidates, out=None):
        """Incremental R-squared of the flexible basis over a straight line.
        Ordinal features are eligible and use the indicator basis, which is the
        standard test of a linear-trend assumption: BMI in particular is a
        textbook case, since both low and high categories carry risk."""
        rows = []
        for c in candidates:
            if c not in shap_df.columns or c not in sel_df.columns:
                continue
            x, phi = sel_df[c].to_numpy(float), shap_df[c].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(phi)
            x, phi = x[ok], phi[ok]
            tss = float(((phi - phi.mean()) ** 2).sum())
            B = self._basis(x)
            if x.size < 50 or B.shape[1] < 2 or tss == 0:
                continue
            r2_lin = self._fit_r2(np.c_[np.ones(x.size), x], phi, tss)
            r2_flex = self._fit_r2(np.c_[np.ones(x.size), B], phi, tss)
            k = B.shape[1]
            den = (1 - r2_flex) / max(x.size - k - 1, 1)
            F = ((r2_flex - r2_lin) / max(k - 1, 1) / den) if den > 0 else np.inf
            rows.append(dict(feature=c, n=int(x.size), n_levels=int(np.unique(x).size),
                             basis='indicator' if np.unique(x).size <= self.n_bins else 'spline',
                             r=float(np.corrcoef(x, phi)[0, 1]),
                             delta_r2=float(r2_flex - r2_lin),
                             p=float(fdist.sf(F, max(k - 1, 1), max(x.size - k - 1, 1)))))
        t = pd.DataFrame(rows)
        if not len(t):
            return [], t
        t['p_fdr'] = multipletests(t['p'], method='fdr_bh')[1]
        t['selected'] = (t.p_fdr < self.alpha) & (t.delta_r2 > self.min_delta_r2)
        if out:
            t.to_csv(out, sep='\t', index=False)
        sel = t.loc[t.selected, 'feature'].tolist()
        print(f'[non-linearity] {len(sel)}/{len(t)} flagged: {sel}')
        return sel, t

    # ------------------------------------------------ rule 3: interaction --
    def residual_fraction(self, x, phi):
        """Share of Var(phi_x) not explained by x itself. ~0 -> Pattern B."""
        x, phi = np.asarray(x, float), np.asarray(phi, float)
        tss = float(phi.var())
        if tss == 0:
            return 0.0
        if np.unique(x[np.isfinite(x)]).size < 2:
            return 1.0                                   # Pattern A: x constant
        A = np.c_[np.ones(x.size), self._basis_screen(x)]
        return float(np.var(phi - A @ np.linalg.lstsq(A, phi, rcond=None)[0]) / tss)

    def within_value_dispersion(self, x, phi, scale):
        """How much a feature's attribution varies among patients who share its
        value, in absolute terms: the SD of phi_x around a flexible function of
        x (indicators for few-valued x, a spline otherwise), divided by the
        mean absolute attribution over all features in the subcohort, the same
        scale as the interaction effect sizes.
        """
        x, phi = np.asarray(x, float), np.asarray(phi, float)
        if np.unique(x[np.isfinite(x)]).size < 2 or scale <= 0:
            return np.nan
        A = np.c_[np.ones(x.size), self._basis_screen(x)]
        resid = phi - A @ np.linalg.lstsq(A, phi, rcond=None)[0]
        return float(resid.std() / scale)

    def _dispersion_ci(self, x, phi, scale, b=200):
        """Bootstrap interval for within_value_dispersion, with the basis built
        once on the full sample and rows resampled, so the knots stay fixed."""
        x, phi = np.asarray(x, float), np.asarray(phi, float)
        if np.unique(x[np.isfinite(x)]).size < 2 or scale <= 0:
            return np.nan, np.nan
        A = np.c_[np.ones(x.size), self._basis_screen(x)]
        v = np.empty(b)
        for k in range(b):
            i = self.rng.integers(0, x.size, x.size)
            Ai, pi = A[i], phi[i]
            v[k] = float((pi - Ai @ np.linalg.lstsq(Ai, pi, rcond=None)[0]).std() / scale)
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    def _cutpoint(self, x, var, cutpoints):
        """Continuous variables need a pre-specified cut point: a threshold read
        off the attribution plot is not defensible, since the zero crossing of
        phi_x sits at the cohort mean of h(x) and moved from 58.6 to 72.0 years
        for one risk function under three age distributions."""
        u = np.unique(np.asarray(x, float)[np.isfinite(x)])
        if cutpoints and var in cutpoints:
            return float(cutpoints[var]), 'pre-specified'
        if u.size == 2:
            return float(u[0]), 'binary'
        if u.size <= self.n_bins:
            return float(np.median(u)), 'median of observed levels'
        return None, 'continuous: no pre-specified cut point, skipped'

    @staticmethod
    def _slopes(Y, PHI):
        """Column-wise OLS slope of each attribution on its own feature, with an
        HC3 heteroskedasticity-robust variance. 
        """
        n = Y.shape[0]
        Yc, Pc = Y - Y.mean(0), PHI - PHI.mean(0)
        den = (Yc ** 2).sum(0)
        safe = np.where(den > 0, den, 1.0)
        beta = np.where(den > 0, (Yc * Pc).sum(0) / safe, np.nan)
        resid = Pc - Yc * beta
        h = np.clip(1.0 / n + Yc ** 2 / safe, 0, 0.999)
        var = np.where(den > 0, ((Yc ** 2) * (resid ** 2) / (1 - h) ** 2).sum(0) / safe ** 2,
                       np.nan)
        return beta, var

    @staticmethod
    def _sandwich_wald(estimate, se, scale):
        """Two-sided Wald P and 95% interval from an HC3 heteroskedasticity-robust
        standard error.
        """
        estimate, se = np.asarray(estimate, float), np.asarray(se, float)
        valid = np.isfinite(estimate) & np.isfinite(se) & (se > 1e-8 * max(scale, 1e-300))
        p = np.ones(estimate.shape)
        p[valid] = 2 * norm.sf(np.abs(estimate[valid]) / se[valid])
        lo, hi = estimate - 1.96 * se, estimate + 1.96 * se
        lo[~valid] = hi[~valid] = np.nan
        return lo, hi, np.maximum(p, np.finfo(float).tiny)

    @staticmethod
    def _bootstrap_wald(estimate, boots):
        estimate, boots = np.asarray(estimate, float), np.asarray(boots, float)
        if len(boots) < 30:
            raise ValueError('n_boot_screen must be >= 30; use >= 300 for analysis')
        finite = np.isfinite(boots)
        count = finite.sum(axis=0)
        se = np.full(estimate.shape, np.nan)
        enough = count >= 2
        clean = np.where(finite, boots, np.nan)
        se[enough] = np.nanstd(clean[:, enough], axis=0, ddof=1)
        valid = (np.isfinite(estimate) & np.isfinite(se) & (count >= .9 * len(boots))
                 & (se > 1e-10 * np.maximum(np.abs(estimate), np.finfo(float).tiny)))
        p = np.ones(estimate.shape)
        p[valid] = np.exp(np.maximum(np.log(2.) + norm.logsf(np.abs(estimate[valid]) / se[valid]),
                                    np.log(np.finfo(float).tiny)))
        lo, hi = estimate - 1.96 * se, estimate + 1.96 * se
        lo[~valid] = hi[~valid] = np.nan
        return lo, hi, p

    def _binary_contrast_screen(self, x, cut, shap_df, X_df, strat_var, cols):
        """Binary partners: split on the stratifying feature's observed value and
        compare the within-stratum contrast E[phi|y=1] - E[phi|y=0].
        """
        x = np.asarray(x, float)
        m1, m2 = x <= cut, x > cut
        if m1.sum() < self.min_cell or m2.sum() < self.min_cell or not cols:
            return pd.DataFrame()
        Y1 = X_df.loc[m1, cols].to_numpy(float)
        P1 = shap_df.loc[m1, cols].to_numpy(float)
        Y2 = X_df.loc[m2, cols].to_numpy(float)
        P2 = shap_df.loc[m2, cols].to_numpy(float)
        d1, v1 = self._slopes(Y1, P1)
        d2, v2 = self._slopes(Y2, P2)
        diff = d1 - d2
        att_scale = float(np.nanmean(np.abs(np.r_[P1, P2])))

        if self.se_method == 'sandwich':
            # the two strata are independent samples, so the variance of the
            # difference is the sum of the two HC3 variances
            lo, hi, p = self._sandwich_wald(diff, np.sqrt(v1 + v2), att_scale)
        else:
            boots = np.empty((self.n_boot_screen, len(cols)))
            for b in range(self.n_boot_screen):
                i1 = self.rng.integers(0, Y1.shape[0], Y1.shape[0])
                i2 = self.rng.integers(0, Y2.shape[0], Y2.shape[0])
                boots[b] = self._slopes(Y1[i1], P1[i1])[0] - self._slopes(Y2[i2], P2[i2])[0]
            lo, hi, p = self._bootstrap_wald(diff, boots)

        # One denominator for every partner: the mean absolute attribution over all
        # candidates in this subcohort. Using each partner's own mean|phi_y| would
        # favour rare flags, because for phi_y = b(y - p) that mean is 2bp(1 - p):
        # a partner at 2% prevalence is scaled up about tenfold relative to one at
        # 30% for the same absolute difference in contrast, and rare flags carry
        # the noisiest contrasts.
        phi_scale = np.full(len(cols), float(np.nanmean(np.abs(shap_df.to_numpy(float)))))
        ok = np.isfinite(diff) & np.isfinite(p)
        return pd.DataFrame(dict(strat=strat_var, partner=np.asarray(cols)[ok],
                                 method="within-stratum contrast",
                                 cut_point=cut, n_group1=int(m1.sum()),
                                 n_group2=int(m2.sum()),
                                 contrast_group1=d1[ok], contrast_group2=d2[ok],
                                 effect_size=diff[ok], ci_low=lo[ok], ci_high=hi[ok],
                                 abs_effect=np.abs(diff[ok]), phi_scale=phi_scale[ok],
                                 effect_vs_scale=np.abs(diff[ok]) / np.maximum(phi_scale[ok], 1e-12),
                                 p_screen=np.minimum(p[ok], 1.0)))

    def _product_term_screen(self, x, phi_x, X_df, strat_var, cols):
        """Screen continuous and ordinal partners using product terms.
        
        Regress the stratifying feature's attribution on a flexible basis in x,
        the partners, and products of each partner with centered x. Product
        coefficients measure modification of the attribution slope.
        
        Standardize x and partners before fitting so effect sizes share a common
        scale across partners."""
        x = np.asarray(x, float)
        phi_x = np.asarray(phi_x, float)
        report_cols = list(cols)
        binary_adjust = [c for c in X_df if c != strat_var
                         and pd.api.types.is_numeric_dtype(X_df[c])
                         and X_df[c].nunique(dropna=True) == 2]
        cols = [c for c in dict.fromkeys(report_cols + binary_adjust)
                if X_df[c].nunique(dropna=True) > 1]
        if not cols or np.unique(x[np.isfinite(x)]).size < 2:
            return pd.DataFrame()

        def z(v):
            v = np.asarray(v, float)
            sd = np.nanstd(v)
            return (v - np.nanmean(v)) / (sd if sd > 0 else 1.0)

        C = np.column_stack([z(X_df[c].to_numpy(float)) for c in cols])
        xz = z(x).reshape(-1, 1)
        A = np.c_[np.ones(x.size), self._basis_screen(x), C, C * xz]
        k = len(cols)
        coef = np.linalg.lstsq(A, phi_x, rcond=None)[0]
        beta = coef[-k:]

        if self.se_method == 'sandwich':
            # HC3 for the k product coefficients only. pinv rather than inv:
            # with ~50 binary adjusters and their products the cross-product
            # matrix can be close to singular.
            resid = phi_x - A @ coef
            Pinv = np.linalg.pinv(A.T @ A)
            AP = A @ Pinv
            h = np.clip((AP * A).sum(axis=1), 0, 0.999)          # leverages
            M = AP[:, -k:]
            se = np.sqrt((((resid / (1 - h)) ** 2)[:, None] * M ** 2).sum(axis=0))
            lo, hi, p = self._sandwich_wald(beta, se, float(np.nanmean(np.abs(phi_x))))
        else:
            boots = np.empty((self.n_boot_screen, k))
            for b in range(self.n_boot_screen):
                j = self.rng.integers(0, x.size, x.size)
                boots[b] = np.linalg.lstsq(A[j], phi_x[j], rcond=None)[0][-k:]
            lo, hi, p = self._bootstrap_wald(beta, boots)

        # Same denominator as the binary branch. Inputs are standardised, so beta
        # is the change in phi_x's slope on x per 1-SD change in the partner: a
        # fixed span, which puts it on roughly (not exactly) the scale of the
        # binary branch's 0-to-1 contrast. The two branches are reported
        # separately so that the threshold's effect on each can be checked.
        phi_scale = float(self._phi_scale) if getattr(self, "_phi_scale", None) \
            else float(np.nanmean(np.abs(phi_x)))
        ok = np.isfinite(beta) & np.isfinite(p) & np.isin(cols, report_cols)
        return pd.DataFrame(dict(strat=strat_var, partner=np.asarray(cols)[ok],
                                 method="product-term regression",
                                 cut_point=np.nan, n_group1=int(x.size), n_group2=0,
                                 contrast_group1=np.nan, contrast_group2=np.nan,
                                 effect_size=beta[ok], ci_low=lo[ok], ci_high=hi[ok],
                                 abs_effect=np.abs(beta[ok]), phi_scale=phi_scale,
                                 effect_vs_scale=np.abs(beta[ok]) / max(phi_scale, 1e-12),
                                 p_screen=np.minimum(p[ok], 1.0)))

    # ---- nominal features -------------------------------------------------------
    def _feature_frame(self, X_df):
        """Feature values for screening. Each one-hot group is recombined into one
        categorical code named after its parent feature (0 for the reference level,
        where every dummy is zero; j for the j-th dummy); other columns are kept."""
        F, nominal = X_df.copy(), set()
        for parent, cols in self.groups.items():
            cols = [c for c in cols if c in F.columns]
            if not cols:
                continue
            D = F[cols].to_numpy(float)
            code = np.where(D.max(axis=1) > 0, D.argmax(axis=1) + 1, 0).astype(float)
            F = F.drop(columns=cols)
            F[parent] = code
            nominal.add(parent)
        return F, nominal

    def _to_features(self, cols):
        """Map one-hot dummy names to their parent feature, keeping order."""
        parent = {d: p for p, ds in self.groups.items() for d in ds}
        return list(dict.fromkeys(parent.get(c, c) for c in cols))

    @staticmethod
    def _ols_hc3(A, y):
        """OLS fit with what is needed for HC3 covariances of any block of
        coefficients: coefficients, A (A'A)^+ and the HC3 residual weights."""
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        resid = y - A @ coef
        AP = A @ np.linalg.pinv(A.T @ A)
        h = np.clip((AP * A).sum(axis=1), 0, 0.999)
        return coef, AP, (resid / (1 - h)) ** 2

    @staticmethod
    def _hc3_block(AP, w, idx):
        M = AP[:, idx]
        return (M * w[:, None]).T @ M

    @staticmethod
    def _wald_block(b, V, scale):
        """Joint Wald test that the coefficients b are zero, with covariance V.
        With one coefficient it equals the two-sided z test. A covariance that is
        numerically zero (attributions with no estimation error) gives P = 1."""
        b, V = np.atleast_1d(np.asarray(b, float)), np.atleast_2d(np.asarray(V, float))
        if (not np.all(np.isfinite(b)) or not np.all(np.isfinite(V))
                or np.max(np.diag(V)) <= (1e-8 * max(scale, 1e-300)) ** 2):
            return np.nan, 1.0
        stat = float(b @ np.linalg.pinv(V) @ b)
        return stat, float(chi2.sf(stat, b.size))

    def _levels(self, v, masks):
        """Reference level and the other levels with at least min_level patients
        in every mask (stratum); binary features have reference 0."""
        lev = np.unique(v[np.isfinite(v)])
        ref = 0.0 if 0.0 in lev else lev[0]
        enough = lambda l: all(((v == l) & m).sum() >= self.min_level for m in masks)
        if not enough(ref):
            return ref, []
        return ref, [l for l in lev if l != ref and enough(l)]

    def _stratified_contrast_screen(self, labels, shap_df, F, strat_var, cols, cut=np.nan):
        """
        Within-stratum contrasts of binary and nominal partners, compared across
        the strata of the focal feature: two strata at a cut point, or one per
        level of a nominal focal feature.
        """
        labels = np.asarray(labels, float)
        strata = [s for s in np.unique(labels[np.isfinite(labels)])
                  if (labels == s).sum() >= self.min_cell]
        if len(strata) < 2 or not cols:
            return pd.DataFrame()
        masks = [labels == s for s in strata]
        S = len(strata)
        scale = float(self._phi_scale) if getattr(self, "_phi_scale", None) \
            else float(np.nanmean(np.abs(shap_df.to_numpy(float))))
        rows = []
        for c in cols:
            v, phi = F[c].to_numpy(float), shap_df[c].to_numpy(float)
            ref, L = self._levels(v, masks)
            if not L:
                continue
            m = len(L)
            betas, Vs = [], []
            for mk in masks:
                A = np.column_stack([np.ones(mk.sum())] + [(v[mk] == l).astype(float) for l in L])
                coef, AP, w = self._ols_hc3(A, phi[mk])
                betas.append(coef[1:])
                Vs.append(self._hc3_block(AP, w, list(range(1, m + 1))))
            d = np.concatenate([betas[0] - betas[s] for s in range(1, S)])
            Sig = np.zeros((d.size, d.size))
            for a in range(S - 1):
                for b in range(S - 1):
                    Sig[a * m:(a + 1) * m, b * m:(b + 1) * m] = Vs[0] + (Vs[a + 1] if a == b else 0)
            stat, p = self._wald_block(d, Sig, scale)
            j = int(np.nanargmax(np.abs(d))) if np.any(np.isfinite(d)) else 0
            eff = float(d[j])
            se = float(np.sqrt(Sig[0, 0])) if d.size == 1 else np.nan
            rows.append(dict(strat=strat_var, partner=c, method="within-stratum contrast",
                             cut_point=cut, n_group1=int(masks[0].sum()),
                             n_group2=int(sum(mk.sum() for mk in masks[1:])),
                             contrast_group1=float(betas[0][0]) if m == 1 else np.nan,
                             contrast_group2=float(betas[1][0]) if (m == 1 and S == 2) else np.nan,
                             df=int(d.size), wald=stat, effect_size=eff,
                             ci_low=eff - 1.96 * se if d.size == 1 else np.nan,
                             ci_high=eff + 1.96 * se if d.size == 1 else np.nan,
                             abs_effect=abs(eff), phi_scale=scale,
                             effect_vs_scale=abs(eff) / max(scale, 1e-12),
                             p_screen=min(p, 1.0)))
        return pd.DataFrame(rows)

    def _product_block_screen(self, x, phi_x, F, strat_var, cols, nominal):
        """
        Product-term regression generalised to nominal features: the focal
        feature enters the products as its standardized value, or as its
        standardized level indicators when nominal; a nominal partner enters as
        its standardized level indicators. For each partner, all of its product
        coefficients are tested jointly (Wald statistic, HC3 covariance), so a
        nominal feature with k levels contributes k - 1 degrees of freedom per
        level indicator of the other feature. Binary features enter, with their
        products, for adjustment only. Without nominal features this is the
        product-term regression.
        """
        x, phi_x = np.asarray(x, float), np.asarray(phi_x, float)
        report = list(cols)
        adjust = [c for c in F if c != strat_var and c not in nominal
                  and pd.api.types.is_numeric_dtype(F[c]) and F[c].nunique(dropna=True) == 2]
        use = [c for c in dict.fromkeys(report + adjust) if F[c].nunique(dropna=True) > 1]
        if not use or np.unique(x[np.isfinite(x)]).size < 2:
            return pd.DataFrame()

        def z(v):
            v = np.asarray(v, float)
            sd = np.nanstd(v)
            return (v - np.nanmean(v)) / (sd if sd > 0 else 1.0)

        allrows = [np.ones(x.size, bool)]
        if strat_var in nominal:
            _, fl = self._levels(x, allrows)
            if not fl:
                return pd.DataFrame()
            FX = np.column_stack([z(x == l) for l in fl])
        else:
            FX = z(x)[:, None]
        blocks, C = {}, []
        for c in use:
            v = F[c].to_numpy(float)
            if c in nominal:
                _, L = self._levels(v, allrows)
                cols_c = [z(v == l) for l in L]
            else:
                cols_c = [z(v)]
            if cols_c:
                blocks[c] = (len(C), len(cols_c))
                C += cols_c
        C = np.column_stack(C)
        kx = FX.shape[1]
        prods = np.column_stack([C[:, i] * FX[:, j] for i in range(C.shape[1]) for j in range(kx)])
        A = np.c_[np.ones(x.size), self._basis_screen(x), C, prods]
        off = A.shape[1] - prods.shape[1]
        coef, AP, w = self._ols_hc3(A, phi_x)
        scale = float(self._phi_scale) if getattr(self, "_phi_scale", None) \
            else float(np.nanmean(np.abs(phi_x)))
        rows = []
        for c in report:
            if c not in blocks:
                continue
            s0, width = blocks[c]
            idx = [off + i * kx + j for i in range(s0, s0 + width) for j in range(kx)]
            b = coef[idx]
            V = self._hc3_block(AP, w, idx)
            stat, p = self._wald_block(b, V, scale)
            jj = int(np.nanargmax(np.abs(b))) if np.any(np.isfinite(b)) else 0
            eff = float(b[jj])
            se = float(np.sqrt(V[0, 0])) if b.size == 1 else np.nan
            rows.append(dict(strat=strat_var, partner=c, method="product-term regression",
                             cut_point=np.nan, n_group1=int(x.size), n_group2=0,
                             contrast_group1=np.nan, contrast_group2=np.nan,
                             df=int(b.size), wald=stat, effect_size=eff,
                             ci_low=eff - 1.96 * se if b.size == 1 else np.nan,
                             ci_high=eff + 1.96 * se if b.size == 1 else np.nan,
                             abs_effect=abs(eff), phi_scale=scale,
                             effect_vs_scale=abs(eff) / max(scale, 1e-12),
                             p_screen=min(p, 1.0)))
        return pd.DataFrame(rows)

    def stratified_screen(self, x, cut, shap_df, X_df, strat_var, cols, nominal=frozenset()):
        """Route each partner to the statistic that is valid for its type and
        stack the results. Binary and nominal partners go to the within-stratum
        contrast test when the focal feature defines strata (a cut point, or its
        levels when nominal); continuous and ordinal partners, and nominal ones
        when the focal feature has no strata, go to the product-term regression."""
        binary, nom, cont = [], [], []
        for c in cols:
            if c in nominal:
                nom.append(c)
            elif X_df[c].dropna().nunique() == 2:
                binary.append(c)
            else:
                cont.append(c)
        if strat_var in nominal:
            labels, cutv = np.asarray(x, float), np.nan
        elif cut is not None:
            labels, cutv = (np.asarray(x, float) > cut).astype(float), cut
        else:
            labels, cutv = None, np.nan
        parts = []
        if labels is not None and (binary or nom):
            parts.append(self._stratified_contrast_screen(labels, shap_df, X_df, strat_var,
                                                          binary + nom, cutv))
        prod = cont + (nom if labels is None else [])
        if prod and strat_var in shap_df.columns:
            parts.append(self._product_block_screen(x, shap_df[strat_var].to_numpy(float),
                                                    X_df, strat_var, prod, nominal))
        parts = [p for p in parts if len(p)]
        if not parts:
            return pd.DataFrame()
        print(f"  [{strat_var}] {len(binary)} binary, {len(nom)} nominal (contrast"
              f"{'' if labels is not None else ' n/a'}), {len(cont)} continuous/ordinal partners")
        return pd.concat(parts, ignore_index=True)

    def interaction(self, shap_df, X_df, cutpoints=None, candidates=None,
                    subcohort="", out=None, diag_out=None):
        """Screen one subcohort. FDR is not applied here: the low- and
        high-risk subcohorts are screened separately and their candidates
        merged afterwards, so multiplicity has to be controlled once over the
        pooled set of tests rather than twice over halves of it."""
        X_df, nominal = self._feature_frame(X_df.loc[shap_df.index])
        cand = [c for c in self._to_features(candidates or X_df.columns) if c in X_df.columns]
        self._phi_scale = float(np.nanmean(np.abs(shap_df.to_numpy(float))))
        diag, tabs = [], []
        scale = self._phi_scale
        for v in [c for c in shap_df.columns if c in X_df.columns]:
            x, phi = X_df[v].to_numpy(float), shap_df[v].to_numpy(float)
            constant = np.unique(x[np.isfinite(x)]).size < 2
            disp = self.within_value_dispersion(x, phi, scale)
            lo, hi = self._dispersion_ci(x, phi, scale)
            cut, basis = self._cutpoint(x, v, cutpoints)
            # constant: no variation in this subcohort, so nothing can be split;
            # B: attribution essentially a function of the feature's own value;
            # A/C: patients with the same value receive clearly different
            # attributions, the case for an interaction screen
            gate = (not constant) and lo >= self.dispersion_thresh
            pattern = 'constant' if constant else ('A/C' if gate else 'B')
            row = dict(subcohort=subcohort, variable=v, pattern=pattern,
                       dispersion=disp, disp_ci_low=lo, disp_ci_high=hi,
                       dispersion_threshold=self.dispersion_thresh,
                       residual_fraction=self.residual_fraction(x, phi),
                       cut_point=cut, cut_basis=basis, screened=False)
            if gate:
                cols = [c for c in cand if c != v and c in shap_df.columns
                        and X_df[c].nunique(dropna=True) > 1]
                t = self.stratified_screen(x, cut, shap_df, X_df, v, cols, nominal)
                if len(t):
                    t.insert(0, "subcohort", subcohort)
                    tabs.append(t)
                    row['screened'] = True
            diag.append(row)

        diag_df = pd.DataFrame(diag).sort_values('dispersion', ascending=False)
        if diag_out:
            diag_df.to_csv(diag_out, sep='\t', index=False)
        n = diag_df.pattern.value_counts()
        print(f"[interaction/{subcohort}] {len(diag_df)} variables: {n.get('constant', 0)} constant, "
              f"{n.get('B', 0)} Pattern B (no clear within-value dispersion), "
              f"{n.get('A/C', 0)} A/C; {int(diag_df.screened.sum())} screened")
        tests = pd.concat(tabs, ignore_index=True) if tabs else pd.DataFrame()
        if out and len(tests):
            tests.to_csv(out, sep='\t', index=False)
        return tests, diag_df

    def merge_interaction_candidates(self, tests, out=None):
        """Pool the tests from both subcohorts, control multiplicity once, and
        take the union of the surviving pairs.
        """
        if not len(tests):
            return {}, pd.DataFrame()
        t = tests.copy()
        t['n_tests_total'] = len(t)
        t['pair'] = ['||'.join(sorted([a, b])) for a, b in zip(t.strat, t.partner)]
        # A pair can be tested up to four times (either feature as the stratifier,
        # in either subcohort), and those tests are strongly dependent. BH over the
        # raw tests would therefore not control the FDR over unique pairs. Combine
        # within each pair by Bonferroni, which is valid under any dependence among
        # its replicates, then apply BH across unique pairs.
        g = t.groupby('pair')['p_screen']
        pair_p = (g.min() * g.size()).clip(upper=1.0)
        pair_fdr = pd.Series(multipletests(pair_p, method='fdr_bh')[1], index=pair_p.index)
        t['n_tests_in_pair'] = t['pair'].map(g.size())
        t['p_pair'] = t['pair'].map(pair_p)
        t['p_fdr'] = t['pair'].map(pair_fdr)
        t['selected'] = (t.p_fdr < self.alpha) & (t.effect_vs_scale >= self.min_abs_effect)
        if out:
            t.to_csv(out, sep='\t', index=False)

        sel = t[t.selected].sort_values('effect_vs_scale', ascending=False)
        uniq = sel.drop_duplicates('pair')
        spec = {}
        for _, r in uniq.iterrows():
            spec.setdefault(r.strat, []).append(r.partner)

        by_sub = sel.groupby('subcohort')['pair'].nunique().to_dict()
        by_meth = sel.drop_duplicates('pair').groupby('method').size().to_dict()
        both = (sel.groupby('pair')['subcohort'].nunique() == 2).sum()
        print(f"[interaction] {len(t)} tests over {t.pair.nunique()} unique pairs; "
              f"{len(uniq)} pairs selected (unique-pair BH) "
              f"by subcohort {by_sub}, by method {by_meth}, in both subcohorts {both}")
        return spec, t

    def target_model_filter(self, tests, X, y, continuous, groups=None, df=4, cache=None):
        """Check each screened pair in the target Cox model format before it can enter.
        """
        from patsy import dmatrix
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from sksurv.util import Surv
        groups = groups or self.groups or {}
        if not len(tests) or not tests['selected'].any():
            return tests, pd.DataFrame()

        # Reuse cached scores across margins on the same training data
        # and null design. Raw scores are cached; BH is recomputed for each
        # margin's current candidate family. Do not mutate X/y during the search.
        cache = {} if cache is None else cache
        signature = (id(X), id(y), tuple(X.columns), tuple(continuous), df,
                     tuple((g, tuple(c)) for g, c in groups.items()))
        if cache and cache.get('signature') != signature:
            raise ValueError('Cox score cache belongs to a different training design')
        cache['signature'] = signature
        if 'tester' not in cache:
            cols = {}
            for c in X.columns:
                v = X[c].to_numpy(float)
                u = np.unique(v[np.isfinite(v)])
                if c in continuous and u.size > self.n_bins:
                    B = np.asarray(dmatrix(f"0 + cr(a, df={df})", {"a": v}, return_type="dataframe"))
                    for j in range(B.shape[1] - 1):
                        cols[f"{c}__s{j}"] = B[:, j]
                elif 2 < u.size <= self.n_bins:
                    for lv in u[1:]:
                        cols[f"{c}__l{lv:g}"] = (v == lv).astype(float)
                else:
                    cols[c] = v
            M = pd.DataFrame(cols, index=X.index)
            M = M.loc[:, M.std() > 0]
            yy = Surv.from_arrays(event=np.asarray(y["event"], bool), time=np.asarray(y["time"], float))
            base = CoxPHSurvivalAnalysis(alpha=1e-6, ties="breslow").fit(M, yy)
            tester = CoxScoreTester(M.to_numpy(), base.predict(M), yy["time"], yy["event"])
            cache.update(tester=tester, scores={})
        tester = cache['tester']

        rows = []
        for pair, r in (tests[tests['selected']].sort_values('effect_vs_scale', ascending=False)
                        .drop_duplicates('pair').set_index('pair').iterrows()):
            if pair not in cache['scores']:
                ca, cb = self._cols(r['strat'], X, groups), self._cols(r['partner'], X, groups)
                Z = [((X[u] - X[u].mean()) * (X[v] - X[v].mean())).to_numpy(float)
                     for u in ca for v in cb if u != v]
                if not Z:
                    continue
                stat, p = tester.test(np.column_stack(Z))
                cache['scores'][pair] = (stat, p, len(Z))
            stat, p, degrees = cache['scores'][pair]
            rows.append(dict(pair=pair, strat=r['strat'], partner=r['partner'],
                             df=degrees, score_stat=stat, target_p=p))
        log = pd.DataFrame(rows)
        if not len(log):
            return tests, log
        log['target_p_fdr'] = multipletests(log['target_p'], method='fdr_bh')[1]
        log['target_confirmed'] = log['target_p_fdr'] < self.alpha
        t = tests.copy()
        t['screen_selected'] = t['selected']
        t = t.merge(log[['pair', 'target_p', 'target_p_fdr', 'target_confirmed']],
                    on='pair', how='left')
        t['target_confirmed'] = t['target_confirmed'].fillna(False).astype(bool)
        t['selected'] = t['screen_selected'] & t['target_confirmed']
        print(f"[target model] {len(log)} screened pairs tested with spline main effects; "
              f"{int(log['target_confirmed'].sum())} remain after BH")
        return t, log

    @staticmethod
    def riley_parameter_budget(X, y, shrinkage=0.9, penalizer=1e-6,
                               sensitivity=(0.85, 0.90, 0.95)):
        """
        Largest number of predictor parameters for which the expected global
        shrinkage factor stays at or above `shrinkage`, given this sample size:
        criterion (i) of Riley et al. (Stat Med 2019;38:1276-96), inverted from
            n = p / ((S - 1) ln(1 - R2_CS / S))
        to  p_max = n (S - 1) ln(1 - R2_CS / S).

        R2_CS is the Cox-Snell R2 of the model fitted to X, from its likelihood
        ratio statistic against the null model: 1 - exp(-LR / n). X should be the
        design before any interaction term is added. Its R2 is at most that of the
        augmented model, so the budget errs on the side of fewer terms.

        The apparent R2 is optimistic, since it is measured on the data the model
        was fitted to. It is scaled by the heuristic shrinkage factor 1 - p0 / LR
        (van Houwelingen and le Cessie), where p0 is the number of columns of X,
        before being used. Both values are returned.

        S is the one constant left. It is reported at a grid of values so the
        dependence of the budget on it is visible; the main analysis uses 0.9.

        Riley's criterion is defined on candidate parameters. When terms are
        selected from a larger candidate set, the selected model is more optimistic
        than this bound implies; the held-out evaluation, not the criterion, guards
        against that.
        """
        from lifelines import CoxPHFitter
        d = X.copy()
        d["time"] = np.asarray(y["time"], float)
        d["event"] = np.asarray(y["event"], int)
        f = CoxPHFitter(penalizer=penalizer).fit(d, "time", "event")
        lr = float(f.log_likelihood_ratio_test().test_statistic)
        n, p0 = len(d), X.shape[1]
        r2_app = 1.0 - np.exp(-lr / n)
        s_heur = max(0.0, 1.0 - p0 / lr) if lr > 0 else 0.0
        r2cs = s_heur * r2_app

        def pmax(S):
            return int(np.floor(n * (S - 1.0) * np.log(1.0 - r2cs / S))) if r2cs < S else 0

        grid = {f"p_max_S{S:.2f}": pmax(S) for S in sensitivity}
        print(f"[riley] n={n}, events={int(d['event'].sum())}, LR={lr:.1f}, "
              f"apparent R2_CS={r2_app:.4f}, heuristic shrinkage={s_heur:.3f}, "
              f"adjusted R2_CS={r2cs:.4f}; p_max by S: "
              + ", ".join(f"{S:.2f}->{pmax(S)}" for S in sensitivity))
        return dict(n=n, events=int(d["event"].sum()), lr=lr, p_base=p0,
                    r2_cs_apparent=r2_app, heuristic_shrinkage=s_heur, r2_cs=r2cs,
                    shrinkage=shrinkage, p_max=pmax(shrinkage), **grid)

    @staticmethod
    def spec_within_budget(tests, base_frame, budget, groups=None):
        """Add selected pairs in order of effect size until the columns they
        expand into would exceed the parameter budget.

        Each pair is costed by the columns it actually adds to base_frame, the
        design after exclusion and non-linear terms have been applied: one
        column for most pairs of binary flags, levels-minus-one for a one-hot or
        indicator-expanded feature, the product of the two for a pair of
        expanded features. A pair whose main effect was excluded is skipped and
        listed in the log with that reason.

        The budget must be fixed without reference to the test set.
        """
        groups = groups or {}
        if not len(tests) or budget <= 0:
            return {}, pd.DataFrame()
        ranked = (tests[tests['selected']].sort_values('effect_vs_scale', ascending=False)
                  .drop_duplicates('pair'))
        if not len(ranked):
            print("[budget] no selected pairs to enter")
            return {}, pd.DataFrame()
        spec, rows, used = {}, [], 0
        for _, r in ranked.iterrows():
            ca = Recommender._cols(r['strat'], base_frame, groups)
            cb = Recommender._cols(r['partner'], base_frame, groups)
            cost = sum(1 for u in ca for v in cb if u != v)
            row = dict(pair=r['pair'], strat=r['strat'], partner=r['partner'],
                       effect_vs_scale=r['effect_vs_scale'], cost=cost)
            if cost == 0:
                row.update(entered=False, reason='main effect excluded')
            elif used + cost > budget:
                row.update(entered=False, reason='budget exhausted')
            else:
                spec.setdefault(r['strat'], []).append(r['partner'])
                used += cost
                row.update(entered=True, reason='')
            rows.append(row)
        log = pd.DataFrame(rows)
        n_in = int(log['entered'].sum())
        n_excl = int((log['reason'] == 'main effect excluded').sum())
        print(f"[budget] {budget} terms available; {n_in} pairs entered using {used}; "
              f"{n_excl} skipped because a main effect was excluded, "
              f"{len(log) - n_in - n_excl} for lack of budget")
        return spec, log

    # ------------------------------------------------------------ generate --
    def generate(self, candidates_nonlinear, cutpoints=None, candidates=None,
                 low_tag='FRID_low_train', high_tag='FRID_high_train',
                 out='rebuttal_data/recs'):
        """Screen the two subcohorts separately, then merge.

        The low- and high-risk subcohorts each have their own reference point,
        and a feature's attribution is a contrast against that reference.
        Pooling the two before screening would average those contrasts, which
        is what the two-reference design exists to avoid: a feature whose
        positive contribution near the event-group reference is offset by a
        negative one near the non-event reference would appear negligible in
        the pooled set. Each rule is therefore estimated within a subcohort.

        How the two sets are combined differs by rule:
          exclusion    intersection. A feature is dropped only if it is
                       negligible in both; negligible in one is not grounds for
                       removing it from the model.
          non-linear   union. Curvature visible in one subcohort is still
                       curvature.
          interaction  union, with multiplicity controlled once over the
                       pooled tests (see merge_interaction_candidates).
        """
        os.makedirs(out, exist_ok=True)
        s_lo, x_lo = self.load(low_tag)
        s_hi, x_hi = self.load(high_tag)

        # ---- exclusion: intersection of the two subcohorts ------------------
        excl_lo, t_lo = self.exclusion(s_lo, out=f'{out}/exclusion_tests_low.tsv')
        excl_hi, t_hi = self.exclusion(s_hi, out=f'{out}/exclusion_tests_high.tsv')
        excl = sorted(set(excl_lo) & set(excl_hi))
        only_one = sorted(set(excl_lo) ^ set(excl_hi))
        print(f"[exclusion] low {len(excl_lo)}, high {len(excl_hi)}, "
              f"excluded in both {len(excl)}; negligible in only one, retained: "
              f"{len(only_one)}")
        self.exclusion_sensitivity(s_lo, out=f'{out}/exclusion_sensitivity_low.tsv')
        self.exclusion_sensitivity(s_hi, out=f'{out}/exclusion_sensitivity_high.tsv')

        # ---- non-linearity: union -------------------------------------------
        nl_lo, n_lo = self.nonlinear(s_lo, x_lo, candidates_nonlinear,
                                     out=f'{out}/nonlinear_tests_low.tsv')
        nl_hi, n_hi = self.nonlinear(s_hi, x_hi, candidates_nonlinear,
                                     out=f'{out}/nonlinear_tests_high.tsv')
        nl = sorted(set(nl_lo) | set(nl_hi))
        print(f"[non-linearity] low {nl_lo}, high {nl_hi} -> union {nl}")

        # ---- interaction: screen separately, merge with one FDR -------------
        # candidates exclude features already dropped: screening a feature that
        # will not enter the model spends tests and inflates the multiplicity
        # correction for the pairs that remain
        cand = [c for c in self._to_features(candidates or x_lo.columns) if c not in excl]
        tests_lo, diag_lo = self.interaction(s_lo, x_lo, cutpoints=cutpoints,
                                            candidates=cand, subcohort="low",
                                            diag_out=f'{out}/attribution_patterns_low.tsv')
        tests_hi, diag_hi = self.interaction(s_hi, x_hi, cutpoints=cutpoints,
                                             candidates=cand, subcohort="high",
                                             diag_out=f'{out}/attribution_patterns_high.tsv')
        tests = pd.concat([t for t in (tests_lo, tests_hi) if len(t)], ignore_index=True) \
            if (len(tests_lo) or len(tests_hi)) else pd.DataFrame()
        spec, t_all = self.merge_interaction_candidates(
            tests, out=f'{out}/interaction_tests.tsv')
        pd.concat([diag_lo, diag_hi], ignore_index=True).to_csv(
            f'{out}/attribution_patterns.tsv', sep='\t', index=False)

        rec = dict(exclusion=excl, nonlinear=nl, interaction=spec)
        with open(f'{out}/recommendations.json', 'w') as fh:
            json.dump(rec, fh, indent=2)
        n_pairs = sum(len(v) for v in spec.values())
        print(f'exclusion {len(excl)}, nonlinear {len(nl)}, interaction {n_pairs}')
        return rec

    # --------------------------------------------------------------- apply --
    @staticmethod
    def _cols(name, frame, groups):
        """Map a recommendation name onto the columns that actually exist.

        Three cases: the feature is present as-is; it is a one-hot parent whose
        dummies carry the name as a prefix (SHAP columns carry the parent name,
        the model matrix carries the dummies); or it is an ordinal feature that
        the non-linearity step has just expanded into `name_lvl*` indicators, in
        which case an interaction on it must attach to every level, not be
        silently dropped for a missing main effect.
        """
        if name in frame.columns:
            return [name]
        expanded = [c for c in frame.columns if c.startswith(name + '_lvl')]
        if expanded:
            return expanded
        return [c for c in groups.get(name, []) if c in frame.columns]

    @staticmethod
    def apply(X_train, X_test, rec, variant='all', groups=None,
              hierarchy='drop', log=None, n_bins=10):
        """variant: baseline | exclusion | nonlinear | interaction | all.
        hierarchy='drop' removes an interaction whose main effect was excluded;
        'keep' restores the main effect instead."""
        tr, te = X_train.copy(), X_test.copy()
        groups = groups or {}
        excl = rec['exclusion'] if variant in ('exclusion', 'all') else []
        nl = rec['nonlinear'] if variant in ('nonlinear', 'all') else []
        spec = rec['interaction'] if variant in ('interaction', 'all') else {}

        if hierarchy == 'keep':
            needed = set(spec) | {p for v in spec.values() for p in v}
            excl = [c for c in excl if c not in needed]
        drop = [c for f in excl for c in Recommender._cols(f, tr, groups)]
        tr, te = tr.drop(columns=drop), te.drop(columns=drop)
        n_ind = n_quad = 0

        # New columns are accumulated in dicts and attached once at the end.
        new_tr, new_te, drop_nl = {}, {}, []
        for f in nl:
            if f not in tr.columns:
                continue
            u = np.unique(tr[f].dropna().to_numpy(float))
            if u.size <= n_bins:
                # Expand nonlinear ordinal features into training-level indicators.
                # Unseen test levels use the reference category.
                xtr, xte = tr[f].to_numpy(float), te[f].to_numpy(float)
                for v in u[1:]:
                    nm = f'{f}_lvl{v:g}'
                    new_tr[nm] = (xtr == v).astype(float)
                    new_te[nm] = (xte == v).astype(float)
                drop_nl.append(f)
                n_ind += u.size - 1
            else:
                m = float(tr[f].mean())          # training mean, both frames
                tr[f], te[f] = tr[f] - m, te[f] - m
                new_tr[f + '_quad'] = tr[f].to_numpy(float) ** 2
                new_te[f + '_quad'] = te[f].to_numpy(float) ** 2
                n_quad += 1
        if new_tr:
            tr = pd.concat([tr, pd.DataFrame(new_tr, index=tr.index)], axis=1)
            te = pd.concat([te, pd.DataFrame(new_te, index=te.index)], axis=1)
        if drop_nl:
            tr, te = tr.drop(columns=drop_nl), te.drop(columns=drop_nl)

        made, skipped = set(), []
        int_tr, int_te = {}, {}
        for a, partners in spec.items():
            for b in partners:
                key = tuple(sorted((a, b)))
                if key in made or a == b:
                    continue
                ca, cb = (Recommender._cols(a, tr, groups),
                          Recommender._cols(b, tr, groups))
                if not ca or not cb:
                    skipped.append(dict(strat=a, partner=b, reason='main effect excluded'))
                    continue
                made.add(key)
                for u in ca:
                    for v in cb:
                        if u != v:
                            int_tr[f'{u}__x__{v}'] = tr[u].to_numpy(float) * tr[v].to_numpy(float)
                            int_te[f'{u}__x__{v}'] = te[u].to_numpy(float) * te[v].to_numpy(float)
        if int_tr:
            tr = pd.concat([tr, pd.DataFrame(int_tr, index=tr.index)], axis=1)
            te = pd.concat([te, pd.DataFrame(int_te, index=te.index)], axis=1)
        if skipped:
            print(f'{len(skipped)} interaction(s) dropped: main effect excluded')
            if log is not None:
                log.extend(skipped)
        print(f"{variant}: {tr.shape[1]} columns, {len(made)} pairs -> "
              f"{sum('__x__' in c for c in tr.columns)} interaction terms, "
              f"{len(drop)} dropped, {n_quad} quadratics, {n_ind} indicator columns")
        return tr, te

