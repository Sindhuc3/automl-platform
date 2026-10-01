from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


class CompactFeatureSelector(BaseEstimator, TransformerMixin):
    """Leakage-safe M4 Compact recipe executed inside one CV training fold.

    Steps:
      1. remove constant/near-constant columns,
      2. prune highly correlated numeric features, keeping the feature with
         stronger training-fold mutual information,
      3. compare MI against a small shuffled-target null distribution,
      4. retain at least ``min_features`` when possible.

    It is intentionally bounded for M5 screening; M6 can later tune a richer
    selector recipe.
    """

    def __init__(
        self,
        correlation_threshold: float = 0.98,
        n_shuffles: int = 5,
        null_quantile: float = 0.95,
        min_features: int = 5,
        random_state: int = 42,
        task: str | None = None,
    ):
        self.correlation_threshold = correlation_threshold
        self.n_shuffles = n_shuffles
        self.null_quantile = null_quantile
        self.min_features = min_features
        self.random_state = random_state
        self.task = task

    def fit(self, X: Any, y: Any):
        Xdf = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        yv = np.asarray(y)
        if len(Xdf) != len(yv):
            raise ValueError("M5_SELECTOR_LENGTH_MISMATCH")
        if Xdf.shape[1] == 0:
            raise ValueError("M5_SELECTOR_NO_FEATURES")

        numeric = Xdf.apply(pd.to_numeric, errors="coerce")
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        numeric = numeric.fillna(numeric.median(numeric_only=True)).fillna(0.0)

        # Target-independent constant removal.
        variances = numeric.var(axis=0, ddof=0)
        candidates = [c for c in numeric.columns if float(variances[c]) > 0.0]
        if not candidates:
            candidates = list(numeric.columns)

        work = numeric.loc[:, candidates]
        self.mi_scores_ = self._mi(work, yv)

        # Redundancy pruning: within each correlated group retain the feature
        # with the larger training-fold MI.
        corr = work.corr(method="spearman").abs().fillna(0.0)
        keep = list(work.columns)
        for i, a in enumerate(work.columns):
            if a not in keep:
                continue
            for b in work.columns[i + 1:]:
                if b not in keep:
                    continue
                if float(corr.loc[a, b]) >= self.correlation_threshold:
                    drop = b if self.mi_scores_.get(a, 0.0) >= self.mi_scores_.get(b, 0.0) else a
                    if drop in keep and len(keep) > 1:
                        keep.remove(drop)
                        if drop == a:
                            break

        # MI-null threshold is estimated using shuffled labels/targets on the
        # same training fold. This is target-dependent but strictly fold-local.
        rng = np.random.default_rng(self.random_state)
        null_maxima = {c: [] for c in keep}
        for _ in range(max(1, self.n_shuffles)):
            shuffled = rng.permutation(yv)
            shuffled_mi = self._mi(work.loc[:, keep], shuffled)
            for c in keep:
                null_maxima[c].append(shuffled_mi.get(c, 0.0))

        thresholds = {
            c: float(np.quantile(null_maxima[c], self.null_quantile))
            if null_maxima[c] else 0.0
            for c in keep
        }
        selected = [c for c in keep if self.mi_scores_.get(c, 0.0) > thresholds[c]]

        # Never collapse a valid dataset to zero features. If the null gate is
        # too strict, retain the strongest training-fold MI features.
        minimum = min(max(1, int(self.min_features)), len(keep))
        if len(selected) < minimum:
            selected = sorted(keep, key=lambda c: self.mi_scores_.get(c, 0.0), reverse=True)[:minimum]

        self.selected_features_ = list(selected)
        self.feature_names_in_ = np.asarray(Xdf.columns, dtype=object)
        return self

    @staticmethod
    def _mi(X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
        if X.shape[1] == 0:
            return {}
        try:
            classes = np.unique(y)
            is_classification = self.task == "classification" or (
                self.task is None and len(classes) <= 20 and y.dtype.kind in "biuO"
            )
            if is_classification:
                values = mutual_info_classif(
                    X.to_numpy(dtype=float), y, random_state=42
                )
            else:
                values = mutual_info_regression(
                    X.to_numpy(dtype=float), y, random_state=42
                )
        except Exception:
            # The matrix is expected to be numeric after M3. If a downstream
            # recipe supplies an unusual target, fall back to deterministic
            # absolute Pearson association as a conservative proxy.
            values = []
            yn = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy()
            for c in X.columns:
                xv = X[c].to_numpy(dtype=float)
                if np.std(xv) == 0 or np.std(yn) == 0:
                    values.append(0.0)
                else:
                    values.append(abs(float(np.corrcoef(xv, yn)[0, 1])))
            values = np.asarray(values)
        return {str(c): float(v) for c, v in zip(X.columns, values)}

    def transform(self, X: Any):
        Xdf = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        missing = [c for c in self.selected_features_ if c not in Xdf.columns]
        if missing:
            raise ValueError(f"M5_SELECTOR_FEATURES_MISSING: {missing}")
        return Xdf.loc[:, self.selected_features_].copy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_features_, dtype=object)
