from __future__ import annotations

from typing import Any, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from app.preprocessing.constants import OHE_MIN_FREQUENCY, OHE_MAX_CATEGORIES


# Explicit semantic hints are intentionally conservative.  We only infer an
# order when the column name and/or values provide evidence; we never invent
# an order merely because categories happen to be integers or strings.
ORDINAL_NAME_HINTS = (
    "grade", "level", "rank", "rating", "score", "severity", "priority",
    "quality", "stage", "class", "satisfaction", "education", "experience",
    "status", "tier", "size",
)

ORDER_GROUPS = [
    ("strongly disagree", "disagree", "neutral", "agree", "strongly agree"),
    ("very dissatisfied", "dissatisfied", "neutral", "satisfied", "very satisfied"),
    ("very poor", "poor", "fair", "good", "excellent"),
    ("very low", "low", "medium", "high", "very high"),
    ("low", "medium", "high"),
    ("small", "medium", "large"),
    ("beginner", "intermediate", "advanced", "expert"),
    ("junior", "mid", "senior"),
    ("basic", "intermediate", "advanced"),
    ("bronze", "silver", "gold", "platinum"),
]


def _normalise(value: Any) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").replace("-", " ").split())


def _ordered_values_from_labels(values: Iterable[Any]) -> Optional[List[Any]]:
    raw = list(values)
    normalised = {_normalise(v): v for v in raw}
    present = set(normalised)
    for group in ORDER_GROUPS:
        if present and present.issubset(set(group)) and len(present) >= 2:
            return [normalised[label] for label in group if label in present]
    return None


def detect_ordinal_order(column: str, series: pd.Series, data_type: str) -> Optional[List[Any]]:
    """Return an order only when there is defensible semantic evidence."""
    non_null = series.dropna()
    if non_null.empty or non_null.nunique() < 2:
        return None

    # Preserve an explicitly ordered pandas categorical.
    if isinstance(series.dtype, pd.CategoricalDtype) and series.dtype.ordered:
        return list(series.dtype.categories)

    label_order = _ordered_values_from_labels(non_null.unique())
    if label_order is not None:
        return label_order

    name = _normalise(column)
    has_hint = any(token in name.split() or token in name for token in ORDINAL_NAME_HINTS)

    # Numeric categorical values are considered ordinal only with semantic
    # naming evidence.  This avoids turning arbitrary codes into fake orders.
    if data_type == "numeric_categorical" and has_hint:
        numeric = pd.to_numeric(non_null, errors="coerce").dropna()
        if len(numeric) == len(non_null) and numeric.nunique() <= 20:
            return sorted(numeric.unique().tolist())

    return None


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Leakage-safe frequency encoding for high-cardinality categories."""

    def __init__(self, unknown_value: float = 0.0):
        self.unknown_value = unknown_value

    def fit(self, X, y=None):
        frame = self._frame(X)
        self.columns_ = list(frame.columns)
        self.maps_ = {}
        for col in self.columns_:
            values = frame[col].astype(object).where(frame[col].notna(), "__MISSING__")
            self.maps_[col] = values.value_counts(normalize=True, dropna=False).to_dict()
        return self

    def transform(self, X):
        frame = self._frame(X)
        out = np.zeros((len(frame), len(self.columns_)), dtype=float)
        for i, col in enumerate(self.columns_):
            values = frame[col].astype(object).where(frame[col].notna(), "__MISSING__")
            out[:, i] = values.map(self.maps_[col]).fillna(self.unknown_value).astype(float).to_numpy()
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else self.columns_, dtype=object)

    @staticmethod
    def _frame(X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)


def make_one_hot_encoder():
    try:
        return OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=OHE_MIN_FREQUENCY,
            max_categories=OHE_MAX_CATEGORIES,
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=OHE_MIN_FREQUENCY,
            max_categories=OHE_MAX_CATEGORIES,
            sparse=False,
        )


def make_ordinal_encoder(categories: List[Any]):
    return OrdinalEncoder(
        categories=[categories],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )


def automatic_encoding_decision(
    column: str,
    data_type: str,
    unique_count: int,
    unique_ratio: float,
    missing_ratio: float,
    series: Optional[pd.Series] = None,
) -> dict:
    """Choose a representation from feature semantics, not a blanket default."""
    if data_type == "boolean":
        return {"encoding": "boolean", "method": "0_1", "reason_code": "BOOLEAN_BINARY"}

    if unique_count > OHE_MAX_CATEGORIES or unique_ratio >= 0.50:
        return {
            "encoding": "frequency",
            "method": "FrequencyEncoder",
            "reason_code": "HIGH_CARDINALITY_FREQUENCY",
            "reason": "High-cardinality nominal values are compressed to training-derived frequencies instead of creating many one-hot columns.",
        }

    if series is not None:
        order = detect_ordinal_order(column, series, data_type)
        if order is not None:
            return {
                "encoding": "ordinal",
                "method": "OrdinalEncoder",
                "categories": order,
                "reason_code": "MEANINGFUL_ORDER_DETECTED",
                "reason": "The feature has an explicit or strongly indicated semantic order, so ordinal encoding preserves that order.",
            }

    return {
        "encoding": "one_hot",
        "method": "OneHotEncoder",
        "reason_code": "NOMINAL_NO_ORDER",
        "reason": "No defensible semantic order was detected, so one-hot encoding avoids inventing a relationship between categories.",
    }
