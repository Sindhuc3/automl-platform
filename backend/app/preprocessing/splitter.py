from typing import Any, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from app.preprocessing.constants import TEST_SIZE, RANDOM_SEED


def split_data(X: pd.DataFrame, y: pd.Series, problem_type: str):
    stratify = None
    warnings = []
    if problem_type == "classification":
        counts = y.value_counts(dropna=False)
        if len(counts) >= 2 and counts.min() >= 2:
            stratify = y
        else:
            warnings.append({"code": "PRE_NON_STRATIFIED_FALLBACK", "message": "Stratified splitting was not feasible; a normal random split was used."})
    try:
        return (*train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=stratify), warnings)
    except ValueError as exc:
        if stratify is not None:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
            warnings.append({"code": "PRE_NON_STRATIFIED_FALLBACK", "message": str(exc)})
            return Xtr, Xte, ytr, yte, warnings
        raise
