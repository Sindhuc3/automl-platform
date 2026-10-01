from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold


@dataclass(frozen=True)
class Fold:
    """One materialized train/validation fold."""
    repeat: int
    fold: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]

    @property
    def n_train(self) -> int:
        return len(self.train_indices)

    @property
    def n_validation(self) -> int:
        return len(self.validation_indices)


@dataclass(frozen=True)
class CVPlan:
    """Frozen, reusable CV plan shared by every M5 candidate."""
    cv_plan_id: str
    task: str
    n_splits: int
    n_repeats: int
    shuffle: bool
    random_seed: int
    strategy: str
    folds: tuple[Fold, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    def iter_folds(self) -> Iterator[Fold]:
        return iter(self.folds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cv_plan_id": self.cv_plan_id,
            "task": self.task,
            "n_splits": self.n_splits,
            "n_repeats": self.n_repeats,
            "shuffle": self.shuffle,
            "random_seed": self.random_seed,
            "strategy": self.strategy,
            "notes": list(self.notes),
            "folds": [
                {
                    "repeat": f.repeat,
                    "fold": f.fold,
                    "train_indices": list(f.train_indices),
                    "validation_indices": list(f.validation_indices),
                }
                for f in self.folds
            ],
        }


def _classification_splits(y: np.ndarray, requested: int) -> int:
    counts = np.unique(y, return_counts=True)[1]
    if len(counts) < 2:
        raise ValueError("M5_CV_CLASSIFICATION_NEEDS_TWO_CLASSES")
    return min(requested, int(counts.min()))


def build_cv_plan(
    X: Any,
    y: Any,
    *,
    task: str,
    random_seed: int = 42,
    requested_splits: int = 5,
    requested_repeats: int | None = None,
    cv_plan_id: str = "m5_cv_v1",
    quantile_stratify_regression: bool = True,
) -> CVPlan:
    """Materialize deterministic folds once so all candidates see identical folds.

    Classification uses stratification. Regression uses shuffled KFold; when the
    sample size is below 1000 or the target is strongly skewed, approximate
    quantile-bin stratification is used when feasible.
    """
    task = task.lower().strip()
    n = len(y)
    if n < 2:
        raise ValueError("M5_CV_TOO_FEW_ROWS")
    if task not in {"classification", "regression"}:
        raise ValueError(f"M5_CV_UNSUPPORTED_TASK: {task}")

    notes: list[str] = []
    repeats = requested_repeats
    if repeats is None:
        repeats = 3 if n < 500 else 1
    repeats = max(1, int(repeats))

    y_array = np.asarray(y)
    if task == "classification":
        n_splits = _classification_splits(y_array, requested_splits)
        if n_splits < 2:
            raise ValueError("M5_CV_CLASS_MINORITY_TOO_SMALL")
        if n_splits != requested_splits:
            notes.append(f"n_splits_reduced_to_{n_splits}_because_of_minority_count")
        if repeats > 1:
            splitter = RepeatedStratifiedKFold(
                n_splits=n_splits, n_repeats=repeats, random_state=random_seed
            )
            strategy = "repeated_stratified_kfold"
        else:
            splitter = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_seed
            )
            strategy = "stratified_kfold"
        split_iter = splitter.split(np.zeros(n), y_array)
    else:
        n_splits = min(requested_splits, n)
        if n_splits < 2:
            raise ValueError("M5_CV_REGRESSION_NEEDS_TWO_FOLDS")
        use_quantile = False
        if quantile_stratify_regression:
            finite = y_array[np.isfinite(y_array)] if np.issubdtype(y_array.dtype, np.number) else np.array([])
            if finite.size >= 20:
                skew = float(__import__("pandas").Series(finite).skew())
                use_quantile = n < 1000 or abs(skew) > 1.0
        if use_quantile:
            try:
                import pandas as pd
                q = min(max(n_splits, 2), 10)
                bins = pd.qcut(pd.Series(y_array), q=q, labels=False, duplicates="drop").to_numpy()
                if len(np.unique(bins)) >= n_splits and np.bincount(bins).min() >= 2:
                    if repeats > 1:
                        splitter = RepeatedStratifiedKFold(
                            n_splits=n_splits, n_repeats=repeats, random_state=random_seed
                        )
                        split_iter = splitter.split(np.zeros(n), bins)
                    else:
                        splitter = StratifiedKFold(
                            n_splits=n_splits, shuffle=True, random_state=random_seed
                        )
                        split_iter = splitter.split(np.zeros(n), bins)
                    strategy = "quantile_stratified_repeated_kfold" if repeats > 1 else "quantile_stratified_kfold"
                    notes.append("regression_folds_stratified_by_target_quantile_bins")
                else:
                    raise ValueError("insufficient_bin_support")
            except Exception:
                splitter = RepeatedKFold(n_splits=n_splits, n_repeats=repeats, random_state=random_seed)
                split_iter = splitter.split(np.zeros(n))
                strategy = "repeated_kfold" if repeats > 1 else "kfold"
                notes.append("quantile_stratification_unavailable_fell_back_to_kfold")
        else:
            splitter = RepeatedKFold(n_splits=n_splits, n_repeats=repeats, random_state=random_seed)
            split_iter = splitter.split(np.zeros(n))
            strategy = "repeated_kfold" if repeats > 1 else "kfold"

    folds: list[Fold] = []
    for i, (train_idx, val_idx) in enumerate(split_iter):
        repeat = i // n_splits
        fold = i % n_splits
        if task == "classification":
            train_classes = set(np.asarray(y_array)[train_idx])
            val_classes = set(np.asarray(y_array)[val_idx])
            if train_classes != set(np.asarray(y_array)) or not val_classes.issubset(train_classes):
                raise ValueError("M5_CV_CLASS_PRESENCE_VIOLATION")
        folds.append(Fold(repeat, fold, tuple(map(int, train_idx)), tuple(map(int, val_idx))))

    return CVPlan(
        cv_plan_id=cv_plan_id,
        task=task,
        n_splits=n_splits,
        n_repeats=repeats,
        shuffle=True,
        random_seed=random_seed,
        strategy=strategy,
        folds=tuple(folds),
        notes=tuple(notes),
    )
