
import hashlib
import json
import pickle
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from app.config import STORAGE_DIR
from app.database.mongodb import datasets_collection, preprocessing_runs_collection
from app.preprocessing.contract import validate_contract
from app.preprocessing.row_cleaning import clean_rows
from app.preprocessing.column_decisions import build_column_decisions
from app.preprocessing.outliers import fit_iqr_caps, cap_outliers
from app.preprocessing.pipeline_builder import build_pipeline
from app.preprocessing.splitter import split_data
from app.preprocessing.constants import (
    MIN_ROWS_HARD,
    MIN_ROWS_SOFT,
    TEST_SIZE,
    RANDOM_SEED,
    WIDE_OUTPUT_LIMIT,
)


def _now():
    return datetime.now(timezone.utc)


def _safe(x):
    return json.loads(json.dumps(x, default=str))


def _fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _feature_names(pipe):
    try:
        return [str(x) for x in pipe.get_feature_names_out()]
    except Exception:
        return []


def _column_type_lists(decisions):
    used = [d for d in decisions if d.get("action") == "use"]
    numeric = [d["column"] for d in used if d.get("data_type") == "numeric"]
    categorical = [d["column"] for d in used if d.get("data_type") == "categorical"]
    numeric_categorical = [
        d["column"] for d in used if d.get("data_type") == "numeric_categorical"
    ]
    boolean = [d["column"] for d in used if d.get("data_type") == "boolean"]
    return numeric, categorical, numeric_categorical, boolean


def _training_stats(Xtr, column, data_type):
    s = Xtr[column]
    missing = int(s.isna().sum())
    stats = {"training_rows": len(s), "missing_count": missing}

    if data_type == "numeric":
        vals = pd.to_numeric(s, errors="coerce").dropna()
        if len(vals):
            stats["median"] = float(vals.median())
    elif data_type in {"categorical", "numeric_categorical", "boolean"}:
        mode = s.dropna().mode()
        if len(mode):
            stats["mode"] = _json_value(mode.iloc[0])
    return stats


def _explanation_for_decision(
    decision: dict,
    df: pd.DataFrame,
    Xtr: pd.DataFrame,
    outlier_info: dict,
    scaled_profile: bool,
):
    col = decision["column"]
    if decision.get("category") == "target":
        return {
            "why": (
                f"{col} is the user-selected target, so it must be separated from the input features. "
                "Keeping the target outside the feature transformation prevents preprocessing from using "
                "the value we are trying to predict."
            ),
            "what_changed": "The selected target was separated into y; it was not transformed as an input feature.",
            "learned_from": "Target selection and Module 2 validation.",
            "modeling_effect": "Used as the prediction target, not as an input feature.",
        }

    observed = decision.get("observed") or {}
    row_count = observed.get("row_count", len(df))
    missing_count = observed.get("missing_count", int(df[col].isna().sum()))
    missing_pct = observed.get("missing_percent", round(100 * missing_count / max(1, row_count), 2))
    unique = observed.get("unique_count", int(df[col].nunique(dropna=True)))

    if decision.get("action") == "exclude":
        reason = decision.get("reason_code", "")
        if reason == "EXCL_HIGH_MISSING":
            why = (
                f"{col} has {missing_count} missing values out of {row_count} rows ({missing_pct:.2f}%). "
                "That level of missingness leaves too little observed information for the default automatic "
                "pipeline to represent the feature reliably. Rather than inventing values for most of the column, "
                "the pipeline excludes it while preserving the original dataset unchanged."
            )
        else:
            why = (
                f"{col} was classified by an earlier validation stage as {decision.get('reason_code', 'unsafe')}. "
                "The Module 3 automatic pipeline keeps that semantic decision and excludes the column from the "
                "model input rather than silently changing its meaning."
            )
        return {
            "why": why,
            "what_changed": "The column was excluded from the derived modeling dataset; the validated source dataset was not modified.",
            "learned_from": "Module 1 profiling and Module 2 validation, plus Module 3 training-set statistics where applicable.",
            "modeling_effect": "Not supplied to the model.",
        }

    data_type = decision.get("data_type")
    missing_plan = decision.get("missing_plan") or {}
    out = outlier_info.get(col, {})
    sentences = []

    if data_type == "numeric":
        if missing_count:
            method = missing_plan.get("method", "median")
            sentences.append(
                f"{col} is a numeric feature with {missing_count} missing values ({missing_pct:.2f}%) out of {row_count} rows. "
                f"The automatic recipe uses {method} imputation so the available rows are retained without dropping "
                "records simply because this feature is incomplete."
            )
        else:
            sentences.append(
                f"{col} is a numeric feature with no missing values in the modeling rows, so no missing-value "
                "imputation is required."
            )

        capped = int(out.get("train_values_capped", 0))
        if out:
            sentences.append(
                f"The training data was checked using the IQR rule; {capped} training value(s) were capped to the "
                f"learned bounds [{out['lower']:.4g}, {out['upper']:.4g}] so extreme observations do not dominate "
                "the automatic numeric representation."
            )
        if scaled_profile:
            sentences.append(
                "The scaled profile then standardizes this continuous numeric feature using statistics learned from "
                "the training split only; the missingness indicator, when present, is kept separate and is not scaled."
            )
        why = " ".join(sentences)
        what = []
        if missing_count:
            what.append(f"{missing_count} missing value(s) handled with {missing_plan.get('method', 'median')} imputation.")
        if out:
            what.append(f"{out.get('train_values_capped', 0)} training and {out.get('test_values_capped', 0)} test value(s) capped using training-derived IQR bounds.")
        if scaled_profile:
            what.append("StandardScaler applied in the scaled profile.")
        return {
            "why": why,
            "what_changed": " ".join(what) if what else "No value-level change was required for missingness or outliers.",
            "learned_from": "Training split only for imputation, outlier bounds, and scaling statistics.",
            "modeling_effect": "Continuous numeric feature retained in the transformed feature matrix.",
        }

    if data_type in {"categorical", "numeric_categorical", "boolean"}:
        encoding_text = {
            "boolean": "0/1 representation",
            "one_hot": "one-hot encoding",
            "ordinal": "ordinal encoding",
            "frequency": "training-derived frequency encoding",
        }.get(decision.get("encoding"), "categorical encoding")

        if missing_count:
            method = missing_plan.get("method", "most_frequent")
            missing_sentence = (
                f"It has {missing_count} missing values ({missing_pct:.2f}%), so the automatic recipe uses "
                f"{method.replace('constant:__MISSING__', 'an explicit missing category')} rather than removing rows."
            )
        else:
            missing_sentence = "It has no missing values in the modeling rows, so no missing-value imputation is required."

        plan = decision.get("encoding_plan") or {}
        if decision.get("encoding") == "ordinal":
            order_sentence = f"A meaningful order was detected and preserved: {plan.get('categories', [])}."
        elif decision.get("encoding") == "frequency":
            order_sentence = "The feature is high-cardinality, so training-derived category frequencies are used instead of expanding it into many one-hot columns."
        elif data_type == "numeric_categorical":
            order_sentence = "No defensible semantic order was detected, so numeric-looking categories are treated as nominal."
        else:
            order_sentence = "No defensible category order was detected, so no artificial ordering is imposed."
        why = f"{col} is detected as {data_type} with {unique} distinct observed value(s). {missing_sentence} {order_sentence} The automatic representation uses {encoding_text}."
        return {
            "why": why,
            "what_changed": (
                f"Missing values were handled according to the recorded missing-value strategy, then "
                f"{encoding_text} was applied."
            ),
            "learned_from": "Training split only for any imputation statistic; category vocabulary is fitted on training data.",
            "modeling_effect": "Categorical information is represented numerically without leaking test-set category information.",
        }

    return {
        "why": "The feature was retained using the supported automatic preprocessing recipe.",
        "what_changed": "The feature was included in the transformed dataset.",
        "learned_from": "Training data only.",
        "modeling_effect": "Feature supplied to downstream model selection.",
    }


def _attach_explanations(decisions, df, Xtr, outliers, scaled_profile):
    out = []
    for d in decisions:
        copy = dict(d)
        copy["explanation"] = _explanation_for_decision(
            copy, df, Xtr, outliers, scaled_profile
        )
        out.append(copy)
    return out


def _feature_mapping(pipe, decisions):
    names = _feature_names(pipe)
    sources = []
    for d in decisions:
        if d.get("action") == "use":
            sources.append(d["column"])

    mapping = []
    for name in names:
        source = None
        # One-hot names commonly start with source column followed by "_category".
        for col in sorted(sources, key=len, reverse=True):
            if name == col or name.startswith(f"{col}_"):
                source = col
                break
        mapping.append({
            "feature_name": name,
            "source_column": source,
            "transformation": "derived_from_source_column",
        })
    return mapping


def _write_processed_dataset(folder, pipe, X, y, filename):
    arr = pipe.transform(X)
    names = _feature_names(pipe)
    frame = pd.DataFrame(arr, index=X.index, columns=names)
    frame.insert(0, "__target__", y)
    path = folder / filename
    frame.to_csv(path, index=False)
    return path, int(frame.shape[1] - 1), int(frame.shape[0])


def _write_artifact(folder, name, value):
    path = folder / name
    path.write_text(json.dumps(_safe(value), indent=2), encoding="utf-8")
    return path


def run_preprocessing(dataset_id: str, mode="automatic", version=1, overrides=None):
    overrides = overrides or {}

    if mode not in {"automatic", "guided"}:
        raise ValueError("PRE_INVALID_MODE: mode must be automatic or guided.")

    doc = datasets_collection.find_one({"dataset_id": dataset_id}, {"_id": 0})
    if not doc:
        raise ValueError("PRE_DATASET_NOT_FOUND: Dataset not found.")

    contract = validate_contract(doc)
    path = Path(contract["validated_storage_path"])
    if not path.exists():
        raise ValueError("PRE_VALIDATED_DATASET_MISSING: Validated dataset file could not be found.")

    fingerprint_before = _fingerprint(path)
    df = pd.read_csv(path)
    fingerprint_after = _fingerprint(path)
    if fingerprint_before != fingerprint_after:
        raise ValueError("PRE_DATASET_CHANGED: Validated dataset changed while preprocessing was starting.")

    target = contract["target_column"]
    if target not in df.columns:
        raise ValueError("PRE_CONTRACT_MISSING_FIELD: target column is missing from validated dataset.")

    if len(df) < MIN_ROWS_HARD:
        raise ValueError(
            f"PRE_SMALL_DATASET: Dataset has only {len(df)} rows; at least {MIN_ROWS_HARD} rows are required."
        )

    warnings = []
    work, rows = clean_rows(df, target)

    if rows["missing_target_removed"]:
        warnings.append({
            "code": "PRE_MISSING_TARGET_ROWS_REMOVED",
            "message": f"Removed {rows['missing_target_removed']} rows with a missing target before supervised training.",
        })
    if rows["duplicates_removed"]:
        warnings.append({
            "code": "PRE_DUPLICATES_REMOVED",
            "message": f"Removed {rows['duplicates_removed']} exact duplicate rows before splitting.",
        })
    if rows["duplicates_removed"] / max(1, rows["original"]) > 0.20:
        warnings.append({
            "code": "PRE_DUPLICATES_REMOVED_HIGH",
            "message": "More than 20% of the original rows were exact duplicates; this is recorded for review.",
        })

    if len(work) < MIN_ROWS_HARD:
        raise ValueError("PRE_TOO_FEW_ROWS_AFTER_CLEANING: Too few rows remain after removing invalid modeling rows.")

    if len(work) < MIN_ROWS_SOFT:
        warnings.append({
            "code": "PRE_SMALL_DATASET_WARNING",
            "message": f"Only {len(work)} rows remain for modeling. The pipeline continues, but downstream estimates may be less stable.",
        })

    validation = doc.get("target_validation") or {}
    quality = doc.get("quality_report") or {}

    decisions = build_column_decisions(
        work,
        target,
        {
            "column_types": quality.get("column_types") or {},
            "column_profiles": quality.get("column_profiles") or [],
            "excluded_features": validation.get("excluded_features") or [],
        },
    )

    usable = [d["column"] for d in decisions if d.get("action") == "use"]
    if not usable:
        raise ValueError("PRE_NO_USABLE_FEATURES: No usable feature columns remain after automatic preprocessing decisions.")

    X = work[usable].copy()
    # Normalize Python None values in object/bool columns to real NaN so
    # sklearn's imputers apply the recorded missing-value policy consistently.
    for _col in X.columns:
        if X[_col].dtype == "object" or pd.api.types.is_bool_dtype(X[_col]):
            X[_col] = X[_col].where(X[_col].notna(), np.nan)
    y = work[target].copy()
    problem = (
        validation.get("problem_type")
        or doc.get("problem_definition", {}).get("problem_type")
        or ""
    ).lower()

    if problem == "classification":
        encoder = LabelEncoder()
        y2 = pd.Series(
            encoder.fit_transform(y.astype(str)),
            index=y.index,
            name=target,
        )
        target_mapping = {
            str(c): int(i) for i, c in enumerate(encoder.classes_)
        }
    else:
        y2 = pd.to_numeric(y, errors="coerce")
        if y2.isna().any():
            raise ValueError("PRE_INVALID_TARGET_VALUES: Regression target contains non-numeric values.")
        target_mapping = {}

    Xtr, Xte, ytr, yte, split_warnings = split_data(X, y2, problem)
    warnings.extend(split_warnings)

    numeric, categorical, numeric_categorical, boolean = _column_type_lists(decisions)

    # Build per-column missing plans from the training split. The decision thresholds
    # come from the modeling rows, but fitted statistics are learned only from Xtr.
    missing_plans = {}
    encoding_plans = {}
    for d in decisions:
        if d.get("action") == "use":
            col = d["column"]
            from app.preprocessing.missing_values import missing_plan_for_column
            override = (overrides.get("columns", {}).get(col, {}) or {}).get("imputation")
            missing_plans[col] = missing_plan_for_column(Xtr, col, d.get("data_type"), override)
            if d.get("encoding_plan"):
                encoding_plans[col] = d["encoding_plan"]

    # Outliers are capped only when there is enough evidence that IQR capping
    # is warranted. A feature with a few legitimate extremes is not modified
    # merely because it has an IQR fence. Guided mode can explicitly choose
    # iqr_cap or none per column.
    outlier_overrides = overrides.get("columns", {}) or {}
    candidate_caps = fit_iqr_caps(Xtr, numeric)
    caps = {}
    for col, bounds in candidate_caps.items():
        choice = (outlier_overrides.get(col, {}) or {}).get("outlier")
        if choice == "none":
            continue
        if choice == "iqr_cap":
            caps[col] = bounds
            continue
        s = pd.to_numeric(Xtr[col], errors="coerce").dropna()
        outside = ((s < bounds["lower"]) | (s > bounds["upper"])).mean()
        skew = abs(float(s.skew())) if len(s) > 2 and np.isfinite(s.skew()) else 0.0
        if outside >= 0.01 and (skew >= 0.75 or outside >= 0.05):
            caps[col] = bounds
    Xtr2, oc_train = cap_outliers(Xtr, caps)
    Xte2, oc_test = cap_outliers(Xte, caps)

    outliers = {}
    for c, bounds in caps.items():
        outliers[c] = {
            "method": "IQR_CAP",
            "multiplier": 1.5,
            "lower": bounds["lower"],
            "upper": bounds["upper"],
            "train_values_capped": oc_train.get(c, 0),
            "test_values_capped": oc_test.get(c, 0),
            "learned_from": "training_split_only",
        }

    # V1 automatic profiles:
    # plain = no scaling; scaled = StandardScaler on continuous numeric values.
    # This keeps preprocessing model-agnostic. Later model selection can choose the
    # appropriate profile based on model requirements.
    plain = build_pipeline(
        Xtr2, numeric, categorical, numeric_categorical, boolean,
        missing_plans, scaled=False, encoding_plans=encoding_plans
    )
    scaled = build_pipeline(
        Xtr2, numeric, categorical, numeric_categorical, boolean,
        missing_plans, scaled=True, encoding_plans=encoding_plans
    )

    plain.fit(Xtr2, ytr)
    scaled.fit(Xtr2, ytr)

    plain_train = plain.transform(Xtr2)
    plain_test = plain.transform(Xte2)
    scaled_train = scaled.transform(Xtr2)
    scaled_test = scaled.transform(Xte2)

    if plain_train.shape[1] > WIDE_OUTPUT_LIMIT:
        warnings.append({
            "code": "PRE_WIDE_OUTPUT",
            "message": f"Automatic preprocessing produced {plain_train.shape[1]} transformed features.",
        })

    run_id = (
        f"PRE-{_now().strftime('%Y%m%d%H%M%S')}-"
        f"{uuid.uuid4().hex[:8].upper()}"
    )
    folder = path.parent / "preprocessing" / run_id
    folder.mkdir(parents=True, exist_ok=True)

    with open(folder / "preprocessor_plain.pkl", "wb") as f:
        pickle.dump(plain, f)
    with open(folder / "preprocessor_scaled.pkl", "wb") as f:
        pickle.dump(scaled, f)

    split_ids = {
        "train": [_json_value(i) for i in Xtr.index],
        "test": [_json_value(i) for i in Xte.index],
    }
    _write_artifact(folder, "split_row_ids.json", split_ids)

    feature_mapping = {
        "plain": _feature_mapping(plain, decisions),
        "scaled": _feature_mapping(scaled, decisions),
    }

    decision_records = _attach_explanations(
        decisions,
        work,
        Xtr,
        outliers,
        scaled_profile=True,
    )

    # Keep a compact summary for the expandable UI cards.
    for record in decision_records:
        if record.get("action") == "use":
            data_type = record.get("data_type")
            if data_type == "numeric":
                actions = []
                if record.get("missing_plan", {}).get("action") == "impute":
                    actions.append(record["missing_plan"]["method"])
                if record["column"] in outliers:
                    actions.append("IQR capping")
                actions.append("StandardScaler in scaled profile")
                record["display_action"] = " + ".join(actions)
            elif data_type == "boolean":
                record["display_action"] = "0/1 encoding"
            else:
                record["display_action"] = {
                    "one_hot": "One-hot encoding",
                    "ordinal": "Ordinal encoding",
                    "frequency": "Frequency encoding",
                }.get(record.get("encoding"), "Categorical encoding")
        else:
            record["display_action"] = "Excluded"

    report = {
        "summary": {
            "original_rows": len(df),
            "after_cleaning": len(work),
            "missing_target_rows_removed": rows["missing_target_removed"],
            "duplicate_rows_removed": rows["duplicates_removed"],
            "train_rows": len(Xtr),
            "test_rows": len(Xte),
            "input_features": len(df.columns) - 1,
            "used_features": len(usable),
            "excluded_features": len(df.columns) - 1 - len(usable),
            "plain_output_features": int(plain_train.shape[1]),
            "scaled_output_features": int(scaled_train.shape[1]),
            "missing_values_handled": int(sum(
                d.get("missing_plan", {}).get("missing_count", 0)
                for d in decisions if d.get("action") == "use"
            )),
            "outlier_columns_treated": len(outliers),
            "outlier_values_capped_train": int(sum(v["train_values_capped"] for v in outliers.values())),
        },
        "row_accounting": rows,
        "column_decisions": decision_records,
        "outliers": outliers,
        "transformations": {
            "missing_values": {
                "numeric": "mean for approximately symmetric numeric features; median for skewed/heavy-tailed features",
                "categorical_under_5_percent": "most_frequent",
                "categorical_5_to_60_percent": "__MISSING__ category",
                "indicator_threshold": ">= 5%",
                "fit_scope": "training_split_only",
            },
            "encoding": {
                "nominal_categorical": "OneHotEncoder when unordered; FrequencyEncoder for high-cardinality nominal features",
                "numeric_categorical": "OrdinalEncoder when a defensible order is detected; otherwise OneHotEncoder",
                "boolean": "0/1",
                "unknown_categories": "infrequent_if_exist",
                "min_frequency": 0.01,
                "max_categories": 20,
                "high_cardinality": "FrequencyEncoder when safe; identifier/contact columns remain excluded by semantic validation",
            },
            "outliers": {
                "continuous_numeric_only": True,
                "method": "IQR capping",
                "multiplier": 1.5,
                "fit_scope": "training_split_only",
                "rows_deleted": 0,
            },
            "scaling": {
                "plain_profile": "no scaling",
                "scaled_profile": "StandardScaler on continuous numeric features",
                "selection": "Prepared as a reproducible candidate profile; downstream model selection chooses when scaling is required.",
            },
        },
        "split": {
            "test_size": TEST_SIZE,
            "random_seed": RANDOM_SEED,
            "stratified": problem == "classification"
            and not any(w.get("code") == "PRE_NON_STRATIFIED_FALLBACK" for w in split_warnings),
        },
        "target_mapping": target_mapping,
        "feature_mapping": feature_mapping,
        "warnings": warnings,
        "errors": [],
        "reproducibility": {
            "validated_dataset_fingerprint": fingerprint_before,
            "run_id": run_id,
            "random_seed": RANDOM_SEED,
        },
        "automatic_policy": {
            "mode": "automatic",
            "deterministic": True,
            "test_set_used_to_fit_preprocessors": False,
            "source_dataset_modified": False,
            "guided_overrides_supported": True,
        },
    }

    # Materialize the processed training/test data for both profiles.
    plain_train_path, _, _ = _write_processed_dataset(
        folder, plain, Xtr2, ytr, "processed_train_plain.csv"
    )
    plain_test_path, _, _ = _write_processed_dataset(
        folder, plain, Xte2, yte, "processed_test_plain.csv"
    )
    scaled_train_path, _, _ = _write_processed_dataset(
        folder, scaled, Xtr2, ytr, "processed_train_scaled.csv"
    )
    scaled_test_path, _, _ = _write_processed_dataset(
        folder, scaled, Xte2, yte, "processed_test_scaled.csv"
    )

    _write_artifact(folder, "preprocessing_report.json", report)
    config = {
        "mode": mode,
        "version": version,
        "overrides": overrides,
        "test_size": TEST_SIZE,
        "random_seed": RANDOM_SEED,
        "automatic_encoding_policy": "semantic order -> ordinal; unordered nominal -> one-hot; high-cardinality nominal -> frequency; boolean -> 0/1",
        "automatic_scaling_profiles": ["plain", "scaled"],
    }
    _write_artifact(folder, "preprocessing_config.json", config)
    _write_artifact(folder, "feature_mapping.json", feature_mapping)

    result = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "version": version,
        "mode": mode,
        "status": "completed",
        "inputs": {
            "target_column": target,
            "problem_type": problem,
            "validated_dataset_fingerprint": fingerprint_before,
        },
        "config_effective": config,
        "row_accounting": rows,
        "report": report,
        "artifact_paths": {
            "preprocessor_plain": str(folder / "preprocessor_plain.pkl"),
            "preprocessor_scaled": str(folder / "preprocessor_scaled.pkl"),
            "split_row_ids": str(folder / "split_row_ids.json"),
            "report": str(folder / "preprocessing_report.json"),
            "config": str(folder / "preprocessing_config.json"),
            "feature_mapping": str(folder / "feature_mapping.json"),
            "processed_train_plain": str(plain_train_path),
            "processed_test_plain": str(plain_test_path),
            "processed_train_scaled": str(scaled_train_path),
            "processed_test_scaled": str(scaled_test_path),
        },
        "created_at": _now(),
    }

    preprocessing_runs_collection.insert_one(_safe(result))
    return _safe(result)


def get_preprocessing(dataset_id, run_id=None):
    q = {"dataset_id": dataset_id}
    if run_id:
        q["run_id"] = run_id
        return preprocessing_runs_collection.find_one(q, {"_id": 0})
    return list(
        preprocessing_runs_collection.find(q, {"_id": 0})
        .sort("created_at", -1)
        .limit(20)
    )
