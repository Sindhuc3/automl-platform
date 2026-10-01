from typing import Dict, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, FunctionTransformer

from app.preprocessing.encoding import make_one_hot_encoder, make_ordinal_encoder, FrequencyEncoder


def _boolean_to_int(x):
    return x.astype(int)


def _imputer_for(plan: dict):
    method = (plan or {}).get("method")
    if method in {"mean", "median", "most_frequent"}:
        return SimpleImputer(strategy=method)
    if method == "constant:__MISSING__":
        return SimpleImputer(strategy="constant", fill_value="__MISSING__")
    return None


def _group_by_plan(columns: List[str], decisions_by_col: Dict[str, dict], key: str, values: set):
    return [c for c in columns if (decisions_by_col.get(c, {}).get(key) in values)]


def build_pipeline(
    df_train: pd.DataFrame,
    numeric: List[str],
    categorical: List[str],
    numeric_categorical: List[str],
    boolean: List[str],
    missing_plans: Dict[str, dict],
    scaled: bool = False,
    encoding_plans: Dict[str, dict] | None = None,
) -> ColumnTransformer:
    """Build one unfitted recipe; actual statistics are fitted only on X_train."""
    encoding_plans = encoding_plans or {}
    transformers = []

    # Numeric continuous columns are grouped by imputation strategy so the
    # selected automatic/guided strategy is actually executed, not just displayed.
    for method in ("mean", "median", None):
        cols = [c for c in numeric if missing_plans.get(c, {}).get("method") == method]
        if not cols:
            continue
        steps = []
        if method:
            steps.append(("imputer", SimpleImputer(strategy=method)))
        if scaled:
            steps.append(("scaler", StandardScaler()))
        transformer = Pipeline(steps) if steps else "passthrough"
        transformers.append((f"numeric_{method or 'none'}", transformer, cols))

    numeric_indicators = [c for c in numeric if missing_plans.get(c, {}).get("indicator", False)]
    if numeric_indicators:
        transformers.append(("numeric_missing_indicators", MissingIndicator(features="all"), numeric_indicators))

    # Boolean values are represented as 0/1.  This is a representation choice,
    # not nominal one-hot encoding.
    for group_name, cols in (("boolean", boolean),):
        if cols:
            transformers.append((group_name, Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("to_int", FunctionTransformer(_boolean_to_int, feature_names_out="one-to-one")),
            ]), cols))
            bool_indicators = [c for c in cols if missing_plans.get(c, {}).get("indicator", False)]
            if bool_indicators:
                transformers.append(("boolean_missing_indicators", MissingIndicator(features="all"), bool_indicators))

    cat_cols = categorical + numeric_categorical
    for encoding in ("one_hot", "ordinal", "frequency"):
        cols = [c for c in cat_cols if encoding_plans.get(c, {}).get("encoding") == encoding]
        if not cols:
            continue

        # Missing strategy can differ by column, so build a pipeline per
        # (encoding, imputation) combination.
        for missing_method in ("most_frequent", "constant:__MISSING__", None):
            selected = [c for c in cols if missing_plans.get(c, {}).get("method") == missing_method or (missing_method is None and not missing_plans.get(c, {}).get("method"))]
            if not selected:
                continue
            steps = []
            if missing_method:
                steps.append(("imputer", _imputer_for(missing_plans[selected[0]])))
            if encoding == "one_hot":
                steps.append(("encoder", make_one_hot_encoder()))
            elif encoding == "frequency":
                steps.append(("encoder", FrequencyEncoder()))
            else:
                # Ordinal categories may differ by column, so they cannot be
                # represented by one shared OrdinalEncoder with one category list.
                for col in selected:
                    pass
                # handled below per column
                steps = None

            if steps is not None:
                transformers.append((f"{encoding}_{missing_method or 'none'}", Pipeline(steps), selected))

    # Ordinal columns require their own category order.
    ordinal_cols = [c for c in cat_cols if encoding_plans.get(c, {}).get("encoding") == "ordinal"]
    for col in ordinal_cols:
        plan = encoding_plans[col]
        steps = []
        method = missing_plans.get(col, {}).get("method")
        categories = list(plan["categories"])
        if method == "constant:__MISSING__":
            # Keep the missing state distinct without mixing incompatible
            # numeric/string category types inside OrdinalEncoder.
            numeric_categories = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in categories)
            sentinel = -1 if numeric_categories else "__MISSING__"
            categories = categories + [sentinel]
            steps.append(("imputer", SimpleImputer(strategy="constant", fill_value=sentinel)))
        elif method:
            steps.append(("imputer", _imputer_for(missing_plans[col])))
        steps.append(("encoder", make_ordinal_encoder(categories)))
        transformers.append((f"ordinal_{col}", Pipeline(steps), [col]))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
