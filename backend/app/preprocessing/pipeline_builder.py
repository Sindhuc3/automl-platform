
from typing import Dict, List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from app.preprocessing.encoding import make_one_hot_encoder


def build_pipeline(
    df_train: pd.DataFrame,
    numeric: List[str],
    categorical: List[str],
    numeric_categorical: List[str],
    boolean: List[str],
    missing_plans: Dict[str, dict],
    scaled: bool = False,
) -> ColumnTransformer:
    """
    Build an unfitted, leakage-safe recipe.

    Numeric continuous:
      median imputation; StandardScaler only in scaled profile.
      Missing indicators are separate and are never scaled.

    Categorical/numeric-categorical:
      one-hot; low missingness -> most frequent, >=5% -> explicit
      __MISSING__ category.

    Boolean:
      most-frequent imputation and 0/1 representation; missing indicator
      is separate when the missingness threshold is met.
    """
    transformers = []

    numeric_with_indicator = [
        c for c in numeric
        if missing_plans.get(c, {}).get("indicator", False)
    ]
    numeric_without_indicator = [
        c for c in numeric if c not in numeric_with_indicator
    ]

    if numeric_without_indicator:
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if scaled:
            steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(steps), numeric_without_indicator))

    if numeric_with_indicator:
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if scaled:
            steps.append(("scaler", StandardScaler()))
        transformers.append((
            "numeric_missing",
            Pipeline(steps),
            numeric_with_indicator,
        ))
        transformers.append((
            "numeric_missing_indicators",
            MissingIndicator(features="all"),
            numeric_with_indicator,
        ))

    # Boolean: preserve as 0/1, not one-hot.
    if boolean:
        bool_indicator = [
            c for c in boolean
            if missing_plans.get(c, {}).get("indicator", False)
        ]
        bool_without_indicator = [c for c in boolean if c not in bool_indicator]

        if bool_without_indicator:
            transformers.append((
                "boolean",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("to_int", FunctionTransformer(lambda x: x.astype(int), feature_names_out="one-to-one")),
                ]),
                bool_without_indicator,
            ))

        if bool_indicator:
            transformers.append((
                "boolean_missing",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("to_int", FunctionTransformer(lambda x: x.astype(int), feature_names_out="one-to-one")),
                ]),
                bool_indicator,
            ))
            transformers.append((
                "boolean_missing_indicators",
                MissingIndicator(features="all"),
                bool_indicator,
            ))

    cat_cols = categorical + numeric_categorical
    if cat_cols:
        most_frequent = [
            c for c in cat_cols
            if missing_plans.get(c, {}).get("method") == "most_frequent"
        ]
        constant = [
            c for c in cat_cols
            if missing_plans.get(c, {}).get("method") == "constant:__MISSING__"
        ]
        no_missing = [
            c for c in cat_cols
            if c not in most_frequent and c not in constant
        ]

        if most_frequent:
            transformers.append((
                "categorical_most_frequent",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", make_one_hot_encoder()),
                ]),
                most_frequent,
            ))

        if constant:
            transformers.append((
                "categorical_missing_category",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                    ("encoder", make_one_hot_encoder()),
                ]),
                constant,
            ))

        if no_missing:
            transformers.append((
                "categorical_no_missing",
                Pipeline([
                    ("encoder", make_one_hot_encoder()),
                ]),
                no_missing,
            ))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
