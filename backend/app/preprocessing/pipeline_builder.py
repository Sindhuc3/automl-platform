from typing import Dict, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, FunctionTransformer

from app.preprocessing.encoding import make_one_hot_encoder


def _boolean_to_int(x):
    """
    Convert boolean values to integer 0/1.

    This is defined at module level so the preprocessing
    pipeline can be safely serialized with pickle.
    """
    return x.astype(int)


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
    Build an unfitted, leakage-safe preprocessing recipe.

    Numeric continuous:
      - median imputation
      - StandardScaler only in the scaled profile
      - missing indicators are separate and never scaled

    Categorical / numeric-categorical:
      - one-hot encoding
      - low missingness -> most frequent
      - >=5% missingness -> explicit __MISSING__ category

    Boolean:
      - most-frequent imputation
      - 0/1 representation
      - missing indicator when the missingness threshold is met
    """

    transformers = []

    # ---------------------------------------------------------
    # NUMERIC FEATURES
    # ---------------------------------------------------------

    numeric_with_indicator = [
        c
        for c in numeric
        if missing_plans.get(c, {}).get("indicator", False)
    ]

    numeric_without_indicator = [
        c
        for c in numeric
        if c not in numeric_with_indicator
    ]

    # Numeric columns without missingness indicators
    if numeric_without_indicator:
        steps = [
            ("imputer", SimpleImputer(strategy="median"))
        ]

        if scaled:
            steps.append(
                ("scaler", StandardScaler())
            )

        transformers.append(
            (
                "numeric",
                Pipeline(steps),
                numeric_without_indicator,
            )
        )

    # Numeric columns with missingness indicators
    if numeric_with_indicator:
        steps = [
            ("imputer", SimpleImputer(strategy="median"))
        ]

        if scaled:
            steps.append(
                ("scaler", StandardScaler())
            )

        transformers.append(
            (
                "numeric_missing",
                Pipeline(steps),
                numeric_with_indicator,
            )
        )

        transformers.append(
            (
                "numeric_missing_indicators",
                MissingIndicator(features="all"),
                numeric_with_indicator,
            )
        )

    # ---------------------------------------------------------
    # BOOLEAN FEATURES
    # ---------------------------------------------------------

    bool_indicator = [
        c
        for c in boolean
        if missing_plans.get(c, {}).get("indicator", False)
    ]

    bool_without_indicator = [
        c
        for c in boolean
        if c not in bool_indicator
    ]

    # Boolean columns without missingness indicators
    if bool_without_indicator:
        transformers.append(
            (
                "boolean",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "to_int",
                            FunctionTransformer(
                                _boolean_to_int,
                                feature_names_out="one-to-one",
                            ),
                        ),
                    ]
                ),
                bool_without_indicator,
            )
        )

    # Boolean columns with missingness indicators
    if bool_indicator:
        transformers.append(
            (
                "boolean_missing",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "to_int",
                            FunctionTransformer(
                                _boolean_to_int,
                                feature_names_out="one-to-one",
                            ),
                        ),
                    ]
                ),
                bool_indicator,
            )
        )

        transformers.append(
            (
                "boolean_missing_indicators",
                MissingIndicator(features="all"),
                bool_indicator,
            )
        )

    # ---------------------------------------------------------
    # CATEGORICAL FEATURES
    # ---------------------------------------------------------

    cat_cols = categorical + numeric_categorical

    if cat_cols:

        # Less than 5% missing -> most frequent
        most_frequent = [
            c
            for c in cat_cols
            if missing_plans.get(c, {}).get("method")
            == "most_frequent"
        ]

        # 5% to 60% missing -> explicit missing category
        constant = [
            c
            for c in cat_cols
            if missing_plans.get(c, {}).get("method")
            == "constant:__MISSING__"
        ]

        # No missing-value strategy required
        no_missing = [
            c
            for c in cat_cols
            if c not in most_frequent
            and c not in constant
        ]

        # ---------------------------------------------
        # Most-frequent categorical imputation
        # ---------------------------------------------

        if most_frequent:
            transformers.append(
                (
                    "categorical_most_frequent",
                    Pipeline(
                        [
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="most_frequent"
                                ),
                            ),
                            (
                                "encoder",
                                make_one_hot_encoder(),
                            ),
                        ]
                    ),
                    most_frequent,
                )
            )

        # ---------------------------------------------
        # Explicit missing category
        # ---------------------------------------------

        if constant:
            transformers.append(
                (
                    "categorical_missing_category",
                    Pipeline(
                        [
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="constant",
                                    fill_value="__MISSING__",
                                ),
                            ),
                            (
                                "encoder",
                                make_one_hot_encoder(),
                            ),
                        ]
                    ),
                    constant,
                )
            )

        # ---------------------------------------------
        # Categorical columns without missing values
        # ---------------------------------------------

        if no_missing:
            transformers.append(
                (
                    "categorical_no_missing",
                    Pipeline(
                        [
                            (
                                "encoder",
                                make_one_hot_encoder(),
                            )
                        ]
                    ),
                    no_missing,
                )
            )

    # ---------------------------------------------------------
    # FINAL COLUMN TRANSFORMER
    # ---------------------------------------------------------

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )