
from typing import Optional, List
from sklearn.preprocessing import OneHotEncoder
from app.preprocessing.constants import OHE_MIN_FREQUENCY, OHE_MAX_CATEGORIES


def make_one_hot_encoder():
    """Safe V1 automatic encoder for nominal categories."""
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


def automatic_encoding_decision(
    column: str,
    data_type: str,
    unique_count: int,
    unique_ratio: float,
    missing_ratio: float,
) -> dict:
    """
    Automatic V1 rule:
      - nominal categorical -> one-hot
      - numeric-categorical -> one-hot unless a reliable order is explicitly supplied
      - high-cardinality -> exclude rather than invent target encoding
      - ordinal encoding is reserved for Guided Mode where the user supplies order.
    """
    if unique_count > OHE_MAX_CATEGORIES or unique_ratio >= 0.50:
        return {
            "encoding": "excluded_high_cardinality",
            "method": None,
            "reason_code": "HIGH_CARDINALITY",
        }

    return {
        "encoding": "one_hot",
        "method": "OneHotEncoder",
        "reason_code": "NOMINAL_SAFE_DEFAULT",
    }
