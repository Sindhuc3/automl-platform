from typing import Any, Dict, Tuple
import pandas as pd


def clean_rows(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    original = len(df)
    missing_mask = df[target].isna()
    missing_ids = df.index[missing_mask].tolist()
    work = df.loc[~missing_mask].copy()

    duplicate_mask = work.duplicated(keep="first")
    duplicate_ids = work.index[duplicate_mask].tolist()
    work = work.loc[~duplicate_mask].copy()

    return work, {
        "original": original,
        "missing_target_removed": int(missing_mask.sum()),
        "missing_target_row_ids": [int(x) if isinstance(x, int) else str(x) for x in missing_ids],
        "duplicates_removed": int(duplicate_mask.sum()),
        "duplicate_row_ids": [int(x) if isinstance(x, int) else str(x) for x in duplicate_ids],
        "final": len(work),
    }
