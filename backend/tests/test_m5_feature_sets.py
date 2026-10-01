import numpy as np
import pandas as pd

from app.model_selection.cv.feature_sets import CompactFeatureSelector


def test_compact_selector_is_fit_only_on_training_fold():
    rng = np.random.default_rng(42)
    y = np.array([0, 1] * 40)
    X = pd.DataFrame({
        "signal": y + rng.normal(0, 0.1, 80),
        "duplicate_signal": y + rng.normal(0, 0.1, 80),
        "noise": rng.normal(0, 1, 80),
        "constant": 1.0,
    })

    selector = CompactFeatureSelector(
        correlation_threshold=0.98,
        n_shuffles=2,
        null_quantile=0.95,
        min_features=2,
        random_state=42,
    )
    selector.fit(X.iloc[:60], y[:60])
    transformed = selector.transform(X.iloc[60:])
    assert transformed.shape[0] == 20
    assert transformed.shape[1] >= 1
    assert "constant" not in selector.selected_features_
    assert set(selector.get_feature_names_out()).issubset(set(X.columns))
