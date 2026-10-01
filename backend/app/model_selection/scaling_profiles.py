"""Model-level scaling declarations for M5.

Actual transformers are supplied by M3. M5 only selects the requested
profile; it does not fit a scaler itself.
"""

SCALING_PROFILES = {
    "none": {"name": "No scaling", "fit_scope": "m3_pipeline"},
    "standard": {"name": "Standard scaling", "fit_scope": "m3_pipeline"},
}


def get_scaling_profile(name: str) -> dict:
    try:
        return dict(SCALING_PROFILES[name])
    except KeyError as exc:
        raise ValueError(f"M5_UNKNOWN_SCALING_PROFILE: {name}") from exc
