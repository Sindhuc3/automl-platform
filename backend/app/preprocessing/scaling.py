
from sklearn.preprocessing import StandardScaler


def make_scaler(profile: str = "scaled"):
    """
    Automatic V1 keeps two reproducible profiles:
      plain  -> no scaling
      scaled -> StandardScaler on continuous numeric features only.

    RobustScaler/MinMaxScaler remain Guided choices until a separate,
    research-backed automatic scaler-selection policy is defined.
    """
    if profile == "plain":
        return None
    return StandardScaler()
