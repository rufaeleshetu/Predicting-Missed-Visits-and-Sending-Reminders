from typing import List, Tuple, Union
import pandas as pd
from sklearn.model_selection import train_test_split

def split_features_target(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    return_xy: bool = False,
) -> Union[
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
]:
    """
    Splits df into train/test for features + target.

    Default return (4):
        X_train, X_test, y_train, y_test

    Optional return (6) if return_xy=True:
        X, y, X_train, X_test, y_train, y_test
    """
    X = df.loc[:, feature_cols].copy()
    y = df.loc[:, target_col].astype(int).copy()

    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat
    )

    if return_xy:
        return X, y, X_train, X_test, y_train, y_test

    return X_train, X_test, y_train, y_test
