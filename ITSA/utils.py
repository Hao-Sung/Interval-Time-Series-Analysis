import pandas as pd
from sklearn.preprocessing import StandardScaler


def refactor_interval(df, standard=True):
    """restructure interval data into center-range format

    Args:
        df (pandas.DataFrame): dataframe with ``Upper Bound`` and ``Lower Bound`` columns
        standard (bool, optional): if True, then both "Range" and "Central" columns are standardized.
            Defaults to True.

    Returns:
        pandas.DataFrame: dataframe with ``Central`` and ``Range`` columns
    """
    res = pd.DataFrame()
    res["Range"] = df["Upper Bound"] - df["Lower Bound"]
    res["Central"] = (df["Upper Bound"] + df["Lower Bound"]) / 2
    if standard:
        res[["Range", "Central"]] = StandardScaler().fit_transform(
            res[["Range", "Central"]]
        )
    return res
