import re
from math import pi

import pandas as pd
import numpy as np
from scipy.stats import norm
from texttable import Texttable

from ITSA.utils import refactor_interval


def _generate_decay_init_pos(n_particle, std_upper, n_lag):
    limits = np.linspace(0.1, 0.9, n_lag)
    initial = [np.random.uniform(-l, l, n_particle) for l in limits]
    initial = np.array(initial).T

    sigma_initial = np.random.uniform(0, std_upper, (n_particle, 1))
    initial = np.hstack((initial, sigma_initial))
    return initial


def acf(x, n_sample, std_upper=5, n_lag=10):
    """Calculate the autocorrelation function.

    Args:
        x (pandas.DataFram): The interval time-series process.
        n_sample (int): The number of samples at each lag.
        std_upper (int, optional): Upper Bound limiting candidate solutions of `std` to  the specified range.Defaults to 5.
            This argument is needed because we use estimate parameters of IVMA to calculate acf values.
        n_lag (int, optional): Number of lags to return autocorrelation for. Defaults to 10.

    Returns:
        numpy.ndarray: The autocorrelation function for lags 0, 1, …, n_lag.
    """
    # avoid "circular import" problem
    from ITSA.models.IVMA import IVMA

    # * Because IVMA might invoke ValueError when the length of MA lags is too large,
    # * we use while loop to ensure that fitting process is completed.
    while True:
        try:
            order = n_lag + 2
            # * Generate reasonable initial positions for PSO to make
            # * fitting processe more smooth.
            init_pos = _generate_decay_init_pos(100, std_upper, order)
            model = IVMA(x, order, n_sample)
            model.init_pos = init_pos
            model.fit(std_upper=std_upper, iters=2000)
            params = np.append(1, -1 * model.params[:-1])
            break
        except ValueError:
            pass

    variance = np.power(params, 2).sum()
    covariance = [params[:-i].dot(params[i:]) for i in range(1, n_lag + 1)]
    acf_values = np.append(1, np.array(covariance) / variance)

    return acf_values


def pacf(x, n_sample, std_upper, n_lag=10):
    """Calculate the partial autocorrelation function.

    Args:
        x (pandas.DataFram): The interval time-series process.
        n_sample (int): The number of samples at each lag.
        std_upper (int, optional): Upper Bound limiting candidate solutions of `std` to  the specified range.Defaults to 5.
            This argument is needed because we use estimate parameters of IVMA to calculate acf values.
        n_lag (int, optional): Number of lags to return autocorrelation for. Defaults to 10.

    Returns:
        numpy.ndarray: The partial autocorrelation for lags 0, 1, …, n_lag.
    """
    acf_values = acf(x, n_sample, std_upper, n_lag)[1:]

    # initialization
    n_lag = acf_values.shape[0]
    pacf_matrix = np.zeros((n_lag, n_lag))
    pacf_matrix[0, 0] = acf_values[0]

    # * apply Durbin-Levinson algorithm
    for k in np.arange(n_lag - 1):
        acf_k = acf_values[k + 1]

        phi_array = np.array([])
        for j in range(k + 1):
            if j != k:
                phi_kj = (
                    pacf_matrix[k - 1, j]
                    - pacf_matrix[k, k] * pacf_matrix[(k - 1), k - j - 1]
                )
                pacf_matrix[k, j] = phi_kj
                phi_array = np.append(phi_array, phi_kj)
            if j == k:
                phi_kk = pacf_matrix[k, k]
                phi_array = np.append(phi_array, phi_kk)

        num_adj = phi_array.dot(acf_values[: k + 1][::-1])
        den_adj = phi_array.dot(acf_values[: k + 1])
        pacf_kk = (acf_k - num_adj) / (1 - den_adj)
        pacf_matrix[k + 1, k + 1] = pacf_kk

    pacf_values = np.array([pacf_matrix[i, i] for i in range(n_lag)])
    pacf_values = np.append(1, pacf_values)
    return pacf_values


def _projection(vec, dir_vec):
    # reshape vector to column vector
    vec = np.array(vec).reshape(2, 1)
    dir_vec = np.array(dir_vec).reshape(2, 1)
    # following projection formula
    scalar = np.dot(dir_vec.transpose(), vec) / np.dot(dir_vec.transpose(), dir_vec)
    projection = scalar * dir_vec
    projection_length = np.linalg.norm(projection)
    return projection_length


def _interval_correlation_tbl(horizontal, vertical, diagonal, off_diagonal):
    expect = 2 / np.sqrt(pi)

    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_cols_dtype(["t", "a", "a", "a", "a"])
    table.set_cols_align(["c"] * 5)
    table.add_rows(
        [
            ["Term", "Horizontal", "Vertical", "Diagonal", "Off_Diagonal"],
            ["Expected", expect, expect, expect, expect],
            ["Data", horizontal, vertical, diagonal, off_diagonal],
            ["Related corr", "rho_cc", "rho_rr", "cross_corr", "cross_corr"],
        ]
    )

    tbl = table.draw()
    hline = re.search("=.+=", tbl).group()
    insert_index = (re.search("Related corr", tbl).span())[0]

    header = "Statistic Details".center(len(hline)) + "\n" + hline + "\n"
    main = tbl[:insert_index]
    fotter = hline + "\n" + tbl[insert_index:]
    tbl = header + main + fotter

    print("\n")
    print(tbl)
    print("\n")


def interval_correlation(df1, df2, refactor=True, detail=True):
    """compute interval correlation, based on segment projection.

    Args:
        df1 (pandas.DataFrame): A dataframe with interval data.
        df2 (pandas.DataFrame): A dataframe with interval data.
        refactor (bool, optional):
            if True, interval data are restructure into center-range format. Defaults to True.
        detail (bool, optional):
            if True, summary table is displayed in the console. Defaults to True.

    Returns:
        list: list of numbers representing average segment projections on
            horizontal axis, vertical axis, diagonal line and off-diagonal line

    """
    if refactor:
        df1 = refactor_interval(df1)
        df2 = refactor_interval(df2)

    diff = pd.DataFrame()
    diff["delta_x"] = df1["Central"] - df2["Central"]
    diff["delta_y"] = df1["Range"] - df2["Range"]

    dir_vec = {
        "horizontal": [1, 0],
        "vertical": [0, 1],
        "diagonal": [1, -1],
        "off-diagonal": [1, 1],
    }
    res = []
    for _, dir_vec in dir_vec.items():
        proj_series = diff.apply(_projection, axis=1, dir_vec=dir_vec)
        res.append(proj_series.mean())

    detail and _interval_correlation_tbl(*res)

    return res


def six_pairwise_correlation(df1, df2):
    """compute the six pairwise correlations between two interval data

    Args:
        df1 (pandas.DataFrame): dataframe with ``Upper Bound`` and ``Lower Bound`` columns
        df2 (pandas.DataFrame): dataframe with ``Upper Bound`` and ``Lower Bound`` columns

    Returns:
        tuple: six pairwise correlations
    """
    df1 = refactor_interval(df1, standard=False)
    df2 = refactor_interval(df2, standard=False)

    coor_central1_to_central2 = np.corrcoef(df1["Central"], df2["Central"])[1, 0]
    coor_range1_to_range2 = np.corrcoef(df1["Range"], df2["Range"])[1, 0]

    coor_central1_range2 = np.corrcoef(df1["Central"], df2["Range"])[1, 0]
    coor_range1_central2 = np.corrcoef(df1["Range"], df2["Central"])[1, 0]

    coor_self_1 = np.corrcoef(df1["Central"], df1["Range"])[1, 0]
    coor_self_2 = np.corrcoef(df2["Range"], df2["Central"])[1, 0]

    return (
        coor_central1_to_central2,
        coor_range1_to_range2,
        coor_central1_range2,
        coor_range1_central2,
        coor_self_1,
        coor_self_2,
    )


def expect_min_max(n_sample, std):
    """compute approximate expectation of
        largest order statistic and smallest order statistic
        from normal distribution.
    Args:
        nsample (int): Number of samples from normal distribution.
        std (float): standard deviation of normal distribution.

    Returns:
        tuple: maximum and minimum values.
    """
    alpha = 3 / 8
    max_error = norm.ppf((n_sample - alpha) / (n_sample - 2 * alpha + 1)) * std
    min_error = norm.ppf((1 - alpha) / (n_sample - 2 * alpha + 1)) * std
    return (max_error, min_error)
