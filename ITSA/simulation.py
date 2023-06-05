"""
Simulation functions for interval time series
"""
import numpy as np
import pandas as pd
from scipy.stats import norm


def air(params, sigma, nsamples, nsimulations):
    """
    Simulate a time series following the Auto-Interval-Regressive Model.

    Args:
        params (list): list of autocorrelation parameters.
        sigma (float): Standard deviation of Gaussian noise, at each lag.
        nsamples (int): The number of random samples from Gaussian distribution, at each lag.
        nsimulations (int): The number of observations to simulate.

    Returns:
        DataFrame: A dataframe with interval time series data.
            the first column is ``Upper Bound``, the second column is ``Lower Bound``.

    """
    # * Since we use np.dot() to calculate simulated values,
    # * parameters must be reversed to generate correct results.
    params = np.array(params)[::-1]
    order = len(params)

    initail = np.random.normal(0, sigma, 2 * order).reshape(order, 2)
    initail_upper = initail.max(axis=1)
    initail_lower = initail.min(axis=1)

    simu_upper = initail_upper.tolist()
    simu_lower = initail_lower.tolist()

    # generate more samples to get stable results
    for i in range(nsimulations * 2):
        error = np.random.normal(0, sigma, nsamples)
        fitted_per_lag_u = np.array(simu_upper[i : order + i]).dot(params) + max(error)
        fitted_per_lag_l = np.array(simu_lower[i : order + i]).dot(params) + min(error)
        simu_upper.append(float(fitted_per_lag_u))
        simu_lower.append(float(fitted_per_lag_l))

    simu_data = {
        "Upper Bound": simu_upper[-nsimulations:],
        "Lower Bound": simu_lower[-nsimulations:],
    }

    return pd.DataFrame(simu_data)


def hvair(phi, beta, nsamples, nsimulations):
    """
    Simulate a time series following the Heteroscedastic Volatility AIR Model.

    Args:
        phi (list): list of autocorrelation parameters.
        beta (list): list of parameters used to compute conditional variance, at each lag.
        nsamples (int): The number of random samples from Gaussian distribution, at each lag.
        nsimulations (int): The number of observations to simulate.

    Returns:
        DataFrame: A dataframe with interval time series data.
            the first column is ``Upper Bound``, the second column is ``Lower Bound``.

    """
    # * Since we use np.dot() to calculate simulated values,
    # * parameters must be reversed to generate correct results.
    # TODO : Double check if order of beta is the same as HVAIR design.
    phi, beta = np.array(phi)[::-1], np.array(beta)[::-1]
    ar_order, arch_order = len(phi), len(beta) - 1
    minima_initial_num = max(ar_order, arch_order)
    alpha = 3 / 8

    initail = np.random.normal(
        scale=np.sqrt(beta[-1]), size=2 * minima_initial_num
    ).reshape(minima_initial_num, 2)
    initail_upper = initail.max(axis=1)
    initail_lower = initail.min(axis=1)

    simu_upper = initail_upper.tolist()
    simu_lower = initail_lower.tolist()
    for i in range(nsimulations * 2):
        gamma_sqr = (
            np.array(simu_upper[-arch_order:]) - np.array(simu_lower[-arch_order:])
        ) ** 2 / (
            norm.ppf((nsamples - alpha) / (nsamples - 2 * alpha + 1))
            - norm.ppf((1 - alpha) / (nsamples - 2 * alpha + 1))
        ) ** 2
        var = beta.dot(np.hstack((gamma_sqr, np.ones(1))))
        error = np.random.normal(0, np.sqrt(var), nsamples)

        fitted_per_lag_u = np.array(simu_upper[-ar_order:]).dot(phi) + max(error)
        fitted_per_lag_l = np.array(simu_lower[-ar_order:]).dot(phi) + min(error)
        simu_upper.append(float(fitted_per_lag_u))
        simu_lower.append(float(fitted_per_lag_l))

    simu_data = {
        "Upper Bound": simu_upper[-nsimulations:],
        "Lower Bound": simu_lower[-nsimulations:],
    }

    return pd.DataFrame(simu_data)


def ivma(params, sigma, nsamples, nsimulations):
    """
    Simulate a time series following the Interval-Valued Moving Averaging Model.

    Args:
        params (list): list of moving-average parameters.
        sigma (float): Standard deviation of Gaussian noise, at each lag.
        nsamples (int): The number of random samples from Gaussian distribution, at each lag.
        nsimulations (int): The number of observations to simulate.

    Returns:
        DataFrame: A dataframe with interval time series data.
            the first column is ``Upper Bound``, the second column is ``Lower Bound``.

    """
    # * Since we use np.dot() to calculate simulated values,
    # * parameters must be reversed to generate correct results.
    params = np.array(params)[::-1]
    order = len(params)

    initail = np.random.normal(0, sigma, (nsamples, order))
    initail_upper = initail.max(axis=0)
    initail_lower = initail.min(axis=0)

    errors_upper = initail_upper.tolist()
    errors_lower = initail_lower.tolist()
    simu_upper = []
    simu_lower = []
    for i in range(nsimulations * 2):
        error = np.random.normal(0, sigma, nsamples)
        fitted_per_lag_u = max(error) - np.array(errors_upper[i : order + i]).dot(
            params
        )
        fitted_per_lag_l = min(error) - np.array(errors_lower[i : order + i]).dot(
            params
        )
        simu_upper.append(float(fitted_per_lag_u))
        simu_lower.append(float(fitted_per_lag_l))
        errors_upper.append(max(error))
        errors_lower.append(min(error))

    simu_data = {
        "Upper Bound": simu_upper[-nsimulations:],
        "Lower Bound": simu_lower[-nsimulations:],
    }

    return pd.DataFrame(simu_data)


def airma(phi, theta, sigma, nsamples, nsimulations):
    """
    Simulate a time series following the Auto-Interval-Regressive Moving Averaging Model.

    Args:
        phi (list): list of autocorrelation parameters.
        theta (list): list of moving-average parameters.
        sigma (float): Standard deviation of Gaussian noise, at each lag.
        nsamples (int): The number of random samples from Gaussian distribution, at each lag.
        nsimulations (int): The number of observations to simulate.

    Returns:
        DataFrame: A dataframe with interval time series data.
            the first column is ``Upper Bound``, the second column is ``Lower Bound``.

    """
    # * Since we use np.dot() to calculate simulated values,
    # * parameters must be reversed to generate correct results.
    phis, thetas = np.array(phi)[::-1], np.array(theta)[::-1]
    ar_order, ma_order = len(phis), len(thetas)

    initail = np.random.normal(0, sigma, (ar_order, nsamples))
    simu_upper = initail.max(axis=1).tolist()
    simu_lower = initail.min(axis=1).tolist()

    initail_error = np.random.normal(0, sigma, (ma_order, nsamples))
    error_upper = initail_error.max(axis=1).tolist()
    error_lower = initail_error.min(axis=1).tolist()

    for i in range(nsimulations * 2):
        error = np.random.normal(0, sigma, nsamples)

        fitted_per_lag_u = (
            np.array(simu_upper[i : ar_order + i]).dot(phis)
            + max(error)
            - np.array(error_upper[i : ma_order + i]).dot(thetas)
        )
        fitted_per_lag_l = (
            np.array(simu_lower[i : ar_order + i]).dot(phis)
            + min(error)
            - np.array(error_lower[i : ma_order + i]).dot(thetas)
        )

        simu_upper.append(fitted_per_lag_u)
        simu_lower.append(fitted_per_lag_l)
        error_upper.append(max(error))
        error_lower.append(min(error))

    simu_data = {
        "Upper Bound": simu_upper[-nsimulations:],
        "Lower Bound": simu_lower[-nsimulations:],
    }

    return pd.DataFrame(simu_data)
