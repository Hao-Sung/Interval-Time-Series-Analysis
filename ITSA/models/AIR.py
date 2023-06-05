import numpy as np
import pandas as pd

from scipy.stats import norm
from pyswarms.single.global_best import GlobalBestPSO

from ITSA.models.Base import Base
from ITSA.models.tools import data_rearrange, get_reverse_data
from ITSA.stattools import expect_min_max


# negative log-likelihood function
def _minus_logLikle_air(theta, df, ar_order, n_sample, n_particle):
    # compute integration limits and integration value
    def cdf_integration(data):
        matrix = data_rearrange(data, ar_order)
        # compute upper limit of integration
        lim_matrix = np.hstack((np.ones((n_particle, 1)), -theta[:, :-1])).dot(matrix)
        # reshape std
        scale_matrix = np.repeat(
            np.array([theta[:, -1]]).T, repeats=lim_matrix.shape[1], axis=1
        )
        cdf_matrix = norm.cdf(lim_matrix, scale=scale_matrix)
        return (cdf_matrix, lim_matrix)

    upper_cdf, upper_lim_matrix = cdf_integration(df["Upper"]["data"])
    lower_cdf, lower_lim_matrix = cdf_integration(df["Lower"]["data"])

    n_obs = len(df["Upper"]["data"])
    component1 = np.log(upper_cdf - lower_cdf).sum(axis=1) * (n_sample - 2)
    component2 = (n_obs - ar_order) * np.log(theta[:, -1] ** 2)
    component3 = (upper_lim_matrix**2 + lower_lim_matrix**2).sum(axis=1) / (
        2 * theta[:, -1] ** 2
    )

    log_liklihood = component1 - component2 - component3
    return -log_liklihood


class AIR(Base):
    """Auto-Interval-Regressive (AIR) Model

    Attributes:
        endog (pandas.DataFrame): The observed interval time-series process.
        order (int): The order of autoregressive model.
        n_sample (int): The number of samples at each lag.

    After `fit` method is performed, the following attributes
        are available:
        - aic (float): The Akaike Information Criterion.
        - bic (float): The Bayesian Information Criterion.
        - hqic (float): The Hannan-Quinn Information Criterion.
        - mde (float): The Mean Distance Error.
        - loglike (float): Maximum Log-Likelihood.
        - params (np.ndarray): The parameters of the model.

    """

    def __init__(self, endog, order, n_sample):
        """Initialize the model

        Args:
            endog (pandas.DataFrame): The observed interval time-series process.
            order (int): the order of autoregressive model.
            n_sample (int): The number of samples at each lag.
        """
        super().__init__(endog, n_sample)
        self.order = order
        self._n_obs = endog.shape[0]
        self._date = endog.index

    def __repr__(self):
        return f"Auto-Interval-Regressive Model(order={self.order}, n={self.n_sample})"

    def fit(self, n_particle=100, std_upper=5, iters=500):
        """Fit the model according to the given interval data.

        Args:
            n_particle (int, optional): Number of particles used in PSO. Defaults to 100.
            std_upper (int, optional):
                Upper Bound limiting candidate solutions of `std` to  the specified range.
                Defaults to 5.
            iters (int, optional): number of iterations. Defaults to 500.
        """
        params_upper_bound = np.hstack((np.ones(self.order), std_upper))
        params_lower_bound = np.hstack((-1 * np.ones(self.order), 0))
        bounds = (params_lower_bound, params_upper_bound)
        options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
        optimizer = GlobalBestPSO(
            n_particles=n_particle,
            dimensions=(self.order + 1),
            options=options,
            bounds=bounds,
        )

        # ? Backcast process added here?
        endog, endog_reverse = get_reverse_data(self._endog)
        kwargs = {
            "df": endog_reverse,
            "ar_order": self.order,
            "n_sample": self.n_sample,
            "n_particle": n_particle,
        }
        _, pos_back = optimizer.optimize(
            _minus_logLikle_air, iters=iters, verbose=False, **kwargs
        )
        self._params = pos_back
        self._endog = endog_reverse
        backcast = self.predict(self.order)
        self._backcast = (
            backcast["Upper Bound"].values[::-1],
            backcast["Lower Bound"].values[::-1],
        )
        # ? Actual fitting process added here?
        kwargs["df"] = endog
        cost, pos = optimizer.optimize(
            _minus_logLikle_air, iters=iters, verbose=False, **kwargs
        )
        self._endog = endog
        self._loglike = -float(cost)

        # * Parameters' order is same as specification in simulation process.
        # * (i.e., ar.L1 => ar.L2 => ar.L3 ...)
        self._params = pos

    def predict(self, step):
        """Compute minimum mean square error forecast of the model.

        Args:
            step (int): The number of observations to predict.

        Returns:
            pandas.DataFrame: Dataframe with ``"Upper Bound`` and ``"Lower Bound`` columns
        """
        n_sample = self.n_sample
        order = self.order

        std = self.params[-1]
        max_error, min_error = expect_min_max(n_sample, float(std))

        ts = self._endog
        upper = ts["Upper"]["data"][-order:]
        lower = ts["Lower"]["data"][-order:]
        ar_params = self.params[:-1][::-1]
        for i in range(step):
            pred_per_step_u = upper[-order:].dot(ar_params) + max_error
            pred_per_step_l = lower[-order:].dot(ar_params) + min_error

            upper = np.append(upper, pred_per_step_u)
            lower = np.append(lower, pred_per_step_l)

        return pd.DataFrame(
            {"Upper Bound": upper[-step:], "Lower Bound": lower[-step:]}
        )

    def fitted_values(self):
        """Compute interval fitted values of the model.

        Returns:
            pandas.DataFrame: Dataframe with ``"Upper Bound`` and ``"Lower Bound`` columns
        """
        n_obs = self._n_obs
        n_sample = self.n_sample
        order = self.order

        std = self.params[-1]
        max_error, min_error = expect_min_max(n_sample, float(std))

        back_upper, back_lower = self._backcast
        upper = np.append(back_upper, self._endog["Upper"]["data"])
        lower = np.append(back_lower, self._endog["Lower"]["data"])

        upper_fitted = np.array([])
        lower_fitted = np.array([])
        ar_params = self.params[:-1][::-1]
        for i in range(n_obs):
            fitted_per_lag_u = upper[i : order + i].dot(ar_params) + max_error
            fitted_per_lag_l = lower[i : order + i].dot(ar_params) + min_error
            upper_fitted = np.append(upper_fitted, fitted_per_lag_u)
            lower_fitted = np.append(lower_fitted, fitted_per_lag_l)

        return pd.DataFrame({"Upper Bound": upper_fitted, "Lower Bound": lower_fitted})

    def residuals(self):
        """Compute interval residuals of the model.

        Returns:
            pandas.DataFrame: Dataframe with ``"Upper Bound`` and ``"Lower Bound`` columns
        """
        n_sample = self.n_sample
        params = self.params

        fit_upper, fit_lower, raw_upper, raw_lower = self._get_comparison_data()
        max_error, min_error = expect_min_max(n_sample, float(params[-1]))

        resid_upper = raw_upper - (fit_upper - max_error)
        resid_lower = raw_lower - (fit_lower - min_error)

        return pd.DataFrame({"Upper Bound": resid_upper, "Lower Bound": resid_lower})

    def summary(self):
        """Model summary table"""
        ar_params = self.params[:-1]
        std = self.params[-1]

        rows = [[f"ar.L{i+1}", ar_params[i]] for i in range(len(ar_params))]
        std_row = ["sigma", std]
        rows.extend(std_row)

        self._summary_brief()
        print("\n")
        self._summary_coef(rows)
