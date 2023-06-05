import numpy as np
import pandas as pd

from scipy.stats import norm
from pyswarms.single.global_best import GlobalBestPSO

from ITSA.models.Base import Base
from ITSA.models.tools import data_rearrange, get_reverse_data
from ITSA.stattools import expect_min_max


def arch_data_preproces(data, arch_order, n_sample):
    upper = data["Upper"]["data"]
    lower = data["Lower"]["data"]

    # the difference between max and min order statistics is same as max*2
    min_max_diff = expect_min_max(n_sample, 1)[0] * 2
    gamma_sqr = (upper - lower) ** 2 / (min_max_diff) ** 2
    gamma_sqr_matrix = data_rearrange(gamma_sqr, arch_order)
    res = np.vstack((np.ones((1, gamma_sqr_matrix.shape[1])), gamma_sqr_matrix[1:, :]))
    return res


def _minus_logLikle_hvair(theta, df, order, n_sample, n_particle):

    # compute integration limits and integration value
    def cdf_integration(data, stochastic_var):
        matrix = ar_data_preprocess(data, ar_order)
        # compute upper limit of integration
        lim_matrix = np.hstack((np.ones((n_particle, 1)), -ar_params)).dot(matrix)
        # modify dimension of two matrix to make them the same
        if ar_order > arch_order:
            stochastic_var = stochastic_var[:, (ar_order - arch_order) :]
        if ar_order < arch_order:
            lim_matrix = lim_matrix[:, -(ar_order - arch_order) :]

        cdf_matrix = norm.cdf(lim_matrix, scale=np.sqrt(stochastic_var))
        return (cdf_matrix, lim_matrix)

    ar_order = order[0]
    arch_order = order[1]

    ar_params = theta[:, :ar_order]
    arch_params = theta[:, ar_order:]

    gamma_sqr = arch_data_preproces(df, arch_order, n_sample)
    stochastic_var = arch_params.dot(gamma_sqr)

    upper_cdf, upper_lim_matrix = cdf_integration(df["Upper"]["data"], stochastic_var)
    lower_cdf, lower_lim_matrix = cdf_integration(df["Lower"]["data"], stochastic_var)

    component1 = np.log(upper_cdf - lower_cdf).sum(axis=1) * (n_sample - 2)
    component2 = np.log(stochastic_var).sum(axis=1)
    component3 = (
        (upper_lim_matrix**2 + lower_lim_matrix**2) / (2 * stochastic_var)
    ).sum(axis=1)

    log_liklihood = component1 - component2 - component3
    return -log_liklihood


class HVAIR(Base):
    """Heteroscedastic Volatility Auto-Interval-Regressive (HVAIR) Model

    Attributes:
        endog (pandas.DataFrame): The observed interval time-series process.
        order (tuple): The order of HVAIR model:
            - The first element is the length of AR lags.
            - The second element is the length of ARCH lags.
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
            order (tuple): The order of HVAIR model:
                - The first element is the length of AR lags.
                - The second element is the length of ARCH lags.
            n_sample (int): The number of samples at each lag.
        """
        super().__init__(endog, n_sample)
        self.order = order
        self._n_obs = endog.shape[0]
        self._date = endog.index

    def __repr__(self):
        return (
            f"Heteroscedastic Volatility AIR Model(order={self.order}, n={self._n_obs})"
        )

    def fit(self, n_particle=100, iters=500):
        """Fit the model according to the given interval data.

        Args:
            n_particle (int, optional): Number of particles used in PSO. Defaults to 100.
            iters (int, optional): number of iterations. Defaults to 500.
        """
        ar_order = self.order[0]
        arch_order = self.order[1]

        params_upper_bound = np.ones(ar_order + (arch_order + 1))
        params_lower_bound = np.hstack(
            (-1 * np.ones(ar_order), np.zeros(arch_order + 1))
        )
        bounds = (params_lower_bound, params_upper_bound)
        options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
        optimizer = GlobalBestPSO(
            n_particles=n_particle,
            dimensions=ar_order + (arch_order + 1),
            options=options,
            bounds=bounds,
        )

        # ? Backcast process added here?
        endog, endog_reverse = get_reverse_data(self._endog)
        kwargs = {
            "df": endog_reverse,
            "order": self.order,
            "n_sample": self.n_sample,
            "n_particle": n_particle,
        }
        _, pos_back = optimizer.optimize(
            _minus_logLikle_hvair, iters=iters, verbose=False, **kwargs
        )
        self._params = pos_back
        self._endog = endog_reverse
        backcast = self.predict(max(self.order))
        self._backcast = (
            backcast["Upper Bound"].values[::-1],
            backcast["Lower Bound"].values[::-1],
        )

        # ? Actual fitting process added here?
        kwargs["df"] = endog
        cost, pos = optimizer.optimize(
            _minus_logLikle_hvair, iters=iters, verbose=False, **kwargs
        )
        self._endog = endog
        self._loglike = -float(cost)

        # * Parameters' order is same as specification in simulation process.
        # * (i.e., ar.L1 => ar.L2 => ar.L3 ...,
        # *        arch.const => arch.L1 => arch.L2 => arch.L3...)
        self._params = pos

    def predict(self, step):
        """Compute minimum mean square error forecast of the model.

        Args:
            step (int): The number of observations to predict.

        Returns:
            pandas.DataFrame: Dataframe with ``"Upper Bound`` and ``"Lower Bound`` columns
        """
        n_sample = self.n_sample
        ar_order = self.order[0]
        arch_order = self.order[1]
        ar_params = self.params[:ar_order][::-1]
        arch_params = self.params[ar_order:]

        ts = self._endog
        max_order = max(ar_order, arch_order)
        upper = ts["Upper"]["data"][-max_order:]
        lower = ts["Lower"]["data"][-max_order:]

        # the difference between max and min order statistics is same as max*2
        min_max_diff = expect_min_max(n_sample, 1)[0] * 2
        for i in range(step):
            gamma_sqr = (upper[-arch_order:] - lower[-arch_order:]) ** 2 / (
                min_max_diff
            ) ** 2
            stochastic_std = np.sqrt(
                arch_params.dot(np.hstack((np.ones(1), gamma_sqr)))
            )
            max_error, min_error = expect_min_max(n_sample, stochastic_std)

            pred_per_step_u = upper[-ar_order:].dot(ar_params) + max_error
            pred_per_step_l = lower[-ar_order:].dot(ar_params) + min_error

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
        n_sample = self.n_sample
        ar_order = self.order[0]
        arch_order = self.order[1]
        ar_params = self.params[:ar_order][::-1]
        arch_params = self.params[ar_order:]

        back_upper, back_lower = self._backcast
        upper = np.append(back_upper, self._endog["Upper"]["data"])
        lower = np.append(back_lower, self._endog["Lower"]["data"])

        upper_fitted = np.array([])
        lower_fitted = np.array([])

        # * Because number of backcast values is based on maximum ``order``,
        # * index must be adjusted when computing variance and fitted interval value.
        max_order = max(ar_order, arch_order)
        var_idx_adj = abs(arch_order - max_order)
        ar_idx_adj = abs(ar_order - max_order)

        # the difference between max and min order statistics is same as max*2
        min_max_diff = expect_min_max(n_sample, 1)[0] * 2
        for i in range(self._n_obs):
            gamma_sqr = (
                upper[var_idx_adj + i : max_order + i]
                - lower[var_idx_adj + i : max_order + i]
            ) ** 2 / (min_max_diff) ** 2
            stochastic_std = np.sqrt(
                arch_params.dot(np.hstack((np.ones(1), gamma_sqr)))
            )
            max_error, min_error = expect_min_max(n_sample, stochastic_std)

            fitted_per_lag_u = (
                upper[ar_idx_adj + i : max_order + i].dot(ar_params) + max_error
            )
            fitted_per_lag_l = (
                lower[ar_idx_adj + i : max_order + i].dot(ar_params) + min_error
            )
            upper_fitted = np.append(upper_fitted, fitted_per_lag_u)
            lower_fitted = np.append(lower_fitted, fitted_per_lag_l)

        return pd.DataFrame({"Upper Bound": upper_fitted, "Lower Bound": lower_fitted})

    def residuals(self):
        """Compute interval residuals of the model.

        Returns:
            pandas.DataFrame: Dataframe with ``"Upper Bound`` and ``"Lower Bound`` columns
        """
        ar_order = self.order[0]
        ar_params = self.params[:ar_order][::-1]

        back_upper, back_lower = self._backcast
        upper = np.append(back_upper, self._endog["Upper"]["data"])
        lower = np.append(back_lower, self._endog["Lower"]["data"])

        ar_effect_upper = np.array([])
        ar_effect_lower = np.array([])

        max_order = max(self.order)
        ar_idx_adj = abs(ar_order - max_order)
        for i in range(self._n_obs):
            ar_upper = upper[ar_idx_adj + i : max_order + i].dot(ar_params)
            ar_lower = lower[ar_idx_adj + i : max_order + i].dot(ar_params)
            ar_effect_upper = np.append(ar_effect_upper, ar_upper)
            ar_effect_lower = np.append(ar_effect_lower, ar_lower)

        resid_upper = self._endog["Upper"]["data"] - ar_effect_upper
        resid_lower = self._endog["Lower"]["data"] - ar_effect_lower

        return pd.DataFrame({"Upper Bound": resid_upper, "Lower Bound": resid_lower})

    def summary(self):
        """Model summary table"""
        ar_order = self.order[0]
        ar_params = self.params[:ar_order][::-1]
        arch_params = self.params[ar_order:]

        rows = [[f"ar.L{i+1}", ar_params[i]] for i in range(len(ar_params))]
        rows_arch = [
            [f"arch.{f'L{i}' if i != 0 else 'const'}", arch_params[i]]
            for i in range(len(arch_params))
        ]

        rows.extend(rows_arch)

        self._summary_brief()
        print("\n")
        self._summary_coef(rows)

    # TODO: compute heteroscedastic volatility
    # @property
    # def volatility(self):
