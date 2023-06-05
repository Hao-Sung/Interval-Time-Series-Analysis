import numpy as np
import pandas as pd

from scipy.stats import norm
from pyswarms.single.global_best import GlobalBestPSO

from ITSA.models.Base import Base
from ITSA.models.tools import get_reverse_data
from ITSA.stattools import expect_min_max


def _minus_logLikle_ivma(theta, df, ma_order, n_sample, n_particle):
    # obtain error series recursively
    def error_recursion_ivma():
        max_error, min_error = expect_min_max(n_sample, 1)
        expect_upper = np.array([[max_error] * ma_order])
        expect_lower = np.array([[min_error] * ma_order])
        # generate initial error
        error_upper = theta[:, -1][:, np.newaxis] * np.repeat(
            expect_upper, n_particle, axis=0
        )
        error_lower = theta[:, -1][:, np.newaxis] * np.repeat(
            expect_lower, n_particle, axis=0
        )

        upper = df["Upper"]["data"]
        lower = df["Lower"]["data"]
        for i in range(len(upper)):
            rec_upper = upper[i] + (theta[:, :-1] * error_upper[:, -ma_order:]).sum(
                axis=1
            )
            rec_lower = lower[i] + (theta[:, :-1] * error_lower[:, -ma_order:]).sum(
                axis=1
            )
            error_upper = np.hstack((error_upper, rec_upper[:, np.newaxis]))
            error_lower = np.hstack((error_lower, rec_lower[:, np.newaxis]))

        return (error_upper[:, ma_order:], error_lower[:, ma_order:])

    error_upper, error_lower = error_recursion_ivma()
    # * For more accurate estimation, drop some errors which might be
    # * greatly effected by initial values
    n_obs = df["Upper"]["data"].shape[0]
    drop_num = min(5 * ma_order, n_obs / 3)
    error_upper = error_upper[:, int(drop_num) :]
    error_lower = error_lower[:, int(drop_num) :]

    error_col_num = error_upper.shape[1]
    scale_matrix = np.repeat(np.array([theta[:, -1]]).T, repeats=error_col_num, axis=1)
    upper_cdf = norm.cdf(error_upper, scale=scale_matrix)
    upper_pdf = norm.pdf(error_upper, scale=scale_matrix)
    lower_cdf = norm.cdf(error_lower, scale=scale_matrix)
    lower_pdf = norm.pdf(error_lower, scale=scale_matrix)

    log_liklihood = np.log(
        n_sample
        * (n_sample - 1)
        * np.power(upper_cdf - lower_cdf, n_sample - 2)
        * upper_pdf
        * lower_pdf
    ).sum(axis=1)

    return -log_liklihood


class IVMA(Base):
    """Interval-Valued Moving Averaging (IVMA) Model

    Attributes:
        endog (pandas.DataFrame): The observed interval time-series process.
        order (int): the length of MA lags.
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
            order (int): the length of MA lags.
            n_sample (int): The number of samples at each lag.
        """
        super().__init__(endog, n_sample)
        self.order = order
        self._n_obs = endog.shape[0]
        self._date = endog.index
        self.init_pos = None

    def __repr__(self):
        return f"Interval-Valued Moving Averaging Model(order={self.order}, n={self._n_obs})"

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
            init_pos=self.init_pos,
        )

        # ? Backcast process added here?
        endog, endog_reverse = get_reverse_data(self._endog)
        kwargs = {
            "df": endog_reverse,
            "ma_order": self.order,
            "n_sample": self.n_sample,
            "n_particle": n_particle,
        }
        _, pos_back = optimizer.optimize(
            _minus_logLikle_ivma, iters=iters, verbose=False, **kwargs
        )
        self._params = np.append(pos_back[:-1][::-1], pos_back[-1])
        self._endog = endog_reverse
        backcast = self.predict(self.order)
        self._backcast = (
            backcast["Upper Bound"].values[::-1],
            backcast["Lower Bound"].values[::-1],
        )

        # ? Actual fitting process added here?
        kwargs["df"] = endog
        cost, pos = optimizer.optimize(
            _minus_logLikle_ivma, iters=iters, verbose=False, **kwargs
        )
        self._endog = endog
        self._loglike = -float(cost)

        # * After reversed, parameters' order is same as
        # * specification in simulation process.
        # * (i.e., ma.L1 => ma.L2 => ma.L3 ...)
        self._params = np.append(pos[:-1][::-1], pos[-1])

    def predict(self, step):
        """Compute minimum mean square error forecast of the model.

        Args:
            step (int): The number of observations to predict.

        Returns:
            pandas.DataFrame: Dataframe with ``"Upper Bound`` and ``"Lower Bound`` columns
        """
        n_sample = self.n_sample
        ma_order = self.order

        std_param = self.params[-1]
        ma_params = self.params[:-1][::-1]

        max_error, min_error = expect_min_max(n_sample, float(std_param))

        _, resids = self._fits_resids()
        resid_upper = resids["Upper Bound"].values[-ma_order:]
        resid_lower = resids["Lower Bound"].values[-ma_order:]

        pred_upper = np.array([])
        pred_lower = np.array([])
        for i in range(step):
            pred_per_lag_u = max_error - resid_upper[-ma_order:].dot(ma_params)
            pred_per_lag_l = min_error - resid_lower[-ma_order:].dot(ma_params)

            resid_upper = np.append(resid_upper, max_error)
            resid_lower = np.append(resid_lower, min_error)

            pred_upper = np.append(pred_upper, pred_per_lag_u)
            pred_lower = np.append(pred_lower, pred_per_lag_l)

        return pd.DataFrame({"Upper Bound": pred_upper, "Lower Bound": pred_lower})

    def _fits_resids(self):
        """compute fitted values and residuals simultaneously"""
        n_obs = self._n_obs
        n_sample = self.n_sample
        ma_order = self.order

        std_param = self.params[-1]
        ma_params = self.params[:-1][::-1]

        max_error, min_error = expect_min_max(n_sample, float(std_param))

        resid_upper = np.repeat(max_error, ma_order)
        resid_lower = np.repeat(min_error, ma_order)
        upper_fitted = np.array([])
        lower_fitted = np.array([])

        back_upper, back_lower = self._backcast
        upper = np.append(back_upper, self._endog["Upper"]["data"])
        upper = upper[upper != None]
        lower = np.append(back_lower, self._endog["Lower"]["data"])
        lower = lower[lower != None]

        # * Since backcast process also needs residuals,
        # * if "_backcast" property is still None, "rec_time" should be adjusted.
        rec_time = n_obs if back_upper is None else n_obs + ma_order
        for i in range(rec_time):
            fitted_per_lag_u = max_error - resid_upper[-ma_order:].dot(ma_params)
            fitted_per_lag_l = min_error - resid_lower[-ma_order:].dot(ma_params)
            upper_fitted = np.append(upper_fitted, fitted_per_lag_u)
            lower_fitted = np.append(lower_fitted, fitted_per_lag_l)

            resid_per_lag_u = upper[i] + resid_upper[-ma_order:].dot(ma_params)
            resid_per_lag_l = lower[i] + resid_lower[-ma_order:].dot(ma_params)
            resid_upper = np.append(resid_upper, resid_per_lag_u)
            resid_lower = np.append(resid_lower, resid_per_lag_l)

        fitted_data = pd.DataFrame(
            {"Upper Bound": upper_fitted[-n_obs:], "Lower Bound": lower_fitted[-n_obs:]}
        )
        resid_data = pd.DataFrame(
            {"Upper Bound": resid_upper[-n_obs:], "Lower Bound": resid_lower[-n_obs:]}
        )
        return (fitted_data, resid_data)

    def fitted_values(self):
        """Compute interval fitted values of the model.

        Returns:
            pandas.DataFrame: Dataframe with ``"Upper Bound`` and ``"Lower Bound`` columns
        """
        fitted_data, _ = self._fits_resids()
        return fitted_data

    def residuals(self):
        """Compute interval residuals of the model.

        Returns:
            pandas.DataFrame: Dataframe with ``"Upper Bound`` and ``"Lower Bound`` columns
        """
        _, resid_data = self._fits_resids()
        return resid_data

    def summary(self):
        """Model summary table"""
        ma_params = self.params[:-1][::-1]
        std_params = self.params[-1]

        rows = [[f"ma.L{i+1}", ma_params[i]] for i in range(len(ma_params))]
        std_row = ["sigma", std_params]
        rows.insert(len(rows), std_row)

        self._summary_brief()
        print("\n")
        self._summary_coef(rows)
