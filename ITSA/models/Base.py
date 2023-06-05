import os
import re
import copy
import warnings
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo

from texttable import Texttable

from ITSA.models.tools import redefine_interval

warnings.filterwarnings("ignore", message="divide by zero encountered in log")


class Base(object):
    """Base class for interval time series models"""

    def __init__(self, endog, n_sample):
        self.endog = endog
        self.n_sample = n_sample
        self.order = None

        self._n_obs = None
        self._date = None
        self._loglike = None
        self._params = None
        self._backcast = (None, None)

    @property
    def endog(self):
        """The observed interval time-series process"""
        return pd.DataFrame(
            {
                self._endog["Upper"]["col_name"]: self._endog["Upper"]["data"],
                self._endog["Lower"]["col_name"]: self._endog["Lower"]["data"],
            }
        )

    @endog.setter
    def endog(self, new_endog):
        if not isinstance(new_endog, pd.DataFrame):
            raise TypeError("endog must be a pandas DataFrame")
        else:
            # ? good idea to define other variable here?
            self._n_obs = new_endog.shape[0]
            self._date = new_endog.index
            self._endog = redefine_interval(new_endog)

    @property
    def aic(self):
        """Akaike information criterion value"""
        aic = 2 * len(self._params) - 2 * self._loglike
        return aic

    @property
    def bic(self):
        """Bayesian information criterion value"""
        bic = len(self._params) * np.log(self._n_obs) - 2 * self._loglike
        return bic

    @property
    def hqic(self):
        """Hannan-Quinn information criterion value"""
        hqic = -2 * self._loglike + 2 * len(self._params) * np.log(np.log(self._n_obs))
        return hqic

    @property
    def mde(self):
        """Mean distance error of fitted values"""
        fit_upper, fit_lower, raw_upper, raw_lower = self._get_comparison_data()

        diff_u = (raw_upper - fit_upper) ** 2
        diff_l = (raw_lower - fit_lower) ** 2
        mde = np.sqrt((diff_u + diff_l).sum() / (2 * self._n_obs))
        return mde

    @property
    def loglike(self):
        """Maximum log-likelihood value"""
        return self._loglike

    @property
    def params(self):
        """Model Parameters"""
        return self._params

    def _get_comparison_data(self):
        fitted_data = self.fitted_values()
        fit_upper = fitted_data["Upper Bound"].values
        fit_lower = fitted_data["Lower Bound"].values

        data = self._endog
        raw_upper = data["Upper"]["data"]
        raw_lower = data["Upper"]["data"]
        return (fit_upper, fit_lower, raw_upper, raw_lower)

    # will be defined in subclasses
    def fitted_values(self):
        pass

    def fitted_check(self, filename=None):
        """Line chart of actual data and fitted values

        Args:
            filename (str, optional): The local path and filename to save the outputted chart to.
            if filename is not set, figure will not be exported. Defaults to None.

        Returns:
            plotly Figure.
        """
        fitted_data = self.fitted_values()
        fit_upper = fitted_data["Upper Bound"].values
        fit_lower = fitted_data["Lower Bound"].values

        data = self._endog
        raw_upper = data["Upper"]["data"]
        raw_lower = data["Lower"]["data"]

        lags = self._date
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=lags,
                y=raw_upper,
                mode="lines+markers",
                line=dict(color="rgb(70,130,180)", width=3),
                name="True Upper Bound",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=lags,
                y=raw_lower,
                mode="lines+markers",
                line=dict(color="rgb(205,38,38)", width=3),
                name="True Lower Bound",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=lags,
                y=fit_upper,
                mode="lines+markers",
                line=dict(color="rgba(30,144,255,0.4)", width=3),
                name="Fitted Upper Bound",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=lags,
                y=fit_lower,
                mode="lines+markers",
                line=dict(color="rgba(255,165,0,0.4)", width=3),
                name="Fitted Lower Bound",
            )
        )
        fig.update_xaxes(rangeslider_visible=True)
        fig.update_layout(
            title="Model Fitting Result",
            xaxis_title="lags",
            yaxis_title="Trend",
            hovermode="x",
        )

        if filename:
            pyo.plot(fig, filename=f".{os.sep}figs{os.sep}{filename}")
        return fig

    def _summary_brief(self):
        vars_name = (
            "D."
            + self._endog["Upper"]["col_name"]
            + "\n"
            + "D."
            + self._endog["Lower"]["col_name"]
        )
        model_name = self.__class__.__name__
        model_order = (
            f"{self.order}" if isinstance(self.order, tuple) else f"({self.order})"
        )
        date_current = date.today().strftime("%b-%d-%Y")

        table = Texttable()
        table.set_deco(Texttable.HEADER)
        table.set_cols_dtype(["t", "a", "t", "a"])
        table.set_cols_align(["l", "r", "l", "r"])
        table.add_rows(
            [
                ["Dep. Variable: ", vars_name, "No. Observations: ", self._n_obs],
                [
                    "Model: ",
                    f"{model_name}{model_order}",
                    "Log Likelihood: ",
                    self.loglike,
                ],
                ["Method: ", "mle", "AIC: ", self.aic],
                ["Optimizer: ", "PSO", "BIC: ", self.bic],
                ["Date: ", date_current, "HQIC: ", self.hqic],
            ]
        )
        tbl = table.draw()
        # customize table with regex
        hline = re.search("=.+=\n", tbl).group()
        main = re.sub("=.+=\n", "", tbl)
        header = f"{model_name} Model Results".center(len(hline)) + "\n" + hline
        print(header + main)

    # will be defined in subclasses
    # todo: add "row" args...
    def _summary_coef(self, coef_row):
        table = Texttable()
        table.set_deco(Texttable.HEADER)
        table.set_cols_dtype(["t", "f"])
        table.set_cols_align(["l", "r"])

        dummy_row = ["model.model", 0.000]
        # ? is deepcopy necessary?
        rows = copy.deepcopy(coef_row)
        rows.insert(0, dummy_row)
        table.add_rows(rows)
        tbl = table.draw()
        # customize table with regex
        hline = re.search("=.+=", tbl).group()
        hline_thin = "-" * len(hline) + "\n"

        main = re.sub("=.+=\n", "", tbl)
        main = re.sub("model.+\n", "", main)
        header = hline + "\n" + "Coefficients".center(len(hline)) + "\n" + hline_thin

        print(header + main)
