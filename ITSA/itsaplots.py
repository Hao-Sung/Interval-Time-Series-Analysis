import os
from math import pi
from scipy.stats import norm

import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo
import matplotlib.pyplot as plt

from ITSA.utils import refactor_interval
from ITSA.stattools import interval_correlation, acf, pacf


def _autocorr_plot_layout(corr_series, corr_type, ci, title):
    n_lag = corr_series.shape[0]
    ci_upper = ci[0]
    ci_lower = ci[1]

    plt.figure(figsize=(10, 5))
    plt.scatter(x=range(0, n_lag), y=corr_series, c="#104E8B")
    plt.vlines(
        x=range(0, n_lag), ymin=0, ymax=corr_series, color="#4F94CD", linewidth=2.5
    )
    plt.axhline(y=0, color="#000000", alpha=0.8)
    plt.axhline(y=ci_upper, color="#8B0000", linestyle="dashed")
    plt.axhline(y=ci_lower, color="#8B0000", linestyle="dashed")

    ymin = ci_lower - 0.2 if min(corr_series) > ci_lower else min(corr_series) - 0.2
    plt.ylim(ymin=ymin, ymax=1.1)

    # Add rounded acf values above the dots
    for i in range(1, n_lag):
        if corr_series[i] > 0:
            plt.text(i - 0.25, corr_series[i] + 0.1, f"{round(corr_series[i],3)}")
        else:
            plt.text(i - 0.25, corr_series[i] - 0.1, f"{round(corr_series[i],3)}")

    plt.title(title, fontsize=15)
    plt.xlabel("lag", fontsize=12)
    plt.ylabel(corr_type, fontsize=12)
    plt.show()


def plot_acf(
    x, n_sample, std_upper=5, n_lag=10, alpha=0.05, title="Autocorrelation Function"
):
    """Plot the autocorrelation function

    Args:
        x (pandas.DataFram): The interval time-series process.
        n_sample (int): The number of samples at each lag.
        std_upper (int, optional): Upper Bound limiting candidate solutions of `std` to  the specified range. Defaults to 5.
            This argument is needed because we use estimate parameters of IVMA to calculate acf values.
        n_lag (int, optional): Number of lags to return autocorrelation for. Defaults to 10.
        alpha (float, optional): significance level used to compute the confidence level. Defaults to 0.05.
        title (str, optional): Plot title. Defaults to "Autocorrelation Function".
    """
    acf_values = acf(x, n_sample, std_upper, n_lag)

    n_obs = x.shape[0]
    ci_upper = norm.ppf(1 - alpha / 2) / np.sqrt(n_obs)
    ci_lower = norm.ppf(alpha / 2) / np.sqrt(n_obs)
    ci = (ci_upper, ci_lower)

    _autocorr_plot_layout(acf_values, "ACF", ci, title)


def plot_pacf(
    x,
    n_sample,
    std_upper=5,
    n_lag=10,
    alpha=0.05,
    title="Partial Autocorrelation Function",
):
    """Plot the partail autocorrelation function

    Args:
        x (pandas.DataFram): The interval time-series process.
        n_sample (int): The number of samples at each lag.
        std_upper (int, optional): Upper Bound limiting candidate solutions of `std` to  the specified range. Defaults to 5.
            This argument is needed because we use estimate parameters of IVMA to calculate acf values.
        n_lag (int, optional): Number of lags to return autocorrelation for. Defaults to 10.
        alpha (float, optional): significance level used to compute the confidence level. Defaults to 0.05.
        title (str, optional): Plot title. Defaults to "Partial Autocorrelation Function".
    """
    pacf_values = pacf(x, n_sample, std_upper, n_lag)

    n_obs = x.shape[0]
    ci_upper = norm.ppf(1 - alpha / 2) / np.sqrt(n_obs)
    ci_lower = norm.ppf(alpha / 2) / np.sqrt(n_obs)
    ci = (ci_upper, ci_lower)

    _autocorr_plot_layout(pacf_values, "PACF", ci, title)


def _add_guiding_lines(fig, limit):
    fig.add_trace(
        go.Scatter(
            x=[-limit, limit],
            y=[limit, -limit],
            mode="lines",
            line=dict(color="rgba(0,0,255,0.2)", dash="dot", width=2),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[limit, -limit],
            y=[limit, -limit],
            mode="lines",
            line=dict(color="rgba(0,0,255,0.2)", dash="dot", width=2),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[limit, -limit],
            y=[0, 0],
            mode="lines",
            line=dict(color="rgba(0,0,255,0.2)", dash="dot", width=2),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 0],
            y=[limit, -limit],
            mode="lines",
            line=dict(color="rgba(0,0,255,0.2)", dash="dot", width=2),
            showlegend=False,
        )
    )
    return fig


def dandelion(
    df1,
    df2,
    name1,
    name2,
    central="former",
    filename=None,
):
    """plot dandelion graph.

    Args:
        df1 (pandas.DataFrame): dataframe with ``Upper Bound`` and ``Lower Bound`` columns
        df2 (pandas.DataFrame): dataframe with ``Upper Bound`` and ``Lower Bound`` columns
        name1 (str): name of first dataframe
        name2 (str): name of second dataframe
        central (str, optional): dataset which is moved to center of the graph. Defaults to "former". Possible values:
            - "former": the first dataframe is chosen.
            - "latter": the second dataframe is chosen.
        filename (str, optional): The local path and filename to save the outputted chart to.
            if filename is not set, figure will not be exported. Defaults to None.

    Returns:
        plotly Figure.
    """
    df1 = refactor_interval(df1)
    df2 = refactor_interval(df2)
    # todo: check if df1 and df2 have the same number of rows

    central_df = df1 if central == "former" else df2
    non_central_df = df2 if central == "former" else df1

    point_origin = go.Scatter(
        x=[0],
        y=[0],
        name=name1,
        mode="markers",
        marker=dict(
            color="rgba(30,144,255,0.8)",
            size=8,
            line=dict(color="rgb(28,28,28)", width=2),
        ),
    )
    points_dandelion = go.Scatter(
        x=non_central_df["Central"] - central_df["Central"],
        y=non_central_df["Range"] - central_df["Range"],
        name=name2,
        mode="markers",
        marker=dict(color="green", size=8, line=dict(color="rgb(28,28,28)", width=2)),
    )
    shapes = [
        dict(
            type="line",
            x0=0,
            y0=0,
            x1=non_central_df["Central"].loc[i] - central_df["Central"].loc[i],
            y1=non_central_df["Range"].loc[i] - central_df["Range"].loc[i],
            line=dict(color="rgba(112,128,114,0.25)", width=2),
        )
        for i in range(df1.shape[0])
    ]

    h_proj, v_proj, d_proj, o_proj = interval_correlation(df1, df2, refactor=False, detail=False)

    c_lim = max(map(lambda dictionary: dictionary["x1"], shapes))
    r_lim = max(map(lambda dictionary: dictionary["y1"], shapes))
    lim = max(c_lim, r_lim)
    ranges = [-(lim + 0.5), lim + 0.5]

    layout = go.Layout(
        shapes=shapes,
        title="Dandelion Graph" + f" << {name1} and {name2} >>",
        xaxis_title="Central",
        yaxis_title="Range",
    )
    fig = go.Figure([point_origin, points_dandelion], layout)
    fig.update_xaxes(range=ranges)
    fig.update_yaxes(range=ranges)

    exp = 2 / np.sqrt(pi)
    fig.add_trace(
        go.Scatter(
            x=[-exp, -exp, exp, exp, -exp],
            y=[-exp, exp, exp, -exp, -exp],
            name="Expect Length",
            mode="lines+markers",
            line=dict(color="rgba(0,0,0,1)", width=3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[-h_proj, -d_proj, 0, o_proj, h_proj, d_proj, 0, -o_proj, -h_proj],
            y=[0, d_proj, v_proj, o_proj, 0, -d_proj, -v_proj, -o_proj, 0],
            name="Actual Length",
            mode="lines+markers",
            line=dict(color="rgba(255,0,0,1)", width=3),
        )
    )

    grid_limit = int(lim) + 10
    fig = _add_guiding_lines(fig, grid_limit)

    if filename:
        pyo.plot(fig, filename=f".{os.sep}figs{os.sep}{filename}")
    return fig


def segment(df1, df2, name1, name2, filename=None):
    """plot segment graph

    Args:
        df1 (pandas.DataFrame): dataframe with ``Upper Bound`` and ``Lower Bound`` columns
        df2 (pandas.DataFrame): dataframe with ``Upper Bound`` and ``Lower Bound`` columns
        name1 (str): name of first dataframe
        name2 (str): name of second dataframe
        filename (str, optional): The local path and filename to save the outputted chart to.
            if filename is not set, figure will not be exported. Defaults to None.

    Returns:
        plotly Figure.
    """
    df1 = refactor_interval(df1)
    df2 = refactor_interval(df2)

    points_df1 = go.Scatter(
        x=df1["Central"],
        y=df1["Range"],
        name=name1,
        mode="markers",
        marker=dict(
            color="rgba(30,144,255,0.8)",
            size=8,
            line=dict(color="rgb(28,28,28)", width=2),
        ),
    )
    points_df2 = go.Scatter(
        x=df2["Central"],
        y=df2["Range"],
        name=name2,
        mode="markers",
        marker=dict(color="green", size=8, line=dict(color="rgb(28,28,28)", width=2)),
    )
    shapes = [
        dict(
            type="line",
            x0=df1["Central"].loc[i],
            y0=df1["Range"].loc[i],
            x1=df2["Central"].loc[i],
            y1=df2["Range"].loc[i],
            line=dict(color="rgba(112,128,114,0.6)", width=2),
        )
        for i in range(df1.shape[0])
    ]

    df1_lim, df2_lim = df1.max().max(), df2.max().max()
    lim = max(df1_lim, df2_lim)
    ranges = [-(lim + 0.5), lim + 0.5]

    layout = go.Layout(
        shapes=shapes,
        title="Segement Plot" + f" << {name1} and {name2} >>",
        xaxis_title="Central",
        yaxis_title="Range",
    )
    fig = go.Figure([points_df1, points_df2], layout)
    fig.update_xaxes(range=ranges)
    fig.update_yaxes(range=ranges)

    grid_limit = int(lim) + 10
    fig = _add_guiding_lines(fig, grid_limit)

    if filename:
        pyo.plot(fig, filename=f".{os.sep}figs{os.sep}{filename}")
    return fig


def rectangle(df1, df2, name1, name2, filename=None):
    """plot rectangle graph

    Args:
        df1 (pandas.DataFrame): dataframe with ``Upper Bound`` and ``Lower Bound`` columns
        df2 (pandas.DataFrame): dataframe with ``Upper Bound`` and ``Lower Bound`` columns
        name1 (str): name of first dataframe
        name2 (str): name of second dataframe
        filename (str, optional): The local path and filename to save the outputted chart to.
            if filename is not set, figure will not be exported. Defaults to None.

    Returns:
        plotly Figure.
    """
    df1 = refactor_interval(df1, standard=False)
    df2 = refactor_interval(df2, standard=False)

    data_min = go.Scatter(
        x=df1["Central"] - df1["Range"] / 2,
        y=df2["Central"] - df2["Range"] / 2,
        mode="markers",
        marker=dict(color="RoyalBlue"),
    )
    data_max = go.Scatter(
        x=df1["Central"] + df1["Range"] / 2,
        y=df2["Central"] + df2["Range"] / 2,
        mode="markers",
        marker=dict(color="RoyalBlue"),
    )

    shapes = [
        dict(
            type="rect",
            x0=df1["Central"].loc[i] - df1["Range"].loc[i] / 2,
            y0=df2["Central"].loc[i] - df2["Range"].loc[i] / 2,
            x1=df1["Central"].loc[i] + df1["Range"].loc[i] / 2,
            y1=df2["Central"].loc[i] + df2["Range"].loc[i] / 2,
            line=dict(color="RoyalBlue"),
        )
        for i in range(len(df1["Central"]))
    ]

    layout = go.Layout(
        shapes=shapes, title="Rectangle Plot", xaxis_title=name1, yaxis_title=name2
    )
    fig = go.Figure([data_min, data_max], layout)
    fig.update_layout(showlegend=False)

    if filename:
        pyo.plot(fig, filename=f".{os.sep}figs{os.sep}{filename}")
    return fig
