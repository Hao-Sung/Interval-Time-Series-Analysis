import copy
import numpy as np


def data_rearrange(data, auto_order):
    """rearrange the data to make it easier to perform matrix operations"""
    res = np.array([data[auto_order : len(data)]])
    for i in range(auto_order):
        res = np.vstack((res, np.array(data[auto_order - i - 1 : len(data) - i - 1])))
    return res


def get_reverse_data(endog):
    """return reverse time series for backcast process"""
    data = copy.deepcopy(endog)
    data_reverse = copy.deepcopy(endog)
    data_reverse["Upper"]["data"] = data_reverse["Upper"]["data"][::-1]
    data_reverse["Lower"]["data"] = data_reverse["Lower"]["data"][::-1]
    return (data, data_reverse)


def redefine_interval(df):
    """validate interval data and return it in 'dict' format"""
    row_num = df.shape[0]
    criteria = (df.iloc[:, 0] > df.iloc[:, 1]).sum()
    if int(criteria) == 0:
        upper_index = 1
        lower_index = 0
    elif int(criteria) == row_num:
        upper_index = 0
        lower_index = 1
    else:
        raise ValueError("Some Upper bounds are smaller than lower bounds.")
    return {
        "Upper": {
            "col_name": df.columns[upper_index],
            "data": df.iloc[:, upper_index].values,
        },
        "Lower": {
            "col_name": df.columns[lower_index],
            "data": df.iloc[:, lower_index].values,
        },
    }
