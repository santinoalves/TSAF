#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 13:19:33 2020

@author: viniciussantino
"""
import pandas as pd
import TSAF.base.TimeSeries as ts
import math
import numpy as np
from pygam import LinearGAM, s, f
from matplotlib import pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from base import TSAFSeries

"""
#define class to system requirements

Identify data characteristics will run a set of default methods to automatic 
identify the data characteristics, in case of problem, the method will return 
what was possible to detect and put a warning on the output with the analysis
and the parameters that the user can set.
"""

def load_json_time_series(file_name: str=None) -> pd.Series:
    series: pd.Series = pd.read_json(path_or_buf=file_name, typ='series')
    return series




def summary_series(time_series: pd.Series):
    trimmed_time_series = _trim_time_series(time_series)
    summary = {'observation': len(trimmed_time_series), 'beginning': trimmed_time_series.index[0],
               'end': trimmed_time_series.index[-1],
               'period': (trimmed_time_series.index[1] - trimmed_time_series.index[0]).total_seconds(),
               'null_observations': sum(trimmed_time_series.isnull()),
               'continuos_observations': len(_identifyBiggestContinuosSegment(trimmed_time_series)),
               'has_zeros': has_zeros(trimmed_time_series)}
    return summary





def summary_data(timeSeriesDic: {str: pd.Series}):
    summary_dic = dict()
    for key in timeSeriesDic.keys:
        summary_dic[key] = summary_series(timeSeriesDic.get(key))

    return summary_dic


# Tidal cycles: 12h24m (average duration of a low tide + a high tide) = 12.4*4 = 50 / 1
# Daily cycle: 24h = 24*4 = 96 / 4
# Annual cycle: 365.242199d = 365*24*4 = 35040 / 96


# poly, linear, diff
def remove_seasonality(time_series: pd.Series = None, function_type: str = None, period_season: [int] = None,
                          names_season: [str] = None, freq_sample: [int] = None) -> pd.Series:
    # data preparation
    dataset = pd.DataFrame()
    index: int = 0
    for name in names_season:
        new_index = list(range(period_season[index]))

        new_index = list(np.floor_divide(new_index, freq_sample[index]))

        new_index = new_index * (int(len(time_series) / period_season[index]) + 1)
        dataset[name] = new_index[0:len(time_series)]
        index = index + 1
    dataset['values'] = time_series.values
    dataset = dataset.dropna()
    data = dataset.to_numpy()
    independent_variables = data[:, 0:-1]
    dependent_variable = data[:, -1]
    if function_type:
        if function_type == 'poly':
            gam = LinearGAM(n_splines=25).fit(X=independent_variables, y=dependent_variable)
        elif function_type == 'linear':
            reg = Lasso().fit(X=independent_variables, y=dependent_variable)
        elif function_type == 'diff':
            return time_series.diff()



def identify_time_series_characteristics(timeSeries: pd.Series):
    characteristics = dict()

    # test stationary
    # find the longest segment of the timeseries free of missing values
    # apply adfuller and kpss to test stationarity, if any of the are positive
    # stationary = positive

    # stationary is using 10% level
    statistical_level = '10%'
    subseries = _identifyBiggestContinuosSegment(timeSeries)
    import statsmodels.tsa.stattools as stat
    from statsmodels.tsa.stattools import kpss
    print('calculation ADF test')
    adf_test = stat.adfuller(subseries,
                             maxlag=200)  # Augmented Dickey Fuller detect strict stationary and difference stationary
    print('calculating kpss test')
    kpss_test = kpss(subseries,
                     regression='c',
                     nlags='auto')  # Kwiatkowski-Phillips-Schmidt-Shin detect strict stationary and trend stationary
    # there is a good practical explanation on: https://www.analyticsvidhya.com/blog/2018/09/non-stationary-time-series-python/
    kpss_stationary = kpss_test[0] <= kpss_test[3][
        statistical_level]  # fail to reject the null hypothesis, the series is stationary
    adf_stationary = adf_test[0] <= adf_test[4][
        statistical_level]  # reject the null hypothesis, there is no unit root, hence the time series is stationary

    # Warning, even though both methods can show that the time_series is stationary, it is not necessarily true, because we can be working with a sample that is not significative to represent the full time-series
    # However, if both methods demonstrate that the time-series is not stationary, the result can be extrapolated to the entire time-series. Since a timse-series need to have the same statistics for any windows (including the ones tested by the tests to be stationary

    # reference of the treatment:Ryan, K. F. and D. E. A. Giles, 1998. Testing for unit roots in economic time-series with missing observations. In T. B. Fomby and R. C. Hill (eds.), Advances in Econometrics. JAI Press, Greenwich, CT, 203-242. (The final Working Paper version of the paper is available here, and the Figures are here.)
    # Alternativelly, we can dropna from the time-series, in this case, base on the available data, if the tests keep holding as stationary,
    # the time-series are stationary, contrariwise, we can't avoid manual inspection to determine stationarity on the time-series.

    na_free_time_series = subseries.dropna()
    adf_test_na_free = stat.adfuller(na_free_time_series)
    kpss_test_na_free = kpss(na_free_time_series, regression='c')
    kpss_stationary_na_free = kpss_test_na_free[0] <= kpss_test_na_free[3][statistical_level]
    adf_stationary_na_free = adf_test_na_free[0] <= adf_test_na_free[4][statistical_level]

    characteristics = dict()

    if kpss_stationary and adf_stationary:
        characteristics["stationary"] = "Sample"
        characteristics["diff"] = False
        characteristics["trend"] = False

    elif kpss_stationary:
        characteristics["stationary"] = "Sample"
        characteristics["diff"] = True
        characteristics["trend"] = False
    elif adf_stationary:
        characteristics["stationary"] = "Sample"
        characteristics["diff"] = False
        characteristics["trend"] = True
    else:
        characteristics["stationary"] = "No"

    if kpss_stationary_na_free and adf_stationary_na_free:
        characteristics["stationary"] = "Population"
    else:
        characteristics["stationary"] = "No"
        # generate IAC

    return characteristics


def identify_data_characteristics(timeSeries: [pd.Series]):
    # TODO implement support to array of time series
    return 0
