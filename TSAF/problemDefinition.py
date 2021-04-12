#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 13:19:33 2020

@author: viniciussantino
"""
import numpy as np
import pandas as pd
from pygam import LinearGAM
from sklearn.linear_model import Lasso
from TSAF.base.Series import Series
import statsmodels.tsa.stattools as stat
from statsmodels.tsa.stattools import kpss
import datetime
from enum import Enum

"""
#define class to system requirements

Identify data characteristics will run a set of default methods to automatic 
identify the data characteristics, in case of problem, the method will return 
what was possible to detect and put a warning on the output with the analysis
and the parameters that the user can set.
"""

'''Load a time series from a JSON file as TSAFSeries'''


def load_json_time_series(file_name: str = None) -> Series:
    series: pd.Series = pd.read_json(path_or_buf=file_name, typ='series')
    return Series(series.array, series.index)


def summary_series(time_series: Series):
    trimmed_time_series = time_series.trim_time_series
    summary = {'observation': len(trimmed_time_series),
               'beginning': (trimmed_time_series.index[0] - datetime.datetime.utcfromtimestamp(0)).total_seconds(),
               'end': (trimmed_time_series.index[-1] - datetime.datetime.utcfromtimestamp(0)).total_seconds(),
               'period': (trimmed_time_series.index[1] - trimmed_time_series.index[0]).total_seconds(),
               'null_observations': sum(trimmed_time_series.isnull()),
               'continuous_observations': len(
                   trimmed_time_series.biggest_continuous_segment) if trimmed_time_series.biggest_continuous_segment is not None else 0,
               'has_zeros': trimmed_time_series.has_zeros}
    return summary


def summary_data(timeSeriesDic: {str: Series}):
    summary_dic = dict()
    for key in timeSeriesDic.keys:
        summary_dic[key] = summary_series(timeSeriesDic.get(key))

    return summary_dic


# Tidal cycles: 12h24m (average duration of a low tide + a high tide) = 12.4*4 = 50 / 1
# Daily cycle: 24h = 24*4 = 96 / 4
# Annual cycle: 365.242199d = 365*24*4 = 35040 / 96
def analysis_components(time_series: Series = None, type="Oceanography", period_season=None, names_season=None,
                        freq_sample=None, show_analysis=False):
    if type != "Oceanography":
        return time_series.seasonality_detector(period_season, names_season, freq_sample, show_analysis)
    else:
        return time_series.seasonality_detector(period_season=[50, 96, 35040],
                                                names_season=["Tidal", "Daily", "Annual"], freq_sample=[1, 4, 96],
                                                show_analysis=show_analysis)


#  "diff" , callable object
def remove_seasonality(time_series: Series = None, seasonal_function=None, period_season: [int] = None,
                       names_season: [str] = None, freq_sample: [int] = None) -> Series:
    if period_season == None:
        period_season = [50, 96, 2688, 35040]
        names_season = ["Tidal", "Daily", "Moon cycle", "Annual"]
        freq_sample = [1, 4, 48, 96]
    if seasonal_function == None:
        return None
    elif isinstance(seasonal_function, str) and seasonal_function == "diff":
        return Series(time_series.diff())
    elif isinstance(seasonal_function, str) and (seasonal_function == "poly" or seasonal_function == "linear"):
        return time_series.remove_seasonality(seasonal_function, period_season, names_season, freq_sample)


def identify_time_series_characteristics(timeSeries: Series):
    # test stationary
    # find the longest segment of the timeseries free of missing values
    # apply adfuller and kpss to test stationarity, if any of the are positive
    # stationary = positive

    # stationary is using 10% level
    statistical_level = '5%'
    significant_level = 0.05
    subseries = timeSeries.biggest_continuous_segment

    print('calculation ADF test')
    adf_test = stat.adfuller(subseries,
                             maxlag=200)  # Augmented Dickey Fuller detect strict stationary and difference stationary
    # print(adf_test[-1].H0)
    # print(adf_test[-1].HA)
    print('calculating kpss test')
    kpss_test = kpss(subseries,
                     regression='c',
                     nlags='auto')  # Kwiatkowski-Phillips-Schmidt-Shin detect strict stationary and trend stationary
    # print(kpss_test[-1].H0)
    # print(kpss_test[-1].HA)
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
        characteristics["diff"] = False
        characteristics["trend"] = True
    elif adf_stationary:
        characteristics["stationary"] = "Sample"
        characteristics["diff"] = True
        characteristics["trend"] = False
    else:
        characteristics["stationary"] = "No"

    if kpss_stationary_na_free and adf_stationary_na_free:
        characteristics["stationary"] = "Population"

    return characteristics

class TypeStationaritySeries(Enum):
    Stationary = 1
    Linear = 2
    Polynomial = 3
    Diff = 4
    Fail = 5

def filter_parameters(methods: pd.DataFrame = None, measures: pd.DataFrame = None, filters: dict = None) -> (
pd.DataFrame, pd.DataFrame, dict):
    filtered_methods = pd.DataFrame(methods.copy())
    filtered_measures = pd.DataFrame(measures.copy())
    pipe_config = dict()
    type_series:TypeStationaritySeries = TypeStationaritySeries.Stationary
    #generate filters
    if filters['priority'] == 'statistical validity':
        type_series = TypeStationaritySeries.Stationary if filters['stationary']['stationary'] != 'No' else (TypeStationaritySeries.Linear if 'No' != filters['linear_seasonal_stationary']['stationary'] else (TypeStationaritySeries.Polynomial if filters['polynomial_seasonal_stationary']['stationary'] != 'No' else (TypeStationaritySeries.Diff if filters['difference_stationary']['stationary'] != 'No' and (filters['horizon'] <= 1 or filters['longterm_simulation'] == 'sequential') else TypeStationaritySeries.Fail)))
    if type_series == TypeStationaritySeries.Stationary:#consider all the applications based on the statistics from data.
        print("code here... case not implemented yet.")
    elif type_series == TypeStationaritySeries.Fail:
        print("code here... case not implemented yet.")
    else:
        filtered_methods['valid_interpretability'] = filtered_methods['intepretability'].apply(lambda x: filters['interpretability'] in x)
        filtered_methods['valid_missing'] = filtered_methods['missing'].apply(lambda x: True if filters['null_observations'] == 0 else x)

    #apply filters on methods
    for column in filtered_methods.columns[len(methods.columns):]:
        filtered_methods = filtered_methods[filtered_methods.eval(filtered_methods[column])]

    return (filtered_methods, filtered_measures, pipe_config)




def identify_data_characteristics(timeSeries: [pd.Series]):
    # TODO implement support to array of time series
    return 0
