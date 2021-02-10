#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  23 10:19:33 2020

@author: viniciussantino
"""
import datetime
import math
import os
import urllib.request
from typing import Any, Union

from numpy import unique
import pandas as pd
from netCDF4 import Dataset, num2date
from pandas.io.json._json import JsonReader
from tqdm import tqdm


# https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/ANMN/NRS/NRSMAI/aggregated_timeseries/IMOS_ANMN-NRS_SZ_20080411_NRSMAI_FV01_PSAL-aggregated-timeseries_END-20200522_C-20201207.nc
# "https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/ANMN/NRS/NRSMAI/aggregated_timeseries/IMOS_ANMN-NRS_VZ_20110728_NRSMAI_FV01_velocity-aggregated-timeseries_END-20191212_C-20201007.nc"
# loadTsUrl(url='https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/ANMN/NRS/NRSMAI/aggregated_timeseries/IMOS_ANMN-NRS_OZ_20080411_NRSMAI_FV01_DOX1-aggregated-timeseries_END-20180621_C-20201207.nc',variables_of_interest = ['DOX1'], index_variable = 'instrument_index', dimensional_variable='NOMINAL_DEPTH')
# ['UCUR','VCUR','WCUR']
def load_time_series_json(url: str = None)->pd.Series:
    series: pd.Series = pd.read_json(url, typ='series', orient='records')
    return series

def load_data_from_site(site: str = None, sensor: str = None):
    dicSites = {'NRSMAI':
        {'CPHL':
            {
                'url': "https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/ANMN/NRS/NRSMAI/aggregated_timeseries"
                       "/IMOS_ANMN-NRS_BZ_20080411_NRSMAI_FV01_CPHL-aggregated-timeseries_END-20180621_C-20201207.nc ",
                'variables': ['CPHL'],
                'index_variable': 'instrument_index',
                'dimensional_variable': 'NOMINAL_DEPTH'
            },
            'DOX1':
                {
                    'url': 'https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/ANMN/NRS/NRSMAI'
                           '/aggregated_timeseries/IMOS_ANMN-NRS_OZ_20080411_NRSMAI_FV01_DOX1-aggregated'
                           '-timeseries_END-20180621_C-20201207.nc',
                    'variables': ['DOX1'],
                    'index_variable': 'instrument_index',
                    'dimensional_variable': 'NOMINAL_DEPTH'
                },
            'DOX2':
                {
                    'url': 'https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/ANMN/NRS/NRSMAI'
                           '/aggregated_timeseries/IMOS_ANMN-NRS_OZ_20090902_NRSMAI_FV01_DOX2-aggregated'
                           '-timeseries_END-20140406_C-20201207.nc',
                    'variables': ['DOX2'],
                    'index_variable': 'instrument_index',
                    'dimensional_variable': 'NOMINAL_DEPTH'
                },
            'PSAL':
                {
                    'url': 'https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/ANMN/NRS/NRSMAI'
                           '/aggregated_timeseries/IMOS_ANMN-NRS_SZ_20080411_NRSMAI_FV01_PSAL-aggregated'
                           '-timeseries_END-20200522_C-20201207.nc',
                    'variables': ['PSAL'],
                    'index_variable': 'instrument_index',
                    'dimensional_variable': 'NOMINAL_DEPTH'
                },
            'TURB':
                {
                    'url': 'https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/ANMN/NRS/NRSMAI'
                           '/aggregated_timeseries/IMOS_ANMN-NRS_UZ_20080411_NRSMAI_FV01_TURB-aggregated'
                           '-timeseries_END-20180621_C-20201207.nc',
                    'variables': ['TURB'],
                    'index_variable': 'instrument_index',
                    'dimensional_variable': 'NOMINAL_DEPTH'
                }
        }
    }
    if site is None:
        for sitesData in dicSites.values():
            _loadSensorsFromSite(sitesData, sensor)

    elif dicSites.get(site) is not None:
        sitesData = dicSites.get(site)
        _loadSensorsFromSite(sitesData, sensor)


def _loadSensorsFromSite(sitesData: [dict], sensor: str = None):
    if sensor is None:
        for sensorData in sitesData.values():
            _loadDataFromDictionary(sensorData)
    else:
        sensorData = sitesData.get(sensor)
        if sensorData is not None:
            _loadDataFromDictionary(sensorData)


def _loadDataFromDictionary(sensorData: dict = None) -> dict:
    url = sensorData["url"]
    variable_of_interest = sensorData['variables']
    index_variable = sensorData['index_variable']
    dimensional_variable = sensorData['dimensional_variable']
    try:
        return load_ts_url(url=url, variables_of_interest=variable_of_interest, index_variable=index_variable,
                           dimensional_variable=dimensional_variable)
    except:
        print("fail to import: %s \n from file: %s", variable_of_interest[0], url)
        return None


def load_ts_url(
        url: str = None
        , variables_of_interest: [str] = None, index_variable: str = None,
        dimensional_variable: str = None):
    dir = os.path.dirname(__file__)

    current_time = datetime.datetime.now()
    date_format = "%Y-%m-%dT%H-%M-%SZ"
    date_beginning_download = datetime.datetime.strftime(current_time, date_format)
    netcdf_file = "netcdf_file_" + date_beginning_download + ".auxdat"

    filename = os.path.join(dir, 'downloads', netcdf_file)

    urllib.request.urlretrieve(url, filename)
    data_netcdf4 = Dataset(filename)
    site = data_netcdf4.site_code
    time_series_of_interest = convert_net_cdf_aggregatedata_in_time_series(data_netcdf4, variables_of_interest,
                                                                           index_variable, dimensional_variable)
    for key in time_series_of_interest.keys():
        # preparing file name of the output json
        data_format = "%Y-%m-%dT%H:%M:%SZ"
        first_date = datetime.datetime.strptime(data_netcdf4.time_coverage_start,
                                                data_format)
        last_date = datetime.datetime.strptime(data_netcdf4.time_coverage_end,
                                               data_format)
        date_format = "%Y-%m-%dT%H-%M-%SZ"
        date_beginning_download = datetime.datetime.strftime(current_time, date_format)
        json_str_file = site + '_' + key + '_' + datetime.datetime.strftime(first_date,
                                                                            date_format) + '_' + datetime.datetime.strftime(
            last_date, date_format)
        filename = os.path.join(dir, 'data', json_str_file)
        #
        for depth in time_series_of_interest[key].keys():
            series: pd.Series = time_series_of_interest[key][depth]
            series.to_json(path_or_buf=filename + '_d_' + str(depth) + '.json')

    return time_series_of_interest


def _calculateIntervalFromNetCDF(data, lowerBoundary, burst_limit_size_in_seconds=30, original_index=0):
    interval = num2date(data.variables['TIME'][lowerBoundary + 1], data.variables['TIME'].units) - num2date(
        data.variables['TIME'][lowerBoundary], data.variables['TIME'].units)
    if interval.total_seconds() > burst_limit_size_in_seconds:
        return num2date(data.variables['TIME'][lowerBoundary + 1], data.variables['TIME'].units) - num2date(
            data.variables['TIME'][original_index], data.variables['TIME'].units)
    else:
        return _calculateIntervalFromNetCDF(data, lowerBoundary + 1, original_index=original_index)


def _write_time_series_from_netcdf_time_series(series, output_dictionary_of_time_series, interval, index,
                                               dimensions_index: pd.Series) -> dict:
    """
    # copy a time series content from a burst-like time series to a stantard time series
    
    
    """
    x = 0
    y = 0

    print("Observations on queue:", len(series))
    delta = (interval / 1.7).total_seconds()
    cumulated_value = 0
    number_of_terms = 0
    prior_index = int(dimensions_index[index[0]])
    progress_bar: tqdm = tqdm(total=int(len(series) / 100000))

    while x < len(series):  # and y < len((finalTM[dimensions_index[index[x]]])):
        if (x % 100000) == 0:
            progress_bar.update()
        if int(dimensions_index[index[x]]) != prior_index:
            if number_of_terms > 0:
                (output_dictionary_of_time_series[prior_index])[y] = cumulated_value / number_of_terms
                cumulated_value = 0
                number_of_terms = 0
            prior_index = int(dimensions_index[index[x]])
            y = 0
        elif math.fabs(
                (series.index[x] - (output_dictionary_of_time_series[prior_index]).index[y]).total_seconds()) < delta:
            cumulated_value = cumulated_value + series[x]
            number_of_terms = number_of_terms + 1
            x = x + 1
        elif series.index[x] > (output_dictionary_of_time_series[prior_index]).index[y]:
            # verify if discharge is needed
            if number_of_terms > 0:
                (output_dictionary_of_time_series[prior_index])[y] = cumulated_value / number_of_terms
            y = y + 1
            if y >= len((output_dictionary_of_time_series[prior_index])):
                print("system ERROR, not expected index")
            cumulated_value = 0
            number_of_terms = 0
        else:
            x = x + 1
    if number_of_terms > 0:
        (output_dictionary_of_time_series[prior_index])[y] = cumulated_value / number_of_terms
    progress_bar.close()
    return output_dictionary_of_time_series


def convert_net_cdf_aggregatedata_in_time_series(source_data: Dataset, interest_variables: [str],
                                                 index_variable: str = 'instrument_index',
                                                 dimensional_variable: str = 'NOMINAL_DEPTH') -> dict:
    """
    #generate pandas time series from aggregated netCDF file
    
    The files deployments as well as the aggregated data product can not be used
    as time series because there are no warranties of the data been with same
    interval.
    
    This function standarlise the data by generating a fixed intertal time-series
    and filling the data from the aggregated data product.
    
    To execute the same procedure on a set of netCDF files, please access adjustTimeSeries.readTimeSeries()
    
    input: netCDF dataset
    output: pandas.series
    """

    # interval is diff(source_data[0]-source_data[1]) all sensors except optical sensors.
    data_format = "%Y-%m-%dT%H:%M:%SZ"

    interval = _calculateIntervalFromNetCDF(source_data, 0)
    interval2 = _calculateIntervalFromNetCDF(source_data, 0, 0)
    first_date = datetime.datetime.strptime(source_data.time_coverage_start,
                                            data_format)  # num2date(source_data.variables['TIME'][0], source_data.variables['TIME'].units)
    last_date = datetime.datetime.strptime(source_data.time_coverage_end,
                                           data_format)  # num2date(source_data.variables['TIME'][-1], source_data.variables['TIME'].units)
    print("initial date:", first_date)
    print("lastDate:", last_date)
    numerical_time_index = source_data.variables['TIME'][:]
    time_index = num2date(numerical_time_index[:], source_data.variables['TIME'].units)

    print("interval", interval)
    print("interval 2", interval2)
    print("dif", (interval == interval2))
    array_sensors = dict()
    dimensions = unique(source_data.variables[dimensional_variable][:])
    dimensions_index = source_data.variables[dimensional_variable][:]
    date_with_safety = last_date + interval

    print("lastDate 2:", date_with_safety)

    index_ = pd.Series(source_data.variables[index_variable][:])
    for k in range(0, len(interest_variables)):
        print("generating pandas series from source")
        source_series = pd.Series(source_data.variables[interest_variables[k]][:], index=time_index[:])
        output_dict = dict()
        date_range = pd.date_range(start=first_date, end=date_with_safety, freq=interval)
        for m in dimensions:
            output_dict[int(m)] = pd.Series(index=date_range)

        print("starting processing time series")
        array_sensors[interest_variables[k]] = _write_time_series_from_netcdf_time_series(source_series, output_dict,
                                                                                          interval, index_,
                                                                                          dimensions_index)

    return array_sensors
