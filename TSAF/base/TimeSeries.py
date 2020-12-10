#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 15:30:17 2020

@author: viniciussantino
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
# =====================================================================

class TimeSeries:
    def __init__(self, data:pd.Series):
        self.data = data
        
        #from sklearn.preprocessing import MinMaxScaler
        #self.scaler = MinMaxScaler()
        #self.scaler.fit(self.data)
        
   # def normalised(self):
   #     return self.scaler.transform(self.data)
    
   # def denormalise(self, newTS):
   #     return self.scaler.inverse_transform(newTS)
    
    def root_mean_squared_error(actual, predicted):
        from sklearn.metrics import mean_squared_error
        from math import sqrt
        return sqrt(mean_squared_error(actual, predicted))
    
    def adjusted_R2(actual, predicted,order):
        from sklearn.metrics import r2_score
        return 1-(1-r2_score(actual,predicted))*(len(actual)-1)/(len(actual)-1-order)
    
    def weighted_mean_absolute_percentage_error(actual, predicted):
        return np.sum(np.abs(actual - predicted))/np.sum(np.abs(actual))
    
    def relative_error(actual, predicted):
        from sklearn.metrics import mean_absolute_error
        return mean_absolute_error(predicted,actual)/mean_absolute_error(np.roll(actual,shift=1),actual)
    
    def toMachineLearningFormat(self, lags):
        df2 = pd.DataFrame()
        self.steps = lags
        self.index = []
        for i in range(0, self.steps):
            df2[i] = self.data.shift(periods=-1 * i)
        self.originalMatrix = df2.to_numpy()
        npMatrix = df2.dropna().to_numpy()
        return npMatrix
    
    
    def toMachineLearningFormatWithMissing(self,lags):
        return None
    
    def diff(self):
        return self.data.diff()
    
    def un_diff(self, ts):
        return self.data.shift(1)+ts
    
    def short_analysis(self):
        from statsmodels.graphics.tsaplots import plot_acf
        from statsmodels.graphics.tsaplots import plot_pacf
        print("original time series")
        plot_acf(self.data.dropna(), lags = 100)
        plot_pacf(self.data.dropna(),lags=100)
        print("after diff")
        plot_acf(self.data.diff().dropna(), lags = 100)
        plot_pacf(self.data.diff().dropna(),lags=100)
        
    def from_mlf_to_time_series(self,arrayValues):
        arrayWithNan = np.isnan(np.sum(self.originalMatrix,1))
        timeseriesArray = np.empty(len(arrayWithNan))
        timeseriesArray[:] = np.nan
        positionOnTransformedData = 0
        for i in range(0,len(arrayWithNan)-self.steps):
            #print('Values on i: '+i+' Value on indexData: '+positionOnTransformedData+' Value in arrayOf None: '+arrayWithNan[i])
            if ~arrayWithNan[i]:
                timeseriesArray[i+self.steps-1] = arrayValues[positionOnTransformedData]
                positionOnTransformedData=positionOnTransformedData+1
        return pd.Series(data=timeseriesArray, index=self.data.index)
    
    def linear_standarlisation(self, data=None):
        if data is None:
            data = self.data
        self.scaler = MaxAbsScaler()
        print(self.data.to_numpy().shape)
        self.scaler.fit(self.data.values.reshape(-1,1))
        dt = self.scaler.transform(data.values.reshape(-1, 1))
        print(dt.shape)
        print(dt[1,0])
        return pd.Series(dt.flatten(),data.index)

    def undo_linear_standarlisation(self, ts):
        return ts#pd.Series(self.scaler.inverse_transform(ts.values.reshape(-1,1)).flatten(),ts.index)
    
 