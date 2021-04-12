#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 10:11:00 2021

@author: Vinicius Santino
"""
import pandas as pd
import math
import numpy as np
from pygam import LinearGAM
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from scipy import signal


class Series(pd.Series):

    def __init__(self, data=None, index=None, dtype=None, name=None, copy=False, fastpath=False):
        if isinstance(data, pd.Series):
            super().__init__(data=data.array, index=data.index, dtype=data.dtype, name=data.name)
        else:
            super().__init__(data=data, index=index, dtype=dtype, name=name, copy=copy, fastpath=fastpath)

    @property
    def biggest_continuous_segment(self):
        list_null = self.isnull().values
        list_null = np.append(list_null,True)


        lower_index_biggest_segment = math.inf
        higher_index_biggest_segment = 0
        length_segment = -1
        actual_lower = 0
        for k in range(0, len(list_null)):
            if list_null[k]:
                length = k - actual_lower
                if length > length_segment:
                    lower_index_biggest_segment = actual_lower
                    higher_index_biggest_segment = k
                    length_segment = length
                actual_lower = k + 1
        if lower_index_biggest_segment < higher_index_biggest_segment:
            return Series(self[lower_index_biggest_segment:higher_index_biggest_segment])
        elif length_segment == -1:
            return Series(self)

    @property
    def has_zeros(self) -> str:
        return str(self.dropna().abs().min() == 0)

    @property
    def trim_time_series(self):
        # run from beginning to end and get the first non-null position
        beginning = -1
        index = 0
        while beginning < 0 and index < len(self):
            self.get
            if math.isnan(self[self.index[index]]):
                index = index + 1
            else:
                beginning = index
        # run from end to beginning and get the first non-null position
        end = len(self)
        index = len(self) - 1
        while end == len(self) and index > 0:
            if math.isnan(self[self.index[index]]):
                index = index - 1
            else:
                end = index
        # generate the new Series
        if beginning != end and beginning >= 0 and end < len(self):
            return Series(self[self.index[beginning]:self.index[end]])
        return None

    def remove_seasonality(self, tyoe=None, period_season: [int] = None,
                           names_season: [str] = None, freq_sample: [int] = None):
        # data preparation
        dataset = pd.DataFrame()
        index: int = 0
        for name in names_season:
            new_index = list(range(period_season[index]))

            new_index = list(np.floor_divide(new_index, freq_sample[index]))

            new_index = new_index * (int(len(self) / period_season[index]) + 1)
            dataset[name] = new_index[0:len(self)]
            index = index + 1
        dataset['values'] = self.values
        dataset_na = dataset.dropna()
        data = dataset_na.to_numpy()

        independent_variables = data[:, 0:-1]
        dependent_variable = data[:, -1]
        if tyoe == "poly":
            gam = LinearGAM(max_iter=200,n_splines=50).fit(X=independent_variables[:],
                                              y=dependent_variable[:])
            prediction = gam.predict(X=dataset.to_numpy()[:, 0:-1])
            return Series(data=self.subtract(pd.Series(prediction,self.index)), index=self.index)
        elif tyoe == "linear":
            linear = Lasso().fit(X=independent_variables[:],
                                 y=dependent_variable[:])
            prediction = linear.predict(X=dataset.to_numpy()[:, 0:-1])
            return Series(data=self.subtract(pd.Series(prediction,self.index)), index=self.index)
        else:
            return None
    @property
    def detrend(self):
        new_index = range(len(self))
        a = Series(data = self.values ,index=new_index)
        data = a.dropna()
        X = np.array(data.index).reshape(-1,1)  # values converts it into a numpy array
        Y = np.array(data.values).reshape(-1,1)  # -1 means that calculate the dimension of rows, but have 1 column
        linear_regressor = LinearGAM(n_splines=25)  # create object for the class
        linear_regressor.fit(X, Y)  # perform linear regression
        prediction = linear_regressor.predict(np.array(list(new_index)).reshape(-1,1)).reshape(1,-1)
        print(prediction)
        return Series(data=self.subtract(pd.Series(prediction[0],self.index)), index=self.index)

    def seasonality_detector(self, period_season: [int] = None,
                             names_season: [str] = None, freq_sample: [int] = None,
                             show_analysis: bool = None):
        self.freq_sample = freq_sample
        self.names_season = names_season
        self.period_season = period_season

        # data preparation
        dataset = pd.DataFrame()
        index: int = 0
        for name in names_season:
            new_index = list(range(period_season[index]))

            new_index = list(np.floor_divide(new_index, freq_sample[index]))

            new_index = new_index * (int(len(self) / period_season[index]) + 1)
            dataset[name] = new_index[0:len(self)]
            index = index + 1
        dataset['values'] = self.values
        dataset = dataset.dropna()
        data = dataset.to_numpy()

        independent_variables = data[:, 0:-1]
        dependent_variable = data[:, -1]
        splits = 5
        kf = KFold(n_splits=splits, random_state=None, shuffle=False)
        results_g = []
        results_r = []
        best_gam_model = None
        best_linear_model = None
        error_best_gam = 10
        error_best_linear = 10
        gam = None
        reg = None
        for train_index, test_index in kf.split(independent_variables):
            gam = LinearGAM(n_splines=25).gridsearch(X=independent_variables[train_index], y=dependent_variable[train_index])
            reg = Lasso().fit(X=independent_variables[train_index], y=dependent_variable[train_index])
            predictionsG = gam.predict(X=independent_variables[test_index])
            predictionsR = reg.predict(X=independent_variables[test_index])
            results_r.append(mean_absolute_error(y_pred=predictionsR, y_true=dependent_variable[test_index]))
            results_g.append(mean_absolute_error(y_pred=predictionsG, y_true=dependent_variable[test_index]))
            if error_best_gam > mean_absolute_error(y_pred=predictionsG, y_true=dependent_variable[test_index]):
                error_best_gam = mean_absolute_error(y_pred=predictionsG, y_true=dependent_variable[test_index])
                best_gam_model = gam
            if error_best_linear > mean_absolute_error(y_pred=predictionsR, y_true=dependent_variable[test_index]):
                error_best_linear = mean_absolute_error(y_pred=predictionsR, y_true=dependent_variable[test_index])
                best_linear_model = reg

        result = dict()
        result["linear"] = {'model': best_linear_model, 'scores': results_r}
        result["poly"] = {'model': best_gam_model, 'scores': results_g}

        if show_analysis:

            plt.figure()
            fig, axs = plt.subplots(1, len(period_season))
            fig.suptitle("Partial dependence by seasonal variable", fontsize=14)
            for i, ax in enumerate(axs):
                XX = gam.generate_X_grid(i, max(period_season))
                np.sort(independent_variables)
                ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
                ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')
                ax.set_title(names_season[i])
            plt.show()

            plt.close(fig)
        return result