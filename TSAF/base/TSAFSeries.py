#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 10:11:00 2021

@author: Vinicius Santino
"""
from pandas import Series, DataFrame
import math
import numpy as np


class TSAFSeries(Series):

    @property
    def biggest_continuous_segment(self):
        list_null = self.isnull().values

        lower_index_biggest_segment = math.inf
        higher_index_biggest_segment = 0
        length_segment = 0
        actual_lower = 0

        for k in range(0, len(list_null)):
            if list_null[k]:
                length = k - actual_lower
                if length > length_segment:
                    lower_index_biggest_segment = actual_lower
                    higher_index_biggest_segment = k
                    length_segment = length
                actual_lower = k + 1
        return TSAFSeries(self[lower_index_biggest_segment:higher_index_biggest_segment])

    @property
    def has_zeros(self) -> bool:
        for value in self.dropna().values:
            if value == 0:
                return True
        return False

    @property
    def trim_time_series(self) -> Series:
        # run from beginning to end and get the first non-null position
        beginning = -1
        index = 0
        while beginning < 0 and index < len(self):
            if math.isnan(self[index]):
                index = index + 1
            else:
                beginning = index
        # run from end to beginning and get the first non-null position
        end = len(self)
        index = len(self) - 1
        while end == len(self) and index > 0:
            if math.isnan(self[index]):
                index = index - 1
            else:
                end = index
        # generate the new Series
        if beginning != end and beginning >= 0 and end < len(self):
            return self[beginning:end]
        return None


def _seasonality_detector(self, period_season: [int] = None,
                          names_season: [str] = None, freq_sample: [int] = None, show_analysis: bool = None) -> pd.DataFrame:
    self.freq_sample = freq_sample
    self.names_season = names_season
    self.period_season = period_season

    #data preparation
    dataset = DataFrame()
    index: int = 0
    for name in names_season:
        new_index = list(range(period_season[index]))

        new_index = list(np.floor_divide(new_index, freq_sample[index]))

        new_index = new_index * (int(len(self) / period_season[index]) + 1)
        dataset[name] = new_index[0:len(self)]
        index = index+1
    dataset['values'] = self.values
    dataset = dataset.dropna()
    data = dataset.to_numpy()
    independent_variables = data[:,0:-1]
    dependent_variable = data[:,-1]
    splits = 5
    kf = KFold(n_splits=splits, random_state=None, shuffle=False)
    results_g = []
    results_r = []
    gam = None
    reg = None
    for train_index, test_index in kf.split(independent_variables):
        gam = LinearGAM(n_splines=25).fit(X=independent_variables[train_index],y=dependent_variable[train_index])
        reg = Lasso().fit(X=independent_variables[train_index], y=dependent_variable[train_index])
        predictionsG = gam.predict(X=independent_variables[test_index])
        predictionsR = reg.predict(X=independent_variables[test_index])
        results_r.append(mean_absolute_error(y_pred=predictionsR, y_true=dependent_variable[test_index]))
        results_g.append(mean_absolute_error(y_pred=predictionsG, y_true=dependent_variable[test_index]))

    result = pd.DataFrame()
    result["linear"] = results_r
    result["poly"] = results_g

    if show_analysis:

        #
        #  print("linear model R^2:", reg.score(independent_variables, dependent_variable))
        #
        #  gam = LinearGAM(n_splines=25).gridsearch(independent_variables, dependent_variable)
        plt.figure()
        fig, axs = plt.subplots(1, len(period_season))
        fig.suptitle("Partial dependence by seasonal variable", fontsize=14)
        for i, ax in enumerate(axs):
            XX = gam.generate_X_grid(term=i)
            ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
            ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')
            ax.set_title(names_season[i]);
        plt.show()
        #  gam.summary()
        #
        plt.close(fig)
    return result
