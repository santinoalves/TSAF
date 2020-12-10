#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 13:02:27 2018

@author: viniciussantino
"""


import numpy
import urllib.request
from netCDF4 import Dataset, num2date
import pandas as pd
#import time
import math

import os
#import tensorflow as tf

#import urllib.request



from sklearn.neural_network import MLPRegressor
from matplotlib.pyplot import figure, subplot, plot, xlabel, ylabel, title, setp, show, subplots
from matplotlib.dates import DAILY, DateFormatter, rrulewrapper, RRuleLocator
from datetime import datetime

#import pywt



#import csv
#import math


#from scipy.spatial.distance import euclidean

#from fastdtw import fastdtw

#from operator import itemgetter


#import sklearn.preprocessing as preprocessing


#url = 'https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/ANMN/NRS/NRSDAR/Wave/IMOS_ANMN-NRS_WZ_20110101T225900Z_NRSDAR_FV01_NRSDAR-1101-SUB-Sentinel-or-Monitor-Workhorse-ADCP-22_END-20110427T005906Z_C-20170725T065722Z.nc'
#urllib.request.urlretrieve(url, 'dataI')
#data = Dataset('dataI')
#arrayLabels is a list of labels that will be read from the TM
#arrayData is a list of names (URLS) to import.
def readTimeSeries(arrayLabels,arrayData,name,qualityVariable,qualityFault = 4):
    
    
    
    
    dir = os.path.dirname(__file__)
    filename = os.path.join(dir, 'tempFiles',name)
    arraySensors = pd.DataFrame()
    print(filename)
    urllib.request.urlretrieve(arrayData[0],filename+'First')
    dataFirst = Dataset(filename+'First')
    #print(dataFirst.variables)
    lowerFirst = _calculateLowerLimitBasedOnQualityValue(dataFirst,qualityVariable,qualityFault)
    interval = _calculateIntervalFromNetCDF(dataFirst,lowerFirst)
    firstDate = num2date(dataFirst.variables['TIME'][lowerFirst],dataFirst.variables['TIME'].units )
    
    
    
    urllib.request.urlretrieve(arrayData[-1],filename+'Last')
    dataLast = Dataset(filename+'Last')
    upperLast = _calculateUpperLimit(dataLast,qualityVariable,qualityFault)
    lastDate = num2date(dataLast.variables['TIME'][upperLast],dataLast.variables['TIME'].units)
    
    
    #print('first date:',firstDate)
    #print('last date:',lastDate)
    #print('interval:',interval)
    for k in range (0,len(arrayLabels)):
            arraySensors[arrayLabels[k]] = pd.Series(index= pd.date_range(firstDate, lastDate, freq=interval))
    for j in range(0,len(arrayData)):
        #print("getting url...")
        
        
        urllib.request.urlretrieve(arrayData[j],filename+str(j))
        data = Dataset(filename+str(j))
        
        lowerLimit = _calculateLowerLimitBasedOnQualityValue(data,qualityVariable,qualityFault)
        upperLimit = _calculateUpperLimit(data,qualityVariable,qualityFault)
        times = data.variables['TIME'][:]
        jd = num2date(times[:],data.variables['TIME'].units)
        for k in range (0,len(arrayLabels)):
            hs = pd.Series(data.variables[arrayLabels[k]][lowerLimit:upperLimit],index=jd[lowerLimit:upperLimit])
            arraySensors[arrayLabels[k]] = _writeDataOnDateStructureWithOverwriting(hs,arraySensors[arrayLabels[k]],interval)
    return arraySensors
   
    
def createFullBase(data,classNames):
    #working with all data
    #data = [darwinAirTemperature,darwinBiochemSB,darwinBiochemSF, darwinPressure, darwinWaves,darwinWindSpeedMax,darwinWindSpeedMin  ]
    #classNames = ['darwinAirTemperature','darwinBiochemSB','darwinBiochemSF', 'darwinPressure', 'darwinWaves','darwinWindSpeedMax','darwinWindSpeedMin']
    #define the frequency
    frequency = data[0].index[1] - data[0].index[0]
    maxMinData = data[0].index[0]
    minMaxData = data[0].index[-1]
    for i in range(0,len(data)):
        #define the frequency and date ranges
        frequency = max(frequency, data[i].index[1] - data[i].index[0])
        maxMinData = max(maxMinData,data[i].index[0])
        minMaxData  = min(minMaxData,data[i].index[-1])
    #prepare list of labels: 
    columns=[]
    arraySensors = pd.DataFrame()
    for i in range(0,len(data)):
        for j in range(0,len(data[i].columns)):
            columns.append(classNames[i]+'_'+data[i].columns[j])
            arraySensors[columns[-1]]=_writeDataOnDateStructureSampling(data[i][data[i].columns[j]],pd.Series(index= pd.date_range(maxMinData, minMaxData, freq=frequency)), frequency)
    
    print("database created with variables: ",arraySensors.columns, "data shape: ",arraySensors.shape)
    return arraySensors
    
def _calculateLowerLimitBasedOnQualityValue(tm, qualityVariable, qualityFault):
    qualityControl = tm.variables[qualityVariable]
   
    lowerLimit = -1
    upperLimit = len(qualityControl)
    for x in range (0, upperLimit-1):
        if (not qualityControl[x] == qualityFault) and lowerLimit == -1:
            lowerLimit = x
            break
    return lowerLimit

def _calculateUpperLimit(tm, qualityVariable, qualityFault):
    qualityControl = tm.variables[qualityVariable]
    upperLimit = len(qualityControl[:])
    for x in range (len(qualityControl[:])-1, 0, -1):
            if (not qualityControl[x] == qualityFault) and upperLimit == len(qualityControl[:]):
                upperLimit = x
                break
    return upperLimit

def _calculateIntervalFromNetCDF(data,lowerBoundary):
    interval = num2date(data.variables['TIME'][lowerBoundary+1],data.variables['TIME'].units) - num2date(data.variables['TIME'][lowerBoundary],data.variables['TIME'].units)
    if interval.total_seconds() > 60:
        return interval
    else:
        return _calculateIntervalFromNetCDF(data,lowerBoundary+1)
    

    
def _writeDataOnDateStructureWithOverwriting(timeSeries,finalTM, interval):
    x = 0
    y = 0
    #print('boundary of x',len(timeSeries))
    #print('boundary of y',len(finalTM))
    delta = (interval/1.7).total_seconds()
    cumulatedValue = 0
    numberOfTerms = 0
    while x < len(timeSeries) and  y < len(finalTM):
        
        if math.fabs((timeSeries.index[x] - finalTM.index[y]).total_seconds()) < delta:
            cumulatedValue = cumulatedValue + timeSeries[x]
            numberOfTerms = numberOfTerms + 1
           # finalTM[y] = timeSeries[x]
           # y = y+1
            x = x+1
        elif timeSeries.index[x] > finalTM.index[y]:
            #verify if discharge is needed
            if numberOfTerms > 0:
                finalTM[y] = cumulatedValue/numberOfTerms
            y = y+1
            cumulatedValue = 0
            numberOfTerms = 0
        else:
            x = x+1
    if numberOfTerms > 0:
        finalTM[y] = cumulatedValue/numberOfTerms
    return finalTM
    
def _writeDataOnDateStructureSampling(timeSeries,finalTM, interval):
    x = 0
    y = 0
    #print('boundary of x',len(timeSeries))
    #print('boundary of y',len(finalTM))
    delta = (interval/1.7).total_seconds()
    cumulatedValue = 0
    numberOfTerms = 0
    while x < len(timeSeries) and  y < len(finalTM):
        
        if math.fabs((timeSeries.index[x] - finalTM.index[y]).total_seconds()) < delta:
            cumulatedValue = timeSeries[x]
            numberOfTerms = numberOfTerms + 1
           # finalTM[y] = timeSeries[x]
           # y = y+1
            x = x+1
        elif timeSeries.index[x] > finalTM.index[y]:
            #verify if discharge is needed
            if numberOfTerms > 0:
                finalTM[y] = cumulatedValue
            y = y+1
            cumulatedValue = 0
            numberOfTerms = 0
        else:
            x = x+1
    if numberOfTerms > 0:
        finalTM[y] = cumulatedValue
    return finalTM

def adjustTimeSeries(value1Series,time1Series,value2Series,time2Series):
    #find the first time that are present in both series --> startperiod
    #find the last time that are present in both series --> endperiod
    #adjust granularity of series based on the first
    #return series adjusted and cutted on start and end period
    generalBeginning = 0
    while generalBeginning < len(time1Series):
        if time1Series[generalBeginning] < time2Series[0]:
            generalBeginning = generalBeginning+1
        else:
            break
    
    generalEnding = len(time1Series)-1
    while generalEnding > 0:
        if time1Series[generalEnding] > time2Series[-1]:
            generalEnding = generalEnding-1
        else:
            break
    
    returnTimeSeries1 = numpy.zeros((3,(generalEnding-generalBeginning)))
   
    returnTimeSeries2 = numpy.zeros((3,(generalEnding-generalBeginning)))
    j = 0
    i = 0
    for i in range(generalBeginning,generalEnding):
        while j < len(time2Series):
            if time1Series[i] <= time2Series[j]:
                returnTimeSeries1[0,i - generalBeginning] = i
                returnTimeSeries1[1,i - generalBeginning] = time1Series[i]
                returnTimeSeries1[2,i - generalBeginning] = value1Series[i]
                
                returnTimeSeries2[0,i - generalBeginning] = j
                returnTimeSeries2[1,i - generalBeginning] = time2Series[j]
                returnTimeSeries2[2,i - generalBeginning] = value2Series[j]
                
                break
            else:
                j = j+1
    return [returnTimeSeries1,returnTimeSeries2]


def adjustMeans(timeseries,deltaTime = 0.001388888888889,labelTime = 'TIME',listLabels = ['CNDC','TEMP','PRES_REL','DOXY','CPHL','TURB'], startTimeSeries=0):
    originalTimeSeries = numpy.zeros((len(listLabels)+1,len(timeseries.variables[labelTime])))
    for j in range(0,len(listLabels)+1):
        if j  >= len(listLabels):
            originalTimeSeries[j] = timeseries.variables[labelTime][:]
        else:
            originalTimeSeries[j] = timeseries.variables[listLabels[j]][:]
    
    periods = 0
    cummulatedMeasures = 1
    recalculatedTimeSeries = numpy.zeros((len(listLabels)+1,len(timeseries.variables[labelTime])))
    for j in range(0,len(listLabels)+1):
        recalculatedTimeSeries[j,0] = originalTimeSeries[j,startTimeSeries]
       
            
    
    for i in range(startTimeSeries,len(originalTimeSeries[0])-1):
        
        print(i)
        #new period
        if  originalTimeSeries[-1,i+1] -  originalTimeSeries[-1,i]  > deltaTime:
            
            #extract means
            for j in range(0,len(listLabels)):
                recalculatedTimeSeries[j,periods] = recalculatedTimeSeries[j,periods] / cummulatedMeasures
            
            print('novo periodo')
            print(periods)
            periods = periods + 1
            cummulatedMeasures = 1
            for j in range(0,len(listLabels)+1):
                if j  >= len(listLabels):
                    recalculatedTimeSeries[j,periods] =  originalTimeSeries[-1,i+1]
                else:
                    recalculatedTimeSeries[j,periods] = recalculatedTimeSeries[j,periods] +  originalTimeSeries[j,i+1]
        else:
            cummulatedMeasures = cummulatedMeasures + 1
            for j in range(0,len(listLabels)+1):
                if j  >= len(listLabels):
                    recalculatedTimeSeries[j,periods] =  originalTimeSeries[-1,i+1]
                else:
                    recalculatedTimeSeries[j,periods] = recalculatedTimeSeries[j,periods] +  originalTimeSeries[j,i+1]
                    
        if i >= len(originalTimeSeries[0])-1:
            for j in range(0,len(listLabels)):
                recalculatedTimeSeries[j,periods] = recalculatedTimeSeries[j,periods] / cummulatedMeasures
  


    return recalculatedTimeSeries[:,0:periods]
                
        
def checkGapsOnTimeSeries(dataset,deltaLackData = 0.013888888889):
    holes = 0
    print('PositionOfGaps')
    for i in range (0,len(dataset)-1):
        if dataset[i+1] - dataset[i] > deltaLackData:
            holes = holes + 1
            print(i)
    print('quantity of holes')
    print(holes)

    
def printTimeSeries(CDF4Variable ,timeseries, fileName = '/Users/viniciussantino/Documents/PhD/Code/figure1.pdf',labelsArray = ['CNDC','TEMP','PRES_REL','DOX1_3','CPHL','TURB']):
    timeData = num2date(timeseries[-1,:], CDF4Variable.variables['TIME'].units)
    
    fig1 = figure(num=None, figsize=(25, 20), dpi=80, facecolor='w', edgecolor='k')
    
    for i in range(0,len(labelsArray)):
        ax1=subplot((len(labelsArray)+1)/2,2,i+1)
        plot (timeData,timeseries[i,:])
        title(#DF4Variable.title + '\n' +
      #'%0.2f m depth\n'% TEMP.sensor_depth +
      #'location: lat=%0.2f; lon=%0.2f' % (CDF4Variable.variables['LATITUDE'][:], 
      #                                    CDF4Variable.variables['LONGITUDE'][:])
          CDF4Variable.variables[labelsArray[i]].long_name)
        xlabel(CDF4Variable.variables['TIME'].long_name)
        ylabel( CDF4Variable.variables[labelsArray[i]].units)

        rule = rrulewrapper(DAILY,  interval=60)
        formatter = DateFormatter('%d/%m/%y')
        loc = RRuleLocator(rule)
        ax1.xaxis.set_major_locator(loc)
        ax1.xaxis.set_major_formatter(formatter)
        labels = ax1.get_xticklabels()
        setp(labels, rotation=30, fontsize=10)
    show()
    
    fig1.savefig(fileName, dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='png',
        transparent=True, bbox_inches=None, pad_inches=0.1,
        frameon=None)
    
    
def scatter_matrix_all(frame, alpha=0.5, figsize=None, grid=False, diagonal='hist', marker='.', density_kwds=None, hist_kwds=None, range_padding=0.05, **kwds):

    import matplotlib.pyplot as plt
    from matplotlib.artist import setp
    import pandas.core.common as com
    from pandas.compat import range, lrange, zip
    from statsmodels.nonparametric.smoothers_lowess import lowess
    import numpy as np
    
    df = frame
    num_cols = frame._get_numeric_data().columns.values
    n = df.columns.size
    fig, axes = plt.subplots(nrows=n, ncols=n, figsize=figsize, squeeze=False)

    # no gaps between subplots
    fig.subplots_adjust(wspace=0, hspace=0)

    mask = com.notnull(df)
    marker = _get_marker_compat(marker)

    hist_kwds = hist_kwds or {}
    density_kwds = density_kwds or {}

    # workaround because `c='b'` is hardcoded in matplotlibs scatter method
    kwds.setdefault('c', plt.rcParams['patch.facecolor'])

    boundaries_list = []
    for a in df.columns:
        if a in num_cols:
            values = df[a].values[mask[a].values]
        else:
            values = df[a].value_counts()
        rmin_, rmax_ = np.min(values), np.max(values)
        rdelta_ext = (rmax_ - rmin_) * range_padding / 2.
        boundaries_list.append((rmin_ - rdelta_ext, rmax_+ rdelta_ext))

    for i, a in zip(lrange(n), df.columns):
        for j, b in zip(lrange(n), df.columns):
            ax = axes[i, j]

            if i == j:
                if a in num_cols:    # numerical variable
                    values = df[a].values[mask[a].values]
                    # Deal with the diagonal by drawing a histogram there.
                    if diagonal == 'hist':
                        ax.hist(values, **hist_kwds)
                    elif diagonal in ('kde', 'density'):
                        from scipy.stats import gaussian_kde
                        y = values
                        gkde = gaussian_kde(y)
                        ind = np.linspace(y.min(), y.max(), 1000)
                        ax.plot(ind, gkde.evaluate(ind), **density_kwds)
                    ax.set_xlim(boundaries_list[i])
                else:                # categorical variable
                    values = df[a].value_counts()
                    ax.bar(list(range(df[a].nunique())), values)
            else:
                common = (mask[a] & mask[b]).values
                # two numerical variables
                if a in num_cols and b in num_cols:
                    if i > j:
                        ax.scatter(df[b][common], df[a][common], marker=marker, alpha=alpha, **kwds)
                        # The following 2 lines add the lowess smoothing
                        ys = lowess(df[a][common], df[b][common])
                        ax.plot(ys[:,0], ys[:,1], 'red')
                    else:
                        pearR = df[[a, b]].corr()
                        ax.text(df[b].min(), df[a].min(), 'r = %.4f' % (pearR.iloc[0][1]))
                    ax.set_xlim(boundaries_list[j])
                    ax.set_ylim(boundaries_list[i])
                # two categorical variables
                elif a not in num_cols and b not in num_cols:
                    if i > j:
                        from statsmodels.graphics import mosaicplot
                        mosaicplot.mosaic(df, [b, a], ax, labelizer=lambda k:'')
                # one numerical variable and one categorical variable
                else:
                    if i > j:
                        tol = pd.DataFrame(df[[a, b]])
                        if a in num_cols:
                            label = [ k for k, v in tol.groupby(b) ]
                            values = [ v[a].tolist() for k, v in tol.groupby(b) ]
                            ax.boxplot(values, labels=label)
                        else:
                            label = [ k for k, v in tol.groupby(a) ]
                            values = [ v[b].tolist() for k, v in tol.groupby(a) ]
                            ax.boxplot(values, labels=label, vert=False)

            ax.set_xlabel('')
            ax.set_ylabel('')

            _label_axis(ax, kind='x', label=b, position='bottom', rotate=True)
            _label_axis(ax, kind='y', label=a, position='left')

            if j!= 0:
                ax.yaxis.set_visible(False)
            if i != n-1:
                ax.xaxis.set_visible(False)

    for ax in axes.flat:
        setp(ax.get_xticklabels(), fontsize=8)
        setp(ax.get_yticklabels(), fontsize=8)
    return fig
    

def _label_axis(ax, kind='x', label='', position='top', ticks=True, rotate=False):
    from matplotlib.artist import setp
    if kind == 'x':
        ax.set_xlabel(label, visible=True)
        ax.xaxis.set_visible(True)
        ax.xaxis.set_ticks_position(position)
        ax.xaxis.set_label_position(position)
        if rotate:
            setp(ax.get_xticklabels(), rotation=90)
    elif kind == 'y':
        ax.yaxis.set_visible(True)
        ax.set_ylabel(label, visible=True)
        #ax.set_ylabel(a)
        ax.yaxis.set_ticks_position(position)
        ax.yaxis.set_label_position(position)
    return

def _get_marker_compat(marker):
    import matplotlib.lines as mlines
    import matplotlib as mpl
    if mpl.__version__ < '1.1.0' and marker == '.':
        return 'o'
    if marker not in mlines.lineMarkers:
        return 'o'
    return marker


#2 - run autocorrelation analysis to define intervals:
def outcorrAnalysis(data):
    
    cor = []
    for i in range(0,int(len(data)/2)):
        cor.append(data.autocorr(i))
    return cor

def prepareDataUnToMl(alldData,trainingRange = [0,8850], validationRange = [8851,12000], targetVariable = 'darwinBiochemSB_CNDC', maxvaluesArray = [1, 5, 6, 4259, 4260, 4261]):
    normData = pd.DataFrame()
    cleanData= alldData.dropna()
    for i in range(0,len(alldData.columns)):
        maxValue = max(cleanData[alldData.columns[i]])
        minValue = min(cleanData[alldData.columns[i]])
        delta = maxValue - minValue
        normData[alldData.columns[i]] = pd.Series(((alldData[alldData.columns[i]] - minValue) / delta),index=alldData.index[:])
    
    #prepare database for prediction
    database = numpy.ndarray([])
    first = True
    for i in range(trainingRange[0] + max(maxvaluesArray), trainingRange[-1]):
        if not math.isnan(normData[targetVariable][i]):
            error = False
            lineData = numpy.array([normData[targetVariable][i]])
            for j in range (0,len(maxvaluesArray)):
                if math.isnan(normData[targetVariable][i-maxvaluesArray[j]]):
                    error= True
                else:
                    lineData = numpy.append(lineData,normData[targetVariable][i-maxvaluesArray[j]])
            if error == False:
                if first:
                    print('first data: ',normData.index[i], 'using data from: ',normData.index[i-maxvaluesArray[j]] )
                    first = False
                    database = numpy.ndarray(shape = (1,len(maxvaluesArray)+1),buffer = lineData)
                else:
                   database = numpy.append(database,[lineData],axis=0)
          
    
    validationDatabase =  numpy.ndarray([])
    first = True
    for i in range(validationRange[0], validationRange[-1]):
    
        if not math.isnan(normData[targetVariable][i]):
            error = False
            lineData = numpy.array([normData[targetVariable][i]])
            for j in range (0,len(maxvaluesArray)):
                if math.isnan(normData[targetVariable][i-maxvaluesArray[j]]):
                    error= True
                else:
                    lineData = numpy.append(lineData,normData[targetVariable][i-maxvaluesArray[j]])
            #for j in range (0,len(attributesArray)):
            #    if math.isnan(normData[attributesArray[j]][i]):
            #        error= True
            if error == False:
                if first:
                    print('first data: ',normData.index[i], 'using data from: ',alldData.index[i-maxvaluesArray[len(maxvaluesArray)-1]] )
                    first = False
                    validationDatabase = numpy.ndarray(shape = (1,len(maxvaluesArray)+1),buffer = lineData)
                else:
                   validationDatabase = numpy.append(validationDatabase,[lineData],axis=0)
    return database, validationDatabase

def prepareDataMVToML(alldData,trainingRange = [0,8850], validationRange = [8851,12000], targetVariable = 'darwinBiochemSB_CNDC',attributesArray = ['beagleBiochemSB_CNDC', 'darwinBiochemSB_TEMP', 'darwinBiochemSF_CNDC', 'darwinBiochemSF_TEMP']):
    normData = pd.DataFrame()
    cleanData= alldData.dropna()
    for i in range(0,len(alldData.columns)):
        maxValue = max(cleanData[alldData.columns[i]])
        minValue = min(cleanData[alldData.columns[i]])
        delta = maxValue - minValue
        normData[alldData.columns[i]] = pd.Series(((alldData[alldData.columns[i]] - minValue) / delta),index=alldData.index[:])
    
    databaseMult = numpy.ndarray([])
    first = True
    for i in range(trainingRange[0], trainingRange[-1]):
    
        if not math.isnan(normData[targetVariable][i]):
            error = False
            lineData = numpy.array([normData[targetVariable][i]])
            for j in range (0,len(attributesArray)):
                if math.isnan(normData[attributesArray[j]][i]):
                    error= True
                else:
                    lineData = numpy.append(lineData,normData[attributesArray[j]][i])
            if error == False:
                if first:
                    print('first data: ',normData.index[i])#, 'using data from: ',alldData.index[i-maxvaluesArray[j]] )
                    first = False
                    databaseMult = numpy.ndarray(shape = (1,len(attributesArray)+1),buffer = lineData)
                else:
                   databaseMult = numpy.append(databaseMult,[lineData],axis=0)
      
    
    
    validationDatabaseMult = numpy.ndarray([])
    first = True
    for i in range(validationRange[0], validationRange[-1]):
    
        if not math.isnan(normData[targetVariable][i]):
            error = False
            
            lineData = numpy.array([normData.index[i]])
            lineData = numpy.append(lineData,normData[targetVariable][i])
            for j in range (0,len(attributesArray)):
                if math.isnan(normData[attributesArray[j]][i]):
                    error= True
                else:
                    lineData = numpy.append(lineData,normData[attributesArray[j]][i])
            #for j in range (0,len(maxvaluesArray)):
            #    if math.isnan(alldData[targetVariable][i-maxvaluesArray[j]]):
            #        error= True
            if error == False:
                if first:
                    print('first data: ',normData.index[i])#, 'using data from: ',alldData.index[i-maxvaluesArray[j]] )
                    first = False
                    print(lineData)
                    validationDatabaseMult = numpy.ndarray(shape = (1,len(attributesArray)+2),buffer = lineData)
                else:
                   validationDatabaseMult = numpy.append(validationDatabaseMult,[lineData],axis=0)
    return databaseMult, validationDatabaseMult

def detectOutliers (alldData, trainingRange = [0,8850], validationRange = [8851,12000], targetVariable = 'darwinBiochemSB_CNDC',attributesArray = ['beagleBiochemSB_CNDC', 'darwinBiochemSB_TEMP', 'darwinBiochemSF_CNDC', 'darwinBiochemSF_TEMP'],maxvaluesArray = [1, 5, 6, 4259, 4260, 4261]):
    #intructions to proceed with analysis
    #1 - define target variable
    
    
    
    
    
        
    
    #3 - using the autocorr analysis fill the maxValuesArray with the positions of lags 
    
    
    
    
    #4 use the correlation / DTW analyses to define attribuites to identify the TS
    
    
    #5 - select an interest event to test and define as cutDate (some positions before)
    
    
    
    # run the analysis
    
    normData = pd.DataFrame()
    cleanData= alldData.dropna()
    for i in range(0,len(alldData.columns)):
        maxValue = max(cleanData[alldData.columns[i]])
        minValue = min(cleanData[alldData.columns[i]])
        delta = maxValue - minValue
        normData[alldData.columns[i]] = pd.Series(((alldData[alldData.columns[i]] - minValue) / delta),index=alldData.index[:])
    
    #prepare database for prediction
    database = numpy.ndarray([])
    first = True
    for i in range(trainingRange[0] + max(maxvaluesArray), trainingRange[-1]):
        if not math.isnan(normData[targetVariable][i]):
            error = False
            lineData = numpy.array([normData[targetVariable][i]])
            for j in range (0,len(maxvaluesArray)):
                if math.isnan(normData[targetVariable][i-maxvaluesArray[j]]):
                    error= True
                else:
                    lineData = numpy.append(lineData,normData[targetVariable][i-maxvaluesArray[j]])
            if error == False:
                if first:
                    print('first data: ',normData.index[i], 'using data from: ',normData.index[i-maxvaluesArray[j]] )
                    first = False
                    database = numpy.ndarray(shape = (1,len(maxvaluesArray)+1),buffer = lineData)
                else:
                   database = numpy.append(database,[lineData],axis=0)
          
    
    validationDatabase =  numpy.ndarray([])
    first = True
    for i in range(validationRange[0], validationRange[-1]):
    
        if not math.isnan(normData[targetVariable][i]):
            error = False
            lineData = numpy.array([normData[targetVariable][i]])
            for j in range (0,len(maxvaluesArray)):
                if math.isnan(normData[targetVariable][i-maxvaluesArray[j]]):
                    error= True
                else:
                    lineData = numpy.append(lineData,normData[targetVariable][i-maxvaluesArray[j]])
            for j in range (0,len(attributesArray)):
                if math.isnan(normData[attributesArray[j]][i]):
                    error= True
            if error == False:
                if first:
                    print('first data: ',normData.index[i], 'using data from: ',alldData.index[i-maxvaluesArray[len(maxvaluesArray)-1]] )
                    first = False
                    validationDatabase = numpy.ndarray(shape = (1,len(maxvaluesArray)+1),buffer = lineData)
                else:
                   validationDatabase = numpy.append(validationDatabase,[lineData],axis=0)
    print('size of the database: ', database.shape)
    regressor = MLPRegressor()
    regressor = MLPRegressor.fit(regressor,database[:,1:],database[:,0])
    
    #regressor.score(validationDatabase[:,1:],validationDatabase[:,0])
    
    #regressor.score(validationDatabase[:1000,1:],validationDatabase[:1000,0])
    
    #regressor.score(validationDatabase[1000:,1:],validationDatabase[1000:,0])
    
    prediction = regressor.predict(validationDatabase[1:,1:])
    
    
    #------------------
    #multivariate analysis
    
    
    databaseMult = numpy.ndarray([])
    first = True
    for i in range(trainingRange[0], trainingRange[-1]):
    
        if not math.isnan(normData[targetVariable][i]):
            error = False
            lineData = numpy.array([normData[targetVariable][i]])
            for j in range (0,len(attributesArray)):
                if math.isnan(normData[attributesArray[j]][i]):
                    error= True
                else:
                    lineData = numpy.append(lineData,normData[attributesArray[j]][i])
            if error == False:
                if first:
                    print('first data: ',normData.index[i])#, 'using data from: ',alldData.index[i-maxvaluesArray[j]] )
                    first = False
                    databaseMult = numpy.ndarray(shape = (1,len(attributesArray)+1),buffer = lineData)
                else:
                   databaseMult = numpy.append(databaseMult,[lineData],axis=0)
      
    
    
    validationDatabaseMult = numpy.ndarray([])
    first = True
    for i in range(validationRange[0], validationRange[-1]):
    
        if not math.isnan(normData[targetVariable][i]):
            error = False
            
            lineData = numpy.array([normData.index[i]])
            lineData = numpy.append(lineData,normData[targetVariable][i])
            for j in range (0,len(attributesArray)):
                if math.isnan(normData[attributesArray[j]][i]):
                    error= True
                else:
                    lineData = numpy.append(lineData,normData[attributesArray[j]][i])
            for j in range (0,len(maxvaluesArray)):
                if math.isnan(alldData[targetVariable][i-maxvaluesArray[j]]):
                    error= True
            if error == False:
                if first:
                    print('first data: ',normData.index[i])#, 'using data from: ',alldData.index[i-maxvaluesArray[j]] )
                    first = False
                    print(lineData)
                    validationDatabaseMult = numpy.ndarray(shape = (1,len(attributesArray)+2),buffer = lineData)
                else:
                   validationDatabaseMult = numpy.append(validationDatabaseMult,[lineData],axis=0)
      
    #import sklearn.neural_network
    
    print('size of the database multi: ', databaseMult.shape)
    print('validation base: ',validationDatabaseMult.shape)
    regressor2 = MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=20000, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    regressor2 = MLPRegressor.fit(regressor2,databaseMult[:,1:],databaseMult[:,0])
    
    
    
    #regressor2.score(validationDatabaseMult[:,2:],validationDatabaseMult[:,1])
    
    #regressor2.score(validationDatabaseMult[:1000,2:],validationDatabaseMult[:1000,1])
    
    #regressor2.score(validationDatabaseMult[1000:,2:],validationDatabaseMult[1000:,1])
    
    
    prediction2 = regressor2.predict(validationDatabaseMult[1:,2:])
    
    readings = pd.Series(validationDatabaseMult[1:,1],index=validationDatabaseMult[1:,0])
    #univariateStimate = pd.Series(prediction,index=validationDatabaseMult[1:,0])
    #multivariateStimate = pd.Series(prediction2,index=validationDatabaseMult[1:,0])
    
    error = (readings.values - prediction)**2#abs(readings.values - prediction)/(abs(readings.values )+abs(prediction))
    error2 = (readings.values - prediction2)**2#abs(readings.values - prediction2)/(abs(readings.values )+abs(prediction2))
    
    
    univariateError = pd.Series(error,index=readings.index[:])
    multivariateError = pd.Series(error2, index=readings.index[:])
    
    
    vecfunc = numpy.vectorize(datetime.fromordinal)
    
    maxError = max(max(error),max(error2))
    print('MSE error: ',numpy.mean(univariateError),' MSE Std:',numpy.std(univariateError))
    print('MSE error2: ',numpy.mean(multivariateError), ' MSE 2 Std:',numpy.std(multivariateError))
    fig2, (ax2, ax3) = subplots(nrows=2, ncols=1, sharex=True, figsize=(20,10))
    
    ax2.plot(readings.index[:],readings.values[:],readings.index[:],  prediction[:], label="univariate analisis")#,readings.index[:],error)
    ax2.set_ylabel('Univariate',size=12)
    #ax2.set_xlabel('time',size=12)
    ax2.legend(['Sensor','Prediction'])

    ax3.plot(readings.index[:],error)
    ax3.set_ylabel('Error',size=12)
    ax3.set_xlabel('time',size=12)
    ax3.legend(['MSE'])
    
    ax3.set_ylim(-0.01,maxError*1.1)
    ax3.set_xticks([734517.,  734548.,   734577.,  734608.,   734638.])
    ax3.set_xticklabels(vecfunc([734517,  734548,   734577,  734608,   734638]))
   
    fig1, (ax0, ax1) = subplots(nrows=2, ncols=1, sharex=True, figsize=(20,10))
    
    ax0.plot(readings.index[:],readings.values[:],readings.index[:],  prediction2[:], label="multivariate analisis")#,readings.index[:],error2)
    ax0.set_ylabel('Multivariate',size=12)
    #ax0.set_xlabel('time',size=12)
    ax0.legend(['Sensor','Prediction'])
    
    ax1.plot(readings.index[:],error2)
    ax1.set_ylabel('Error',size=12)
    ax1.set_xlabel('time',size=12)
    ax1.legend(['MSE'])
    
    ax1.set_ylim(-0.01,maxError*1.1)
    
    #print(vecfunc([736009,  736024, 736038,  736055, 736069, 736085]))
    ax1.set_xticks([734517.,  734548.,   734577.,  734608.,   734638.])
    ax1.set_xticklabels(vecfunc([734517,  734548,   734577,  734608,   734638]))
    locs = ax1.get_xticks()
    #locs, labels = plt.xticks()
    print(locs)
    print(readings.index[0])
    
    fig3, (ax01) = subplots(nrows=1,ncols=1, sharex=True, figsize=(20,10))
    
    ax01.plot(readings.index[:],error, readings.index[:],error2,label="errors comparison")
    ax01.set_ylabel('Errors',size=12)
    ax01.set_xlabel('time',size=12)
    ax01.legend(['MSE univariate','MSE multivariate'])
    ax01.set_xticks([734517.,  734548.,   734577.,  734608.,   734638.])
    ax01.set_xticklabels(vecfunc([734517,  734548,   734577,  734608,   734638]))
    
    fig1.savefig('multivariateAnalisis.png', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='png',
        transparent=True, bbox_inches=None, pad_inches=0.1,
        frameon=None)
    
    fig2.savefig('univariateAnalisis.png', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='png',
        transparent=True, bbox_inches=None, pad_inches=0.1,
        frameon=None)
    
    fig3.savefig('comparison.png', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='png',
        transparent=True, bbox_inches=None, pad_inches=0.1,
        frameon=None)
    
def batch_producer(training_features,training_target, batch_size,iterator):
    #print("inside the batch producer")
    observations,features_size = training_features.shape
    epoch_size = (observations-1) // batch_size
    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    #print("having a interator")
    #print('i',i.eval())
    #tf.Print(i,[i],"printing")
    #print("after printing")
    x = training_features[iterator * batch_size:(iterator + 1) * batch_size,:]
    y = training_target[iterator * batch_size: (iterator + 1) * batch_size]
    #print("read to return x and y")
    return x, y