'''
This script perfoms the basic process for applying a machine learning
algorithm to a dataset using Python libraries.

The four steps are:
   1. load a dataset (using pandas)
   2. Process the numeric data (using numpy)
   3. Train and evaluate learners (using scikit-learn and tensorflow)
   4. Plot and compare results (using matplotlib)

'''

# Uncomment this call when using matplotlib to generate images
# rather than displaying interactive UI.
#import matplotlib
#matplotlib.use('Agg')


import numpy as np
#import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

import ARIMA as ml_arima

from TimeSeries import TimeSeries
#import adjustTimeSeries as ats
#import seaborn
#from pygam import LinearGAM
from sklearn import neighbors
import split as tscv
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU') 
try: 
  tf.config.experimental.set_memory_growth(physical_devices[0], True) 
except: 
  # Invalid device or cannot modify virtual devices once initialized. 
  pass 

class TimeSeriesCrossValidation:
    def __init__(self,splits:int):
        self.splits = splits
    

# =====================================================================
def load_data(csvname):
    alldData = pd.read_csv(csvname, index_col=0)
    # This dataset was prepared using ats.readTimeSeries and ats.createFullBase
    #to change the dataset from AIMS data, please generate the list of datasets as exemplified on "ML methods - GRU.ipynb"
    return alldData

# =====================================================================
def show_cross_validation(dataOriginal:TimeSeries=None,stationaryData:TimeSeries = None,n_splits=5,gap_ini=100, gap_end=100,lags=50):
    timeSeries = dataOriginal
    data = dataOriginal.toMachineLearningFormat(lags)
    if (stationaryData is not None):
        data = stationaryData.toMachineLearningFormat(lags)
        timeSeries = stationaryData


    kFold = tscv.GapKFold(n_splits=n_splits, gap_before = gap_ini, gap_after = gap_end)
        #cross validation information
    output = np.array([0]*len(data[:,-1]))
    i = 1
    for training, test in kFold.split(data[:,-1]):    
        output[test] = i
        i = i+1
    outputTS = timeSeries.from_mlf_to_time_series(output)
    return 'Cross Validation output test sets', outputTS*-1, outputTS

# =====================================================================

    # a_ = (a - min)/(delta)
    # a_*delta = a - min
    # a =  (a_*delta)+min
def evaluate_learner(dataOriginal:TimeSeries=None,stationaryData:TimeSeries = None,n_splits=6,gap_ini=100, gap_end=100,lags=50):
    '''
    Run multiple times with different algorithms to get an idea of the
    relative performance of each configuration.

    Returns a sequence of tuples containing:
        (title, expected values, actual values)
    for each learner.
    '''
    timeSeries = dataOriginal
    newData = TimeSeries(dataOriginal.linear_standarlisation())
    data_norm = newData.toMachineLearningFormat(lags)
    data = timeSeries.toMachineLearningFormat(lags)
    if (stationaryData is not None):
        newData = TimeSeries(stationaryData.linear_standarlisation())
        data_norm = newData.toMachineLearningFormat(lags)
        timeSeries = stationaryData
        data = timeSeries.toMachineLearningFormat(lags)
        print(data_norm[:10,:])
        
    data_norm = data
    # Use a support vector machine for regression
    from sklearn.svm import SVR

    jobs = -1
    kFold = tscv.GapKFold(n_splits=n_splits, gap_before = gap_ini, gap_after = gap_end)

    # Train using a radial basis function
    '''
    svr = SVR(kernel='rbf', gamma=0.1)
    print("shape of data:",data.shape)
    y_pred = cross_val_predict(svr, data_norm[:,0:-1], data_norm[:,-1], cv=kFold, verbose = 1, n_jobs=jobs)
    y_pred_timeseries = timeSeries.undo_linear_standarlisation(timeSeries.from_mlf_to_time_series(y_pred))
    
    if (timeSeries != dataOriginal):
        y_pred_timeseries = dataOriginal.un_diff(y_pred_timeseries.values)
    pairedSeries = pd.DataFrame(data={'pred':y_pred_timeseries.values,'actual':dataOriginal.data.values},index=dataOriginal.data.index)
    cleanpairedSeries = pairedSeries.dropna()
    rmse = TimeSeries.root_mean_squared_error(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values)
    rel_error = TimeSeries.relative_error(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values)
    wmape = TimeSeries.weighted_mean_absolute_percentage_error(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values)
    r_2 = TimeSeries.adjusted_R2(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values,lags)
    yield 'RBF Model (wmape={:.3f}'.format(wmape)+' rmse={:.3f}'.format(rmse)+' rel. error={:.3f}'.format(rel_error)+' r^2={:.3f})'.format(r_2), dataOriginal.data, y_pred_timeseries

    # Train using a linear kernel
    svr = SVR(kernel='linear')
    y_pred = cross_val_predict(svr, data_norm[:,0:-1], data_norm[:,-1], cv=kFold, verbose = 1, n_jobs=jobs)
    y_pred_timeseries = timeSeries.undo_linear_standarlisation(timeSeries.from_mlf_to_time_series(y_pred))
    
    #y_pred_timeseries = timeSeries.from_mlf_to_time_series(y_pred)
    
    if (timeSeries != dataOriginal):
        y_pred_timeseries = dataOriginal.un_diff(y_pred_timeseries.values)
    pairedSeries = pd.DataFrame(data={'pred':y_pred_timeseries.values,'actual':dataOriginal.data.values},index=dataOriginal.data.index)
    cleanpairedSeries = pairedSeries.dropna()
    rmse = TimeSeries.root_mean_squared_error(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values)
    rel_error = TimeSeries.relative_error(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values)
    wmape = TimeSeries.weighted_mean_absolute_percentage_error(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values)
    r_2 = TimeSeries.adjusted_R2(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values,lags)
    yield 'Linear Model (wmape={:.3f}'.format(wmape)+' rmse={:.3f}'.format(rmse)+' rel. error={:.3f}'.format(rel_error)+' r^2={:.3f})'.format(r_2), dataOriginal.data, y_pred_timeseries

    # Train using a polynomial kernel
    svr = SVR(kernel='poly', degree=2)
    y_pred = cross_val_predict(svr, data_norm[:,0:-1], data_norm[:,-1], cv=kFold, verbose = 1, n_jobs=jobs)
    y_pred_timeseries = timeSeries.undo_linear_standarlisation(timeSeries.from_mlf_to_time_series(y_pred))
    
    #y_pred_timeseries = timeSeries.from_mlf_to_time_series(y_pred)
    
    if (timeSeries != dataOriginal):
        y_pred_timeseries = dataOriginal.un_diff(y_pred_timeseries.values)
    pairedSeries = pd.DataFrame(data={'pred':y_pred_timeseries.values,'actual':dataOriginal.data.values},index=dataOriginal.data.index)
    cleanpairedSeries = pairedSeries.dropna()
    rmse = TimeSeries.root_mean_squared_error(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values)
    rel_error = TimeSeries.relative_error(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values)
    wmape = TimeSeries.weighted_mean_absolute_percentage_error(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values)
    r_2 = TimeSeries.adjusted_R2(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values,lags)
    yield 'Polynomial Model(wmape={:.3f}'.format(wmape)+' rmse={:.3f}'.format(rmse)+' rel. error={:.3f}'.format(rel_error)+' r^2={:.3f})'.format(r_2), dataOriginal.data, y_pred_timeseries
    # Train using a knn
    knn = neighbors.KNeighborsRegressor(n_neighbors = 10)
    y_pred = cross_val_predict(knn, data[:,0:-1], data[:,-1], cv=kFold, verbose = 1, n_jobs=jobs)
    y_pred_timeseries = (timeSeries.from_mlf_to_time_series(y_pred))
    #y_pred_timeseries = timeSeries.from_mlf_to_time_series(y_pred)
    if (timeSeries != dataOriginal):
        y_pred_timeseries = dataOriginal.un_diff(y_pred_timeseries.values)
    pairedSeries = pd.DataFrame(data={'pred':y_pred_timeseries.values,'actual':dataOriginal.data.values},index=dataOriginal.data.index)
    cleanpairedSeries = pairedSeries.dropna()
    rmse = TimeSeries.root_mean_squared_error(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values)
    rel_error = TimeSeries.relative_error(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values)
    wmape = TimeSeries.weighted_mean_absolute_percentage_error(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values)
    r_2 = TimeSeries.adjusted_R2(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values,lags)
    yield 'KNN Regressor (wmape={:.3f}'.format(wmape)+' rmse={:.3f}'.format(rmse)+' rel. error={:.3f}'.format(rel_error)+' r^2={:.3f})'.format(r_2), dataOriginal.data, y_pred_timeseries
    
    # Train using MLP
    from sklearn.neural_network import MLPRegressor
    mlp = MLPRegressor()
    y_pred = cross_val_predict(mlp, data_norm[:,-20:-1], data_norm[:,-1], cv=kFold, verbose = 1, n_jobs=jobs)
    y_pred_timeseries = timeSeries.undo_linear_standarlisation(timeSeries.from_mlf_to_time_series(y_pred))
    y_pred_timeseries = timeSeries.from_mlf_to_time_series(y_pred)
    if (timeSeries != dataOriginal):
        y_pred_timeseries = dataOriginal.un_diff(y_pred_timeseries.values)
    pairedSeries = pd.DataFrame(data={'pred':y_pred_timeseries.values,'actual':dataOriginal.data.values},index=dataOriginal.data.index)
    cleanpairedSeries = pairedSeries.dropna()
    rmse = TimeSeries.root_mean_squared_error(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values)
    rel_error = TimeSeries.relative_error(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values)
    wmape = TimeSeries.weighted_mean_absolute_percentage_error(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values)
    r_2 = TimeSeries.adjusted_R2(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values,lags)
    yield 'MLP Regressor (wmape={:.3f}'.format(wmape)+' rmse={:.3f}'.format(rmse)+' rel. error={:.3f}'.format(rel_error)+' r^2={:.3f})'.format(r_2), dataOriginal.data, y_pred_timeseries
    
    #
    # Train using AR
    from statsmodels.tsa.ar_model import AR
    
    y_pred = np.empty(len(data))
    y_pred[:] = np.NaN
    lag = 5
    
    for train,test in kFold.split(data_norm[:,0:-1]):
        dataWithNan=np.empty(len(data_norm))
        dataWithNan[:] = np.NaN
        dataWithNan[train] = data_norm[train,-1]
        trainingTimeSeries = timeSeries.from_mlf_to_time_series(dataWithNan)
        
        
        model = AR(trainingTimeSeries,missing='drop')
        modelfit = model.fit(trend = 'nc', maxlag=lag)
        
        testData = []
        testData.extend(data_norm[test[0],-(lag+1):-1])
        testData.extend(data_norm[test,-1])
        y_pred[test]=ml_arima.predictAR(modelfit,testData)[-len(test):]
    y_pred_timeseries = timeSeries.undo_linear_standarlisation(timeSeries.from_mlf_to_time_series(y_pred))        
    #y_pred_timeseries = timeSeries.from_mlf_to_time_series(y_pred)
    if (timeSeries != dataOriginal):
        y_pred_timeseries = dataOriginal.un_diff(y_pred_timeseries.values)
    pairedSeries = pd.DataFrame(data={'pred':y_pred_timeseries.values,'actual':dataOriginal.data.values},index=dataOriginal.data.index)
    cleanpairedSeries = pairedSeries.dropna()
    try:
        rmse = TimeSeries.root_mean_squared_error(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values)
    except:
        rmse = np.NaN
    try:
        rel_error = TimeSeries.relative_error(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values)
    except:
        rel_error = np.NaN
    try:
        wmape = TimeSeries.weighted_mean_absolute_percentage_error(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values)
    except:
        wmape = np.NaN
    try:
        r_2 = TimeSeries.adjusted_R2(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values,lags)
    except:
        r_2 = np.NaN
    yield 'AR Regressor (wmape={:.3f}'.format(wmape)+' rmse={:.3f}'.format(rmse)+' rel. error={:.3f}'.format(rel_error)+' r^2={:.3f})'.format(r_2), dataOriginal.data, y_pred_timeseries
    
    #
    # Train using MA
    from statsmodels.tsa.arima_model import ARMA
    
    y_pred = np.empty(len(data))
    y_pred[:] = np.NaN
    lag = 2
    
    for train,test in kFold.split(data_norm[:,0:-1]):
        dataWithNan=np.empty(len(data_norm))
        dataWithNan[:] = np.NaN
        dataWithNan[train] = data_norm[train,-1]
        trainingTimeSeries = timeSeries.from_mlf_to_time_series(dataWithNan)
        
        
       
        
        model = ARMA(endog=trainingTimeSeries,order=(0,lag),missing='drop')
        modelfit = model.fit(trend='nc', method='css', transparams = False, full_outputbool = True, solver = 'bfgs')
        
        testData = []
        testData.extend(data_norm[test[0],-(lag+1):-1])
        testData.extend(data_norm[test,-1])
        y_pred[test]=ml_arima.predictMA(modelfit,testData)[-len(test):]
    y_pred_timeseries = timeSeries.undo_linear_standarlisation(timeSeries.from_mlf_to_time_series(y_pred))        
    #y_pred_timeseries = timeSeries.from_mlf_to_time_series(y_pred)
    if (timeSeries != dataOriginal):
        y_pred_timeseries = dataOriginal.un_diff(y_pred_timeseries.values)
    pairedSeries = pd.DataFrame(data={'pred':y_pred_timeseries.values,'actual':dataOriginal.data.values},index=dataOriginal.data.index)
    cleanpairedSeries = pairedSeries.dropna()
    try:
        rmse = TimeSeries.root_mean_squared_error(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values)
    except:
        rmse = np.NaN
    try:
        rel_error = TimeSeries.relative_error(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values)
    except:
        rel_error = np.NaN
    try:
        wmape = TimeSeries.weighted_mean_absolute_percentage_error(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values)
    except:
        wmape = np.NaN
    try:
        r_2 = TimeSeries.adjusted_R2(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values,lags)
    except:
        r_2 = np.NaN
    #print('MA Regressor (wmape={:.3f}'.format(wmape)+' rmse={:.3f}'.format(rmse)+' rel. error={:.3f}'.format(rel_error)+' r^2={:.3f})'.format(r_2))
    yield 'MA Regressor (wmape={:.3f}'.format(wmape)+' rmse={:.3f}'.format(rmse)+' rel. error={:.3f}'.format(rel_error)+' r^2={:.3f})'.format(r_2), dataOriginal.data, y_pred_timeseries
    
    #
    # Train using ARMA
    from statsmodels.tsa.arima_model import ARMA
    
    #Create a array with the same size of the full set of nan
    #Apply the values of training on this array
    #Apply from ml to ts on this array with nans 
    #Test on the normal test set getting just the last forecasting.
    
    y_pred = np.empty(len(data))
    y_pred[:] = np.NaN
    lag = 1
    lag_ar = 4
    for train,test in kFold.split(data_norm[:,0:-1]):
        dataWithNan=np.empty(len(data_norm))
        dataWithNan[:] = np.NaN
        dataWithNan[train] = data_norm[train,-1]
        trainingTimeSeries = timeSeries.from_mlf_to_time_series(dataWithNan)
        
        model = ARMA(endog=trainingTimeSeries,order=(lag_ar,lag),missing='drop')
        modelfit = model.fit(trend='nc', method='css', transparams = False, full_outputbool = True, solver = 'bfgs')
        
        testData = []
        testData.extend(data_norm[test[0],-(max(lag,lag_ar)+1):-1])
        testData.extend(data_norm[test,-1])
        y_pred[test]=ml_arima.predictARMA(modelfit,testData)[-len(test):]
    y_pred_timeseries = timeSeries.undo_linear_standarlisation(timeSeries.from_mlf_to_time_series(y_pred))        
    #y_pred_timeseries = timeSeries.from_mlf_to_time_series(y_pred)
    if (timeSeries != dataOriginal):
        y_pred_timeseries = dataOriginal.un_diff(y_pred_timeseries.values)
    pairedSeries = pd.DataFrame(data={'pred':y_pred_timeseries.values,'actual':dataOriginal.data.values},index=dataOriginal.data.index)
    cleanpairedSeries = pairedSeries.dropna()
    try:
        rmse = TimeSeries.root_mean_squared_error(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values)
    except:
        rmse = np.NaN
    try:
        rel_error = TimeSeries.relative_error(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values)
    except:
        rel_error = np.NaN
    try:
        wmape = TimeSeries.weighted_mean_absolute_percentage_error(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values)
    except:
        wmape = np.NaN
    try:
        r_2 = TimeSeries.adjusted_R2(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values,lags)
    except:
        r_2 = np.NaN
    #print('MA Regressor (wmape={:.3f}'.format(wmape)+' rmse={:.3f}'.format(rmse)+' rel. error={:.3f}'.format(rel_error)+' r^2={:.3f})'.format(r_2))
    yield 'ARMA Regressor (wmape={:.3f}'.format(wmape)+' rmse={:.3f}'.format(rmse)+' rel. error={:.3f}'.format(rel_error)+' r^2={:.3f})'.format(r_2), dataOriginal.data, y_pred_timeseries
    '''
    
    
    # Train using LSTM
    from scikit_lstm import LSTMRegressor
    from scikit_lstm import LSTM_Model
    #import scikit_lstm
    
    sequence_lenght = lags
    lstm = LSTM_Model(128).generate_model
    regressor = LSTMRegressor(lstm)
    
    param = dict(validation_split = 0.1, epochs = 150, batch_size=200, shuffle=True)
                 
    y_pred = cross_val_predict(regressor, data_norm[:,0:-1], data_norm[:,-1], cv=kFold, verbose = 1, fit_params = param, n_jobs=1)
    y_pred_timeseries = timeSeries.undo_linear_standarlisation(timeSeries.from_mlf_to_time_series(y_pred))
    #y_pred_timeseries = timeSeries.from_mlf_to_time_series(y_pred)
    if (timeSeries != dataOriginal):
        y_pred_timeseries = dataOriginal.un_diff(y_pred_timeseries.values)
    pairedSeries = pd.DataFrame(data={'pred':y_pred_timeseries.values,'actual':dataOriginal.data.values},index=dataOriginal.data.index)
    cleanpairedSeries = pairedSeries.dropna()
    rmse = TimeSeries.root_mean_squared_error(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values)
    rel_error = TimeSeries.relative_error(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values)
    wmape = TimeSeries.weighted_mean_absolute_percentage_error(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values)
    r_2 = TimeSeries.adjusted_R2(cleanpairedSeries['actual'].values, cleanpairedSeries['pred'].values,lags)
    yield 'LSTM Regressor (wmape={:.3f}'.format(wmape)+' rmse={:.3f}'.format(rmse)+' rel. error={:.3f}'.format(rel_error)+' r^2={:.3f})'.format(r_2), dataOriginal.data, y_pred_timeseries
    
    
#'''
    
# =====================================================================


def plot(results, dataName, outputName):
    '''
    Create a plot comparing multiple learners.

    `results` is a list of tuples containing:
        (title, expected values, actual values)
    
    All the elements in results will be plotted.
    '''
    names = []
    for (a,b,c) in results:
        names.append(a)
    subplots = make_subplots(rows=len(results),cols=1, subplot_titles = names, shared_xaxes=True, shared_yaxes=True)
    
    i = 1
    for (title, y, y_pred) in results:
        y_trace = go.Scatter(x=y.index, y=y.values, name='actual', line_color = 'deepskyblue', opacity = 0.8)
        predict_trace = go.Scatter(x=y_pred.index, y=y_pred.values, name='predicted', line_color = 'darkmagenta', opacity = 0.8)
        subplots.add_trace(y_trace, row = i, col = 1)
        subplots.add_trace(predict_trace, row = i, col = 1)
        i = i+1
    subplots.update_layout(title_text = 'Predictions of ' + dataName, autosize = True)
    
    subplots.write_html(outputName, auto_open=True)

if __name__ == '__main__':
    # Download the data set from URL
    #print("Downloading data from {}".format(URL))
    #frame = download_data()
    #import os
   # os.chdir(os.path.dirname(__file__))
    fileName= 'datasetWindWave.csv'
    print("load data from:",fileName)
    frame= load_data(fileName)
    # Process data into feature and label array
    print("Processing {} samples with {} attributes".format(len(frame.index), len(frame.columns)))
    
    
    series_wwsh=TimeSeries(frame['darwinWaves_WWSH'])
    #series_wwsh.short_analysis()
    print('term size = 60 and term h = 100 for WWSH applying all analysis after a differenciation')
    
    
    diff_wwsh = TimeSeries(series_wwsh.diff())
    #diff_wwsh.short_analysis()
    # Evaluate multiple regression learners on the data
    print("Evaluating regression learners")
    for i in range(63,100):
        lags=i
        results = list(evaluate_learner(dataOriginal=series_wwsh,stationaryData=diff_wwsh,n_splits=5,gap_ini=100, gap_end=100,lags=lags))
        results.append((show_cross_validation(dataOriginal=series_wwsh,stationaryData=diff_wwsh,n_splits=5,gap_ini=100, gap_end=100,lags=lags)))
    # Display the results
        print("Plotting the results")
        plot(results, 'Wind Wave Significant Height order'+str(lags),'WWSH'+str(lags)+'.html')
