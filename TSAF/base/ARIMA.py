#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 15:23:06 2020
This class is a man in the middle, making possible to deal with tsa.ARIMA models on the machine learning fashion.
@author: viniciussantino
"""

from statsmodels.tsa.arima_model import ARMAResults
from statsmodels.tsa.ar_model import ARResults
import numpy as np

def predict_AR(coef, history):
    yhat = 0.0
    for i in range(1, min(len(coef),len(history))+1):
        yhat += coef[i-1] * history[-i]
    return yhat


def predictAR(model:ARResults,data):
    predict = []
    for j in range(len(data)-len(model.params)+1):
        #predict.append(model.params[0])
        lags = []
        lags= data[j:(j+len(model.params))]
        predict.append(predict_AR(model.params, lags))
    return(predict)
    


def predictMA(model:ARMAResults,data):
    error = []
    prediction = []
    error_ = []
    k = model.k_ma
    for j in range(len(data)+1):
        if j == 0:
            error.append(data[j])
            prediction.append(0.0)
        elif j < k:
            error_ = error[0:j]
            prediction.append(predict_AR(model.maparams,error_))
            if(j<len(data)):
                error.append(data[j]-prediction[-1])
        else:
            error_ = error[j-k:]
            prediction.append(predict_AR(model.maparams,error_))
            if(j<len(data)):
                error.append(data[j]-prediction[-1])
                '''
        if j < 20: 
            print("predictions original:", model.predict(start=j,end=j))
            print("residual - error: ",model.resid[j:j+len(model.params)] - error)
            print("residual: ",model.resid[j:j+len(model.params)])
            print("error: ",error)
            print("predictions: ",prediction[-1])
            print("model params: ",model.params)
            print("data: ", data[j:j+len(model.params)])
            print(data[j+1:j+len(model.params)+1][0] - prediction[-1])
            print("----------------")
            '''
            
    return prediction[k:]
            
            
            
def predictARMA(model:ARMAResults,data):
    error = []
    prediction = []
    error_ = []
    history = []
    k = model.k_ma
    l = model.k_ar
    index_Nan = 0
    for j in range(len(data)+1):
        if j < l:
            prediction.append(0.0)
            error.append(0.0)
        else:
            if j < k:
                error_ = error[0:j]
            else:
                error_ = error[j-k:]
            if j < l:
                history = data[0:j]
            else:
                history = data[j-l:j]
            prediction.append(predict_AR(model.maparams,error_)+predict_AR(model.arparams,history))    
            if(j<len(data)):
                if np.isnan(data[j]) or np.isnan(prediction[-1]):
                    error.append(0)
                else:
                    error.append(data[j]-prediction[-1])
        '''
        if j <10 :
            print("j value: ",j)
            if  j >=l:
                print("predictions original:", model.predict(start=j,end=j))
            #print("residual - error: ",model.resid[max(0,j-len(model.maparams)):j] - error_)
            print("residual: ",model.resid[max(0,j-len(model.maparams)):j])
            print("error: ",error_)
            print("predictions: ",prediction[-1])
            print("ma params: ",model.maparams)
            print("ar params: ",model.arparams)
            print("data: ", data[:j+1])
            #print(data[j+1:j+len(model.params)+1][0] - predictma[-1])
            print("----------------")
           ''' 
    return prediction[max(k,l):]