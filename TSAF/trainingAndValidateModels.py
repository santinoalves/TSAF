#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 13:20:37 2020

@author: viniciussantino
"""

def generateExperimentSlidingWindowsCV(data,parameters,models,partitions):
    windows_size =  int(len(data)/partitions+1)

    return None