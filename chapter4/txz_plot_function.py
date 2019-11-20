#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 09:40:15 2019

@author: daniel
"""

import matplotlib.pyplot as plt

"""
x: 1D array input
y: 1D array input
"""
def xy2D_relation_1(x, y, legendname = ['data'], xname = 'X', yname = 'Y', size = (10, 7), style = 'o-', savedir = 'plt.pdf'):
    plt.figure(figsize = size)
    xplot = plt.plot(x, y, style)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.legend(xplot, legendname)
    plt.savefig(savedir)
    

def model_weight_plot(model, xname = '', yname = '', layerid = 0, size = (10, 7), savedir = 'model.jpg'):
    plt.figure(figsize = size)
    for i in range(model[layerid].weight.size()[0]):
        weights = model[layerid].weight[i].data.numpy()
        plt.plot(weights, alpha = 0.5, label = i)
    plt.legend()
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.savefig(savedir)