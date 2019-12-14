""" Common tools for NN """

import numpy as np

def sigmoid(z):
    return 1. / (1. + np.exp(-z))

def dsigm(s):
    return s * (1 - s)
    
def xentropy(a, y):
    return -np.mean(y * np.log(a) + (1. - y) * np.log(1. - a), axis=0)[0]

def appone(x):
    return np.column_stack([x, np.ones(x.shape[0])])

def lincomb(x, w):
    return np.dot(appone(x), w)

def inirand(ri, rj, coef=2):
    return coef*(np.random.rand(ri, rj) - 0.5)
