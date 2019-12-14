#! /usr/bin/env python3

""" 
    Sigmoid, one output neuron NN with training history feature. 
    Gradient descent is improved with momentum
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools

from nncomon import *
from adam import Adam

def pairs(v):
    for x, y in zip(v[:-1], v[1:]):
        yield (x, y)

class NNBinary(object):
    """ Neural network for binary classification """
    def __init__(self, r=None):
        if r is not None:
            self.r = r + [1]

    def nparams(self):
        """ Total number of parameters in NN """
        return [0] + list(itertools.accumulate(
            [(ri+1)*rj for ri, rj in pairs(self.r)]))

    def make_batches(self, x, y, bsize):
        """ Divide data into batches """
        if bsize >= self.N:
            return [[x, y]]
        batch = []
        edges = list(range(0, self.N+1, bsize))[:-1] + [self.N]
        for idx, jdx in pairs(edges):
            batch.append([x[idx:jdx], y[idx:jdx]])
        return batch

    def toFile(self, fname):
        np.save(fname, self.wl)

    def fromFile(self, fname):
        self.wl = np.load(fname, allow_pickle=True)

    def weight_views(self, w):
        return [w[ni:nj].reshape(ri+1, -1) for [ni, nj], ri in zip(pairs(self.npars), self.r)]

    def set_bias_mask(self):
        self.bias_mask = np.ones(self.wght.shape, dtype=bool)
        for nj, ri in zip(self.npars[1:], self.r):
            self.bias_mask[nj-ri:nj] = False

    def evaluate(self, x, y):
        a = x.copy()
        for w in self.wl:
            a = sigmoid(lincomb(a, w))
        return xentropy(a, y)

    def predict(self, x):
        for w in self.wl:
            x = sigmoid(lincomb(x, w))
        return (x > 0.5).astype(int)

    def forward(self, x):
        a = [x]
        for w in self.wl:
            a.append(sigmoid(lincomb(a[-1], w)))
        return a

    def train(self, x, y, eta, batch=100, eps=1.e-6, max_iter=10000, seed=None,
              x_test=None, y_test=None, early_stop=True, l2=None):
        if self.r[0] != x.shape[1]:
            self.r = [x.shape[1]] + self.r
            self.npars = self.nparams()
            print('{} parameters to fit'.format(self.npars[-1]))
        self.l2 = l2
        self.eta = eta
        self.early_stop = early_stop and x_test is not None
        self.N = x.shape[0]
        self.eps = eps
        self.err = False
        self.log = {key: [] for key in ['j', 'tj']}

        if seed:
            np.random.seed(seed)
        # allocate all weights
        self.wght = inirand(self.npars[-1], 1, coef=2)
        self.grad = np.empty(self.wght.shape)
        # make weight vies for each layer
        self.wl, self.gl = [self.weight_views(a) for a in [self.wght, self.grad]]
        self.set_bias_mask()

        def eval(check_condition=True):
            self.log['j'].append(self.evaluate(x, y))
            if x_test is not None:
                self.log['tj'].append(self.evaluate(x_test, y_test))
            if check_condition:
                return self.stop_conditions()
        eval(False)

        # mini-batches
        batches = self.make_batches(x, y, batch)
        print('{} batches'.format(len(batches)))
        # adaptive moments
        self.adam = Adam()

        cur_iter = 0
        while cur_iter < max_iter:
            if not cur_iter % 100:
                print('it: {:5d}, J: {:.4f}'.format(cur_iter, self.log['j'][-1]))
            for xi, yi in batches:
                self.process(xi, yi)
            if eval():
                break
            cur_iter += 1
        if cur_iter == max_iter:
            print('train: max number of iterations reached')
        return not self.err

    def process(self, xi, yi):
        ai = self.forward(xi)
        delta = ai[-1] - yi
        for a, w, g in zip(ai[::-1][1:], self.wl[::-1], self.gl[::-1]):
            g[:,:] = np.dot(appone(a).T, delta) / self.N
            delta = np.dot(delta, w[:-1].T) * dsigm(a)
        if self.l2 is not None:
            self.grad[self.bias_mask] += 2.*self.l2*self.wght[self.bias_mask]
        self.wght -= self.eta * self.adam(self.grad)

    def stop_conditions(self):
        if self.early_stop and self.log['tj'][-1] > self.log['tj'][-2]:
            print('early stop')
            return True
        if self.log['j'][-1] > self.log['j'][-2]:
            print('divergent fit')
            self.err = True
            return True
        if self.log['j'][-2] - self.log['j'][-1] < self.eps:
            if self.log['j'][-1] > 0.1:
                print('local minimum')
                self.err = True
            else:
                print('training completed')
            return True
        return False
