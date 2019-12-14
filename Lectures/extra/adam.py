#! /usr/bin/env python3

""" Gradient descent with Adaptive Moment Estimation (ADAM) """

import numpy as np

class Adam(object):
    """ Adaptive Moment Estimation for gradient descent """
    def __init__(self, beta1=0.99, beta2=0.999, eps=10**-8):
        self.b1, self.a1 = beta1, 1-beta1
        self.b2, self.a2 = beta2, 1-beta2
        self.e = eps
        self.m = 0
        self.v = 0

    def __call__(self, g):
        """ Calculates moments for a given gradient vector """
        self.m = self.b1 * self.m + self.a1 * g
        self.v = self.b2 * self.v + self.a2 * g**2
        return self.m / self.a1 / (np.sqrt(self.v / self.a2) + self.e)
