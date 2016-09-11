"""
This is a learning of LISA Theano multi-layer perception model
"""

from __future__ import print_function

__docformat__ = 'restructedtext en'


import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T


from logistic_sgd import LogisticRegression, load_data
# very interesting, you can import classes (LogisticRegression) and functions (loda_data) 
# from customer written modules (logistic_sgy.py)

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None,b=None,
                 activation=T.tanh):
        self.input = input # member variable
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in,n_out)
                    ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)
            
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
            
        self.W = W
        self.b = b