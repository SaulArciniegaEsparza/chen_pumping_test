# -*- coding: utf-8 -*-
"""
CHEN pumping test analysis
Drawdown derivative methods

Functions:
    derivative_k   > Drawdown derivative using k-neighbours
"""

import numpy as _np
from scipy.interpolate import splev as _splev

from drawdown import _data_validation
import linear_regression as _solvers
import derivative as _derivative
from BSpline import BSFK


"""_________________________ SMOOTH DATA ___________________________________"""


# Smoothing data using k-neighbour approximation
# Inputs
#   x        [list, tuple, ndarray] time data
#   y        [list, tuple, ndarray] drawdown data
#   k        [int] window offset [i-k,i+k]
# method     [int] smoothing method
#             0   moving average approximation
#             1   multi-linear regression
#             2   L1 multi-linear regression
# Outputs
#   out      [dict] output smoothed data {t: time, s: drawdown}
def simple_smooth(x, y, k=1, method=0):
    # Check inputs
    x, y = _data_validation.flow_conversion(x, y, scale='radial')
    k, method = int(k), int(method)

    assert 1 < k < len(x), 'Parameter k must be higher than 1'
    assert 0 <= method <= 2, 'Wrong smoothing method'

    m = len(x)
    x_smooth = _np.zeros(m, dtype=_np.float32)
    y_smooth = _np.zeros(m, dtype=_np.float32)

    # Main loop
    for i in xrange(m):
        n1, n2 = i - k, i + k
        if n1 <= 0:  # restrict left elements
            n1 = 0
        if n2 >= m:  # restrict right elements
            n2 = m-1
        # get data
        x_range, y_range = x[n1:(n2 + 1)], y[n1:(n2 + 1)]
        if method == 0:    # moving average
            xs = x_range.mean()
            ys = y_range.mean()
        elif method == 1:  # multi-linear regression
            coef = _solvers.MLR(x_range, y_range)
            xs = x[i]
            ys = coef[0] + coef[1] * x[i]
        elif method == 2:  # L1 spline approximation
            coef = _solvers.L1MLR(x_range, y_range)
            xs = x[i]
            ys = coef[0] + coef[1] * x[i]
        # save data
        x_smooth[i] = xs
        y_smooth[i] = ys

    # Convert outputs
    x_smooth = 10. ** x_smooth
    out = {'t': x_smooth,
           's': y_smooth}
    return(out)  # End of Function


# Constrained Quadratic B-Splines with Free-Knots (CQBSFK)
# Input
#   X        (array) time vector
#   Y        (array) drawdown vector
#   k        shape factor. If k=0 (default) 5 internal knots are used as initial
#            approach. Ik k=1 10 internal knots are used.
#   L        [optional] interval of evaluation
#   dB       [optional] first and second derivate list/tuple [derivate_1,derivate_2]. If dB=[]
#            (default), first and second derivate are computed using the Bourdet L1MLR
#            method with L
#   simply   [optional, integer]  number of elements of the output vector as a percentaje
#            the input lenght arrays
#  nsmooth
# Output
#   out      dictionary with X (new time vector), Y smoothed data with Spline,
#            Y1 and Y2 are the first and second derivates
def CQBSFK(x, y, shape=0, l=0.5, dB=None):
    # Check inputs
    x, y = _data_validation.flow_conversion(x, y, scale='none')
    logx = _np.log10(x)

    shape, l = int(shape), float(l)
    assert 0 <= shape <= 1, "Wrong shape parameter value!"

    # Get derivatives if dB is None
    if dB is None:
        dB = []
    elif type(dB) in (list, tuple):
        if len(dB) == 2:  # derivative was input
            dB1 = _data_validation.to_ndarray(dB[0])
            dB2 = _data_validation.to_ndarray(dB[1])
            if dB1.size != dB2.size:
                raise TypeError('dB input must be a 2 element list with'
                                'numpy arrays with the same length!')
        else:
            dB = []
    else:
        raise TypeError('Wrong type for dB parameter <{}>'.format(str(type(dB))))

    if len(dB) == 0:  # default derivative derivative_l(x, y, l, method=2)
        dB1 = _derivative.derivate_l(x, y, l, method=2)
        dB2 = _derivative.derivate_l(x, dB1, l, method=2)

    # Parameters and constraints
    k = 5
    if shape == 0:  # few data
        nknots = 5
        # Set constraints
        con = [{'p': 1,  # first derivative
                'x': [logx[0], logx[-1]],
                'v': [dB1[0], dB1[-1]]},
               {'p': 2,  # second derivative
                'x': [logx[0], logx[-1]],
                'v': [dB2[0], dB2[-1]]}
               ]

    else:  # long data
        nknots = 10
        # Set constraints
        con = [{'p': 1,  # first derivative
                'x': [logx[0], logx[1], logx[-1]],
                'v': [dB1[0], dB1[1], dB1[-1]]},
               {'p': 2,  # second derivative
                'x': [logx[0], logx[1], logx[-1]],
                'v': [dB2[0], dB2[1], dB2[-1]]}
               ]
    pp, flag = BSFK(logx, y, k, nknots, con)

    if flag > 0:
        y_smooth = _splev(logx, pp, der=0)
        y_der1 = _splev(logx, pp, der=1)
        y_der2 = _splev(logx, pp, der=2)
        out = {'t': x,
               's': y_smooth,
               's1': y_der1,
               's2': y_der2,
               'spline': pp}
    else:
        out = dict(t=[], s=[], s1=[], s2=[], spline=None)
    return(out)  # End Function

