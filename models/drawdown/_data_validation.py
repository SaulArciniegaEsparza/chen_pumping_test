# -*- coding: utf-8 -*-
"""
CHEN pumping test analysis
Data validation for models and units conversions

Functions:
    to_ndarray        > Data conversion to numpy array with float32 as data type
    flow_conversion   > Check input data for pumping test analysis and apply flow conversions
"""

import numpy as _np


"""_______________________ DATA VALIDATION _________________________________"""


# Data conversion to numpy array with float32 as data type
# Inputs:
#  x     [int, float, list, tuple, ndarray] input data
# Outputs
#  x     [ndarray] output data
def to_ndarray(x):
    # Check input data types
    if type(x) in [int, float]:
        x = _np.array([x], dtype=_np.float32)
    if type(x) in [list, tuple]:
        x = _np.array(x, dtype=_np.float32)
    elif type(x) is _np.ndarray:
        if x.dtype not in [_np.float, _np.float16, _np.float32, _np.float64]:
            x = x.astype(_np.float32)
    else:
        raise TypeError('x must be a list, tuple or a numpy array.')
    return(x)  # validate()


# Check input data for pumping test analysis and apply flow conversions
# Inputs
#   x    [list, tuple, ndarray] time data
#   y    [list,tuple, ndarray] drawdown data
# scale  [string] scale or flow type
#         'none' linear scale, t=X s=Y
#         'loglog' time and drawdown base 10 logarithmic, t=log10(X) s=log10(Y)
#         'log' or 'radial' radial flow conversion, t=log10(X) s=Y
#         'linear' linear flow conversion, t=log10(X^0.5) s=log10(Y)
#         'bilinear' bilinear flow conversion, t=log10(X^0.25) s=log10(Y)
#         'spherical' spherical flow conversion, t=log10(X^-0.5) s=log10(Y)
# Output
#   t    [ndarray] converted time
#   s    [ndarray] converted drawdown
def flow_conversion(x, y, scale='none'):
    # Check input data types
    x = to_ndarray(x)
    y = to_ndarray(y)
    
    # Check dimensions
    if x.ndim != 1 and y.ndim != 1:
        raise TypeError('x and y must be a one-dimensional array.')
    elif x.ndim != 1:
        raise TypeError('x must be a one-dimensional array.')
    elif y.ndim != 1:
        raise TypeError('y must be a one-dimensional array.')
    
    if x.size != y.size:
        raise Exception('x and y must have the same length.')
    
    # Check scale method
    scale = scale.lower()
    
    # Convert data types
    if scale == 'none':
        pass
    elif scale == 'loglog':
        t = _np.log10(x)
        s = _np.log10(y)
    elif scale in ['log', 'radial']:
        t = _np.log10(x)
        s = y.copy()
    elif scale == 'linear':
        t = _np.log10(x ** 0.5)
        s = _np.log10(y)
    elif scale == 'bilinear':
        t = _np.log10(x ** 0.25)
        s = _np.log10(y)
    elif scale == 'spherical':
        t = _np.log10(x ** -0.5)
        s = _np.log10(y)
    else:
        raise AssertionError('Bad argument for scale parameter.')
    
    return(t, s)  # flow_conversion()



