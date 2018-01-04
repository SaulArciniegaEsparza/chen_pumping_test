# -*- coding: utf-8 -*-
"""
CHEN pumping test analysis
Data validation for models and units conversions

Functions:
    to_ndarray        > Data conversion to numpy array with float32 as data type
    flow_conversion   > Check input data for pumping test analysis and apply flow conversions
    units_conversion  > Units conversion for time, drawdown and pumping rate
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


# Units conversion for time and drawdown
def units_conversion(x, in_unit='m3/day', out_unit='l/s'):
    # Check inputs
    in_unit = in_unit.lower().split('/')
    out_unit = out_unit.lower().split('/')

    n1, n2 = len(in_unit), len(out_unit)

    if n1 == 0 or n1 > 2:
        raise AssertionError('Bad argument as in_unit={}'.format(in_unit))
    if n2 == 0 or n2 > 2:
        raise AssertionError('Bad argument as out_unit={}'.format(out_unit))
    if n1 != n2:
        raise AssertionError("Units {} can't be converted to {}".format(in_unit, out_unit))

    # units data base
    len_factors = {'cm': {'cm': 1.,
                          'm':  1. / 100.,
                          'ft': 1. / 30.48,
                          'in': 1. / 2.54},
                   'm': {'cm': 100.,
                         'm':  1.,
                         'ft': 3.28084,
                         'in': 39.370079},
                   'ft': {'cm': 30.48,
                          'm':  1. / 3.2808,
                          'ft': 1.,
                          'in': 12.},
                   'in': {'cm': 2.54,
                          'm':  1. / 39.370079,
                          'ft': 1. / 12.,
                          'in': 1.}
                   }

    vol_factors = {'cm3': {'cm3': 1.,
                           'm3':  1. / 100. ** 3,
                           'ft3': 1. / 30.48 ** 3,
                           'in3': 1. / 2.54 ** 3,
                           'l':   1 / 1000.},
                   'm3': {'cm3': 100. ** 3,
                          'm3':  1.,
                          'ft3': 3.28084 ** 3,
                          'in3': 39.370079 ** 3,
                          'l':  1000.},
                   'ft3': {'cm3': 30.48 ** 3,
                           'm3':  1. / 3.2808 ** 3,
                           'ft3': 1.,
                           'in3': 12. ** 3,
                           'l': 28.3168},
                   'in3': {'cm3': 2.54 ** 3,
                           'm3':  1. / 39.370079 ** 3,
                           'ft3': 1. / 12. ** 3,
                           'in3': 1.,
                           'l':   1. / 61.0237},
                   'l': {'cm3': 1000.,
                         'm3':  1. / 1000.,
                         'ft3': 1. / 28.3168,
                         'in3': 61.0237,
                         'l':   1.}
                   }

    time_factors = {'s': {'s':   1.,
                          'min': 1. / 60.,
                          'hr':  1. / 3600.,
                          'day': 1 / 86400.},
                    'min': {'s':   60.,
                            'min': 1.,
                            'hr':  1. / 60.,
                            'day': 1. / 1440.},
                    'hr': {'s':   3600.,
                           'min': 60.,
                           'hr':  1.,
                           'day': 1. / 24.},
                    'day': {'s':   86400.,
                            'min': 1440.,
                            'hr':  24.,
                            'day': 1.}
                    }

    # Check available units
    if n1 == 2:  # volumetric and time conversion
        if in_unit[0] not in vol_factors or in_unit[1] not in time_factors:
            raise AssertionError("Bad input units {}".format(in_unit))
        if out_unit[0] not in vol_factors or out_unit[1] not in time_factors:
            raise AssertionError("Bad output units {}".format(out_unit))
        op = 0
    else:        # length or time conversion
        if in_unit[0] in len_factors and out_unit[0] in len_factors:
            op = 1
        elif in_unit[0] in time_factors and out_unit[0] in time_factors:
            op = 2
        else:
            raise AssertionError("Units {} can't be converted to {}".format(in_unit, out_unit))

    # Convert units
    if op == 0:    # volume
        xc = x * vol_factors[in_unit[0]][out_unit[0]] / time_factors[in_unit[1]][out_unit[1]]
    elif op == 1:  # length
        xc = x * len_factors[in_unit[0]][out_unit[0]]
    else:          # time
        xc = x * time_factors[in_unit[0]][out_unit[0]]

    return(xc)  # units_conversion()

