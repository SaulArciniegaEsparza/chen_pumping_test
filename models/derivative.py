# -*- coding: utf-8 -*-
"""
CHEN pumping test analysis
Drawdown derivative methods

Functions:
    derivative_k   > Drawdown derivative using k-neighbours
    derivative_l   > Drawdown derivative using L conditional

    L1MLR          > L1 Multi-linear regression
    MLR            > Multi-linear regression
"""

import numpy as _np
from drawdown import _data_validation
import solvers as _solvers


"""_______________________ DRAWDOWN METHODS ________________________________"""


# Drawdown derivative using k-neighbours
# Inputs
#   x        [list, tuple, ndarray] time vector
#   y        [list, tuple, ndarray] drawdown vector
#   k        [int] number of elements used for derivative estimation [i-K,i+k]
#   method   [int] derivative method
#             0  Bourdet (default)
#             1  Spane using MLR
#             2  Spane using L1MLR
# Outputs
#   dl       [ndarray] drawdown derivative
def derivative_k(x, y, k=1, method=0):
    # Check inputs
    x, y = _data_validation.flow_conversion(x, y, scale='radial')
    k, method = int(k), int(method)

    assert 1 < k < len(x), 'Parameter k must be higher than 1'
    assert 0 <= method <= 2, 'Wrong derivative method'

    n = len(x)
    dl = _np.zeros(n, dtype=_np.float32)  # derivative vector

    # Main loop
    for i in xrange(n):
        k1, k2 = i - k, i + k
        if k1 <= 0:  # restrict left elements
            k1 = 0
        if k2 >= n:  # restrict right elements
            k2 = n-1

        # Left derivative
        if i != 0:  # verify internal node
            xl = x[k1:(i + 1)]
            yl = y[k1:(i + 1)]
            dx1 = xl[-1] - xl[0]  # left log time difference
            if method == 0:
                ds1 = (yl[-1] - yl[0]) / dx1  # left slope using Bourdet method
            elif method == 1:
                ds1 = _solvers.MLR(xl, yl, False)[1]   # left slope using Spane method
            elif method == 2:
                ds1 = _solvers.L1MLR(xl, yl)[1]        # left slope using Spane modified method

        # Right derivative
        if i != n - 1:  # verify internal node
            xr = x[i:(k2+1)]
            yr = y[i:(k2+1)]
            dx2 = xr[-1] - xr[0]  # right log time difference
            if method == 0:
                ds2 = (yr[-1] - yr[0]) / dx2  # right slope using Bourdet method
            elif method == 1:
                ds2 = _solvers.MLR(xr, yr, False)[1]   # right slope using Spane method
            elif method == 2:
                ds2 = _solvers.L1MLR(xr, yr)[1]        # right slope using Spane modified method

        # Total derivative
        if i == 0:         # left extreme derivative
            dl[i] = ds2
        elif i == n - 1:   # right extreme derivative
            dl[i] = ds1
        else:              # central node derivative
            dl[i] = (ds1 * dx2 + ds2 * dx1) / (dx1 + dx2)

        return(dl)  # End of Function


# Drawdown derivative using L conditional
# Inputs
#   x        [list, tuple, ndarray] time vector
#   y        [list, tuple, ndarray] drawdown vector
#   l        [float] interval of evaluation (log time)
#   method   [int] derivative method
#             0  Bourdet (default)
#             1  Spane using MLR
#             2  Spane using L1MLR
# Outputs
#   dl       [ndarray] drawdown derivative
def derivative_l(x, y, l=1.0, method=0):
    # Check inputs
    x, y = _data_validation.flow_conversion(x, y, scale='radial')
    l, method = float(l), int(method)

    assert 0 < l, 'Parameter l must be higher than 0'
    assert 0 <= method <= 2, 'Wrong derivative method'

    n = len(x)
    dl = _np.zeros(n, dtype=_np.float32)  # derivative vector

    # Compute first derivative
    dl[0] = (y[1] - y[0]) / (y[1] - y[0])  # forward finite difference
    # Compute last derivative
    dxr = _np.abs(x[-1] - x)  # right difference
    # get values that satisfy the L-condition
    xr = x[dxr <= l]
    yr = y[dxr <= l]
    # left log time difference (fixed for right extreme)
    dxd = xr[-1] - xr[0]

    # pseudo right-derivative computation
    if method == 0:    # Bourdet method
        prd = (yr[-1] - yr[0]) / dxd
    elif method == 1:  # Spane method
        prd = _solvers.MLR(xr, yr, False)[1]
    elif method == 2:  # Spane modified method
        prd = _solvers.L1MLR(xr, yr)[1]
    dl[-1] = prd  # store last derivative

    # Compute internal derivative
    for i in xrange(1, n - 1):  # main loop
        # Left derivative
        dxl = _np.abs(x[i] - x[:(i + 1)])  # left difference
        posl = dxl <= l  # left conditional
        if posl.sum() > 1:  # internal left node
            # get values that satisfy the L-condition
            xl = x[:(i + 1)][posl]
            yl = y[:(i + 1)][posl]
            dx1 = xl[-1] - xl[0]  # left log time difference
            if method == 0:
                ds1 = (yl[-1] - yl[0]) / dx1   # left slope using Bourdet method
            elif method == 1:
                ds1 = _solvers.MLR(xl, yl, False)[1]    # left slope using Spane method
            elif method == 2:
                ds1 = _solvers.L1MLR(xl, yl)[1]         # left slope using Spane modified method

        # Right derivative
        dxr = _np.abs(x[i] - x[i:])  # right difference
        posr = dxr <= l  # right conditional
        if posr.sum() > 1 and not posr[-1]:  # internal right node
            # get values that satisfy the L-condition
            xr = x[i:][posr]
            yr = y[i:][posr]
            dx2 = xr[-1] - xr[0]  # left log time difference
            if method == 0:
                ds2 = (yr[-1] - yr[0]) / dx2  # right slope using Bourdet method
            elif method == 1:
                ds2 = _solvers.MLR(xr, yr, False)[1]   # right slope using Spane method
            elif method == 2:
                ds2 = _solvers.L1MLR(xr, yr)[1]        # right slope using Spane modified method
        elif posr.sum() > 1 and posr[-1]:     # internal right node with fixed values
            dx2 = dxd  # fixed right difference
            ds2 = prd  # fixed right derivative

        # Calculate total derivative
        dl[i] = (ds1 * dx2 + ds2 * dx1) / (dx1 + dx2)

    return(dl)  # End Function

