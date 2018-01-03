# -*- coding: utf-8 -*-
"""
CHEN pumping test analysis


"""

import numpy as _np
from drawdown import data_validation as _data_validation


"""______________________ DRAWDOWN METHODS _________________________________"""


# Drawdown derivative using k-neighbours
# Inputs
#   X        [list, tuple, ndarray] time vector
#   Y        [list, tuple, ndarray] drawdown vector
#   k        [int] number of elements used for derivative estimation [i-K,i+k]
#   method   [int] derivative method
#             0  Bourdet (default)
#             1  Spane using MLR
#             2  Spane using L1MLR
# Outputs
#   dL       [ndarray] drawdown derivative
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
                ds1 = MLR(xl, yl, False)[1]   # left slope using Spane method
            elif method == 2:
                ds1 = L1MLR(xl, yl)[1]        # left slope using Spane modified method

        # Right derivative
        if i != n - 1:  # verify internal node
            xr = x[i:(k2+1)]
            yr = y[i:(k2+1)]
            dx2 = xr[-1] - xr[0]  # right log time difference
            if method == 0:
                ds2 = (yr[-1] - yr[0]) / dx2  # right slope using Bourdet method
            elif method == 1:
                ds2 = MLR(xr, yr, False)[1]   # right slope using Spane method
            elif method == 2:
                ds2 = L1MLR(xr, yr)[1]        # right slope using Spane modified method

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
#   X        time array
#   Y        drawdown vector
#   L        interval of evaluation (log time)
#   method  method=0 Bourdet derivate method (default)
#           method=1 Spane derivate method (using MLR)
#           method=2 Spane modified derivate method (using L1MLR)
# Outputs
#   dL       drawdown derivate vector
def derivateL(X,Y,L=1.0,method=0):
    # Check inputs
    X,Y = datavalidation(X,Y)
    n = len(X)  # number of elements
    dL = np.zeros(n)
    # Compute first derivate
    dL[0] = (Y[1]-Y[0])/(X[1]-X[0])  # forward finite difference
    # Compute last derivate
    dxr = np.abs(X[-1]-X)  # right diference
    xr = X[dxr<=L]  # get values that satisfy the L-condition
    yr = Y[dxr<=L]  # get values that satisfy the L-condition
    dxd = xr[-1]-xr[0]  # left log time difference (fixed for right extreme)
    # pseudo right-derivate computation
    if method==0:    # Bourdet method
        prd = (yr[-1]-yr[0])/dxd
    elif method==1:  # Spane method
        prd = MLR(xr,yr,False)[1]
    elif method==2:  # Spane modified method
        prd = L1MLR(xr,yr)[1] 
    dL[-1] = prd  # store last derivate
    # Compute internal derivates
    for i in np.arange(1,n-1,1,dtype=int):  # main loop
        # Left derivate
        dxl = np.abs(X[i]-X[:(i+1)])  # left diference
        posl = dxl<=L  # left conditional
        if np.sum(posl)>1:  # internal left node
            xl = X[:(i+1)][posl]  # get values that satisfy the L-condition
            yl = Y[:(i+1)][posl]  # get values that satisfy the L-condition
            dx1 = xl[-1]-xl[0]  # left log time difference
            if method==0:
                ds1 = (yl[-1]-yl[0])/dx1   # left slope using Bourdet method
            elif method==1:
                ds1 = MLR(xl,yl,False)[1]  # left slope using Spane method
            elif method==2:
                ds1 = L1MLR(xl,yl)[1]      # left slope using Spane modified method
        # Right derivate
        dxr = np.abs(X[i]-X[i:])  # right diference
        posr = dxr<=L  # right conditional
        if np.sum(posr)>1 and not(posr[-1]):  # internal right node
            xr = X[i:][posr]  # get values that satisfy the L-condition
            yr = Y[i:][posr]  # get values that satisfy the L-condition
            dx2 = xr[-1]-xr[0]  # left log time difference
            if method==0:
                ds2 = (yr[-1]-yr[0])/dx2   # right slope using Bourdet method
            elif method==1:
                ds2 = MLR(xr,yr,False)[1]  # right slope using Spane method
            elif method==2:
                ds2 = L1MLR(xr,yr)[1]      # right slope using Spane modified method
        elif np.sum(posr)>1 and posr[-1]:     # internal right node with fixed values
            dx2 = dxd  # fixed right difference
            ds2 = prd  # fixed right derivate
        # Calculate total derivate
        dL[i] = (ds1*dx2+ds2*dx1)/(dx1+dx2)
    return(dL)  # End Function