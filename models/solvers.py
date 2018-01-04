"""
CHEN pumping test analysis
Linear system solvers to be used in other methods

Functions:
    L1MLR          > L1 Multi-linear regression
    MLR            > Multi-linear regression
"""

import numpy as _np
from drawdown import _data_validation


"""______________________ LINEAR SOLVER METHODS ____________________________"""


# L1 Multi-linear regression
# Input
#   x          [ndarray] independent variable nxm
#   y          [ndarray] dependent variable nx1
# Output
#   B          [ndarray] coefficients of fitted model, B[0] intercept, B[1] slope
def L1MLR(x, y, error=1e-6):
    # Check input data
    x = _data_validation.to_ndarray(x)
    y = _data_validation.to_ndarray(y)

    if x.ndim == 1:  # from row vector to columns vector
        x = x.reshape((len(x), 1))
    n, m = x.shape  # size

    # Create matrix system
    A = _np.ones((n, m + 1))
    A[:, 1:] = x[:]  # avoid origin interception
    B1 = MLR(A, y, True)  # first approximation
    B = B1 + 9999.0

    # Main loop
    while _np.max(_np.abs(B - B1)) > error:
        B1 = B.copy()  # update coefficients
        W = _np.abs(B1[0] + _np.dot(x, B1[1:]) - y)
        W[W < error] = error  # replace small values
        W = (1.0 / W) ** 0.5  # observation weights (based on residuals)
        A1 = W[:, None] * A
        B = MLR(A1, W * y, True)  # new coefficients

    return(B)  # End of function


# Multi-linear regression
# Input
#   x          [ndarray] independent variable nxm
#   y          [ndarray] dependent variable nx1
#   origin     [boolean] if true then intersection is fixed with origin (0,0)
# Output
#   coef       [ndarray] coefficients of fitted model, coef[0] intercept, coef[1] slope
def MLR(x, y, origin=False):
    # Check input data
    x = _data_validation.to_ndarray(x)
    y = _data_validation.to_ndarray(y)

    if x.ndim == 1:  # from row vector to columns vector
        x = x.reshape((len(x), 1))
    n, m = x.shape  # size

    if origin:  # origin intersection
        A = x.copy()
    else:       # optimize intersection
        A = _np.ones((n, m + 1))
        A[:, 1:] = x[:]

    # Solve system
    coef = _np.linalg.solve(A, y)
    return(coef)  # End Function