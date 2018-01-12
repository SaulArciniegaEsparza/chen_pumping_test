# -*- coding: utf-8 -*-
"""

B-SPLINE with FREE KNOTS

Provide a flexible and robust fit to one-dimensional data using free-knot splines.
The knots are free and able to cope with rapid change in the underlying model.
Knot removal strategy is used to fit with only a small number of knots. 

This is a simplification of the code BSFK code written by Bruno Luong <brunoluong@yahoo.com>
(2010) in Matlab.
(original code at: https://www.mathworks.com/matlabcentral/fileexchange/25872-free-knot-spline-approximation)

Main function:

                    pp,flag = BSFK(x,y,k,nknots,constraints)

BSFK function is used in the CQBSFK (Constrained Quartic B-Splines with Free Knots)


DEPENDENCES:
numpy    version 1.10.4
scipy    version 0.17.0
cvxopt   version 1.1.9




Authors:

SAUL ARCINIEGA ESPARZA
Institute of Engineering of UNAM
zaul.ae@gmail.com

JOSUE TAGO PACHECO
Earth Sciences Division, Faculty of Engineering, UNAM
josue.tago@gmail.com

ANTONIO HERNANDEZ ESPRIU
Hydrogeology Group, Earth Sciences Division, Faculty of Engineering UNAM
ahespriu@unam.mx


 Reference: algorithm based on Schwetlick & Schutze
 
1. "Least squares approximation by splines with free knots", BIT Num.
    Math., 35 (1995);
2. "Constrained approximation by splines with free knots", BIT Num.
   Math., 37 (1997).
"""

import numpy as np
import scipy.sparse as scs
from scipy.sparse.linalg import eigsh
from copy import deepcopy
from cvxopt import matrix as cvomatrix
from cvxopt.solvers import qp as quadsolver
from cvxopt.solvers import options as cvoptions


"""____________________________ MAIN FUNCTION ______________________________"""


def BSFK(x, y, k, nknots, constraints):
    """
    pp,flag = BSFK(x,y,k,nknots,constraints)
    Least-squares fitting with Free-Knot B-Spline

         s(x) minimizes sum [s(xi)-y(i)]**2
           where s(x) is a spline function

       subject to derivative constricts at the extremes
             s'[0]=s11     s'[-1]=s12
            s''[0]=s21    s''[-1]=s22

    INPUTS
     x                [array] abscissa data
     y                [array] data function of x to be fitted by least-squares
     k                [int] k-1 order of the spline
     nknots           [int]  The number of sub-intervals between two
                       consecutive fixed knots (free knots)
     constraints      [list] n-element list of dictionaries with the derivative constraints
                         list = [{'p': p,'x': x,'v': v}] where p is the derivative order,
                                x is the array of abscissa data where the derivative
                                must be evaluated, and v is ordinate data

    OUTPUTS
     pp              [class] B-spline representation of 1-D curve (see scipy.interpolate.splrep
                     and scipy.interpolate.splev). pp can be evaluated using splev(x, pp)
                     and his derivative of order p can be computed with splev(x, pp, der=p)
     flag            [int] exit code, flag>0 success and flag<0 failure
    """

    # Check inputs
    x, y = to_numpyarrays([x, y])
    k, nknots = int(k), int(nknots)

    assert k >= 1, "k order must be higher than 1"
    assert nknots >= 1, "knots must be higher than 1"

    # Remove nans
    keep = np.where(~(np.isnan(x) + np.isnan(y)))
    x, y = x[keep], y[keep]
    # Sort data in ascending order
    arg = np.argsort(x)
    x = x[arg]
    y = y[arg]
    # Normalize the abscissa in range [0,1]
    xmin, xmax = x[0], x[-1]
    # Target RMS of fit residual
    m = len(y)

    # Default options
    maxiter = 100  # max number of Gauss-Newton iteration
    sigma = Chi2Estimation(y)  # iteration break RSM(residual)<sigma
    epsg = 1.0  # gradient precision
    epsgl = 1e-2  # gradient precision
    epsk = 1e-3  # knot precision
    # factor to determine the threshold of knot-removal
    knotremoval_factor = 9. / 8.
    # normalization [0,1]
    xscale = xmax - xmin
    normfun = lambda X: (X - xmin) / xscale
    unnormfun = lambda X: xmin + X * xscale
    x = normfun(x)
    # Knots generation
    t = np.linspace(0, 1, nknots + 1)
    t, knotidx = extend_knots(t, k)
    # indice of interior knots (free knots)
    p = np.arange(k, len(t) - k, dtype=int)
    # Order of the derivative for the regularization term
    d = min(2, k - 1)
    pntcon = ScalePntcon(constraints, xscale, normfun)  # fix constraints

    # Initialization before initialize the loop
    niter = 0  # number of Gauss-newton iterations
    nouteriter = 0
    validknots = {}
    threshold = None

    # Main Loop
    while True:
        nouteriter += 1
        smoothing = InitPenalization(y, t, k, d, p, knotidx, pntcon)

        # no free knot -> nothing else to do
        if len(p) == 0:
            if not(len(validknots) == 0):
                smoothing = deepcopy(validknots)
                t = deepcopy(smoothing['t'])
                p = deepcopy(smoothing['p'])
                flag = 7
            else:
                flag = 8
            break

        # Build the matrix/rhs of Boor and Rice constrained matrix to avoid
        # collapse of knots: C*t(p) >= h
        # The constraints matrices do *not* depend on free-knots
        C, h = BuildCMat(p)

        told = deepcopy(t)
        f = 1e20
        flag = 6

        # Gauss-Newton iteration
        while niter < maxiter:
            # Keep track the total number of Gauss-newton iterations
            niter += 1

            r, J, yfit = BuildJacobian(x, t, k, smoothing, pntcon)

            fold = f
            f = r.transpose().dot(r)
            if f < m*sigma**2:
                flag = 1
                break
            if abs(f - fold) < (sigma * epsg) ** 2:
                flag = 2
                break
            # Solve the reduced problem to find descend direction
            s = GaussNewtonStep(t, p, r, J, C, h)[0]
            # Line search
            gammaopt, gradline = linearsearch(x, t, p, s, k, smoothing, pntcon)
            if gammaopt <= 0:
                break
            elif abs(gradline) <= sigma * epsgl:
                break
            told = deepcopy(t)
            ds = gammaopt * s
            dt = expand_freeknots(ds, t, p)
            t = t + dt

            if np.max(np.abs(ds)) < epsk:
                break
            # End Gauss-Newton iteration
        # knot-removal routine
        inputs = (x, y, t, k, smoothing, p, validknots, pntcon, threshold, knotremoval_factor)
        outputs = KnotRemoval(*inputs)
        t, p, nremoved, validknots, knotidx, threshold = outputs
        if nremoved == 0:
            # restore the last valid knot state
            if not len(validknots) == 0:
                smoothing = deepcopy(validknots)
                t = deepcopy(smoothing['t'])
                p = deepcopy(smoothing['p'])
            break
        ## while-loop on knot-removal
     # Do last fitting
    yfit, alpha, r = fit(x, y, t, k, smoothing)
    # B-spline representation of 1-D curve
    t = unnormfun(t)
    bspline = (t, alpha, k - 1)
    return(bspline, flag)  # End BSFK()


"""________________________ DEFINE SUBFUNCTIONS ____________________________"""

# out = to_numpyarrays(vec)
# Convert all the elements in vec to numpy arrays
# INPUTS
#  var      [list, tuple] input data
def to_numpyarrays(vec):
    out = []
    for var in vec:
        if type(var) != np.ndarray:
            out.append(np.array(var, dtype=np.float32))
        else:
            out.append(var)
    return(out)  # End to_numpyarrays()


# Estimate the standard deviation of noise from the data
def Chi2Estimation(y):
    n = len(y)
    yfft = np.fft.fft(y) / n ** 0.5

    nremoved = np.max([1.0, np.ceil((n - 1.) / 16.)])
    nremoved = np.min([nremoved, np.floor((n - 1.) / 2.)])
    yfft = yfft[int(nremoved):-int(nremoved)]
    m = np.median(np.hstack((yfft.real, yfft.imag)) ** 2.)
    sigma = (m / (1. - 2. / 3. + 4. / 27. - 8. / 729.)) ** 0.5
    return(sigma)  # End Chi2Estimation()


# t, knotidx = extend_knots(t, k)
# Extend the knots by repeating with k times in both ends
def extend_knots(t, k):
    lt = len(t)
    idx = np.hstack((np.zeros(k), range(1, lt - 1),
                     np.full((k), lt - 1, dtype=np.int)))
    idx = np.array(idx, dtype=np.int)
    t = t[idx]
    knotidx = np.arange(len(t), dtype=np.int)
    return(t, knotidx)  # End extend_knots()


# t = expand_freeknots(freeknot,t,p)
# Expand the free-knots to entire set of knots by padding zeros for fixed knots
def expand_freeknots(freeknots, t, p):
    t = np.zeros(len(t))
    t[p] = freeknots
    return(t)


# smoothing = InitSmoothing(y,t,k,d,p,knotidx)
# Initialization the smoothing dictionary with some quantities we need later on
def InitSmoothing(y, t, k, d, p, knotidx):
    l, nk = len(p), len(t)
    E = np.zeros((nk, l), dtype=np.float32)
    E[p, range(l)] = 1.

    smoothing = {'t': np.array(t, dtype=np.int),
                 'knotidx': np.array(knotidx, dtype=int),  # original index of the knots
                  'k': k,
                  'd': np.array(d),
                  'p': np.array(p),
                  'subsidx': np.arange(len(t), dtype=np.int),
                  'r': k,
                  'm': len(y),
                  'ydata': np.array(y),
                  'ytarget': np.array(y),
                  'E': E,
                  'Deq': np.array([]),
                  'veq': np.array([]),
                  'Xeq': np.array([])}
    return(smoothing)  # End function


# smoothing = InitPenalization(y,t,k,d,p,knotidx, pntcon)
# Initialize smoothing and shape-constraints
def InitPenalization(y, t, k, d, p, knotidx, pntcon):
    smoothing = InitSmoothing(y, t, k, d, p, knotidx)
    smoothing = UpdateConstraints(smoothing, t, pntcon)
    return(smoothing)


# Build the constraint Matrix C*t>=h
# knots separation constraint matrix (see Boor and Rice)
# INPUTS
#  p        index of internal knots
#  t        knots (full knots)
# OTUPUTS
#  C,h      constrints matrixes
def BuildCMat(p):
    # NOTE: normalized serie at X[0]=0 and X[-1]=1
    # Minimum ratio (<1.0)
    Ep = 1. / 16.  # 0.0625
    m = len(p)

    # t(idx)>=t(idx-1)+E[t(idx+1)-t(idx-1)]
    C1 = np.zeros((m, m), dtype=np.float32)
    h1 = np.zeros(m)
    for i in range(m):
        C1[i, i] = 1.
        if m == 1:
            h1[i] = Ep  # (Ep-1)*X[0]-Ep*X[-1]
        elif i == 0:
            C1[i, i + 1] = -Ep
            # h1[i] = (Ep-1)*X[0]
        elif i == m - 1:
            C1[i, i - 1] = Ep - 1.
            h1[i] = Ep
        else:
            C1[i, i - 1] = Ep - 1.
            C1[i, i + 1] = -Ep

    # t(idx)<=t(idx+1)-E[t(idx+1)-t(idx-1)]
    C2 = np.zeros((m, m))
    h2 = np.zeros(m)
    for i in range(m):
        C2[i, i] = -1.
        if m == 1:
            h2[i] = -1. + Ep  # Ep*X[0]+(1-Ep)*X[-1]
        elif i == 0:
            C2[i, i + 1] = 1. - Ep
            # h2[i] = Ep*X[0]
        elif i == m - 1:
            C2[i, i - 1] = Ep
            h2[i] = -1 + Ep  # (1-Ep)*X[-1]
        else:
            C2[i, i - 1] = Ep
            C2[i, i + 1] = 1. - Ep

    # Join constraints
    C = np.vstack((C1, C2))
    h = np.hstack((h1, h2))
    return(C, h)  # End function


# B = Berntein(x,knots,k,alpha=[])
# Compute Bernstein polynomial basis using de Casteljau's algorithm
# INPUTS
# x       array with point coordinates at which the function is to be evaluated
# knots   array of knots points, must be ascending sorted
# k       k-1 order of the spline
# alpha   array with optional coefficients of the basis
# OUTPUTS
#  B      Bernstein polynomial matrix
def Bernstein(x, knots, k, alpha=None):
    # Check inputs
    if alpha is None:
        alpha = []
    x, knots, alpha = to_numpyarrays([x, knots, alpha])

    if len(alpha) == 0:
        opa = False
    else:
        opa = True

    szx = x.shape
    x = x.flatten(1)
    m = len(x)

    # Max possible value of the index
    maxj = len(knots) - k
    j = np.arange(maxj, dtype=np.int)
    js = np.arange(maxj, dtype=np.int)
    # Extreme values
    jmin = js[0]
    jmax = js[-1]
    # Create array
    B = np.zeros((m, jmax + k - jmin), dtype=np.float32)
    # B-Spline right
    col = [np.where((xi >= knots[k - 1: -k]) &
                    (xi <= knots[k:-k + 1]))[0][0] + k - 1 for xi in x]
    col = np.array(col, dtype=np.int)
    row = np.where((col >= 0) & (col <= B.shape[1]))
    col = col[row]
    B[row, col] = 1.

    # Main loop
    for kk in range(1, k):
        for jj in range(jmin, jmax + k - kk):
            # left side
            dt = knots[jj + kk] - knots[jj]
            if dt != 0:
                w1 = (x - knots[jj]) / dt
            else:
                w1 = np.zeros(x.shape)
            # right side
            dt = knots[jj + kk + 1] - knots[jj + 1]
            if dt != 0:
                w2 = (knots[jj + kk + 1] - x) / dt
            else:
                w2 = np.zeros(len(x), dtype=np.float32)
            # Create matrices
            ij = jj - jmin
            Bij = B[:, ij]
            Bij1 = B[:, ij + 1]
            # Element-by-element multiplication
            B[:, ij] = w1 * Bij + w2 * Bij1
    # Original map
    B = B[:, j]      # Bernstein polynomial
    if opa:  # alpha was input
        col = np.any(B, 0)
        if alpha.ndim == 1:
            B = np.dot(B[:, col], alpha[col])
            B = B.reshape(szx, order='F')
        else:
            B = np.dot(B[:, col], alpha[col, :])
            B = B.reshape((np.hstack((szx, alpha.shape[1]))), order='F')
    return(B)  # End of Bernstein()


# B,dB = BernKnotDeriv(x,knots,k,dknots)
# Compute Bernstein polynomial basis by de De Casteljau's algorithm
# and its derivative with respect to the knots
# INPUTS
# x       array of point coordinates at which the function is to be evaluated
# knots   array of knots points, must be ascending sorted
# k       k-1 order of the spline
# dknots  increment of knots, each column is the direction where the
#         derivative is computed.
# OUPUT
#  B      Bernstein polynomial matrix
#  dB     derivative of Bernstein polynomial with respect to the knots
def BernKnotDeriv(x, knots, k, dknots):
    # Convrt inputs to numpy arrays
    x, knots, dknots = to_numpyarrays([x, knots, dknots])

    x = x.flatten(1)
    m = len(x)

    # Maximum index value
    maxj = len(knots) - k
    j = np.arange(maxj, dtype=np.int)
    js = np.arange(maxj, dtype=np.int)

    if dknots.ndim == 2:
        p = dknots.shape[1]
    else:
        p = 1
    dknots = dknots.transpose().reshape((p, 1, dknots.shape[0]))

    # Right and left extreme nodes
    jmin = js[0]
    jmax = js[-1]
    B = np.zeros((m, jmax + k - jmin), dtype=np.float32)

    # B-Spline right
    col = [np.where((xi >= knots[k - 1: -k]) &
                    (xi <= knots[k: -k + 1]))[0][0] + k - 1 for xi in x]
    col = np.array(col)
    row = np.where((col >= 0) & (col <= B.shape[1]))
    col = col[row]
    B[row, col] = 1.

    # Derivative matrix
    dB = np.zeros((p, B.shape[0], B.shape[1]))
    #  first:   knot derivative
    #  second:  abscissa (x)
    #  third:   basis (j)

    # Main loop
    for kk in range(1,k):
        for jj in range(jmin, jmax + k - kk):
            # left side
            dt = knots[jj + kk] - knots[jj]
            if dt != 0:
                dkleft = dknots[:, 0, jj]
                ddt = dknots[:, 0, jj + kk] - dkleft
                w1 = (x - knots[jj]) / dt
                dw1 = np.array([[xi * yi for yi in ddt] for xi in w1])
                for ii in range(dw1.shape[1]):
                    dw1[:, ii] = (-dkleft[ii] - dw1[:, ii]) / dt
            else:
                w1 = np.zeros(len(x))
                dw1 = np.zeros((len(x), p))
            # right side
            dt = knots[jj + kk + 1] - knots[jj + 1]
            if dt != 0:
                dkright = dknots[:, 0, jj + kk + 1]
                ddt = dkright - dknots[:, 0, jj + 1]
                w2 = (knots[jj + kk + 1] - x) / dt
                dw2 = np.array([[xi * yi for yi in ddt] for xi in w2])
                for ii in range(dw2.shape[1]):
                    dw2[:, ii] = (dkright[ii] - dw2[:, ii]) / dt
            else:
                w2 = np.zeros(len(x))
                dw2 = np.zeros((len(x), p))
            # Create matrices
            ij = jj - jmin
            Bij = B[:, ij]
            Bij1 = B[:, ij + 1]
            # Element-by-element multiplication
            mult = (w1[:, None] * dB[:, :, ij].transpose() + dw1 * Bij[:, None]
                    + w2[:, None] * dB[:, :, ij + 1].transpose() + dw2 * Bij1[:, None])
            dB[:, :, ij] = mult.transpose()
            B[:, ij] = w1 * Bij + w2 * Bij1
    # Output values
    B = B[:, j]
    dB = dB[:, :, j]
    return(B, dB)  # End of BernKnotDeriv()


# Dr,td,kd,subs = DerivB(t,k,r)
# Takes the r-nth derivative of a spline function with respect to the coordinates X
# INPUTS
#  t      vector de knots (must be sorted at ascending order)
#  k      array of knots points (must be sorted at ascending order)
#  r      derivative order
# OUTPUTS
#  Dr     derivative matrix that maps coefficients the original coefficients to the derivative coefficients
#  td     subsknots that can be used later to evaluate the derivative
#  kd     spline order after the derivative
#  subs   index of subsknots
def DerivB(t, k, r):
    # Check input data type
    if type(t) != np.ndarray:
        t = np.array(t, dtype=np.int)
    # Create sparse matrix
    n = len(t) - k
    Dr = scs.eye(n, dtype=np.float32)
    for nu in range(r):
        ij = np.arange(nu + 1, n, dtype=np.int)
        dt = t[ij + k - nu - 1] - t[ij]
        h = (k - nu - 1) / dt
        d = n - nu - 1
        h = np.vstack((np.hstack((h, 0)) * -1, np.hstack((0, h))))
        HL = scs.spdiags(h, np.array([0, 1]), d, d + 1)
        Dr = HL.dot(Dr)
    # Output variables
    subs = np.arange(r, len(t) - r, dtype=np.int)
    td = t[subs]
    kd = k - r
    Dr = Dr.toarray()
    return(Dr, td, kd, subs)  # End of DerivB()


# Bspline(x,knots,k,r)
# Compute the Bspline function with coefficient alpha
# INPUT
#  x        array of points coordinates at which the function is to be evaluated
#  knots    array of knots points, must be ascending sorted
#  k        k-1 is the order of the spline
#  r        derivative order
def Bspline(x, knots, k, r):
    x, knots = to_numpyarrays([x, knots])
    ap, td, kd, subs = DerivB(knots, k, r)
    s = Bernstein(x, td, kd, ap)
    return(s)


# BsplineKnotDeriv(x,knots,k,alpha,r,dknots,lamb)
# Computes the Spline function using alpha coefficients and their derivative
# with respect to knots
# INPUTS
#  x       abscissa points array
#  knots   knots array
#  k       k-1 spline order
#  alpha   Bspline coefficients
#  r       derivative order
#  dknots  direction in which derivative is computed
#  lamb    dual variable
# OUTPUTS
#  ds      calculated spline values
#  dalpha  knots derivative in x and y direction of the dknots
def BsplineKnotDeriv(x, knots, k, alpha, r, dknots, lamb):
    x, knots, alpha = to_numpyarrays([x, knots, alpha])
    dknots, lamb = to_numpyarrays([dknots, lamb])
    x = x.flatten(1)

    subs = np.arange(r, len(knots) - r)
    td = knots[subs]
    kd = k - r

    # derivative knots
    if dknots.ndim == 1:
        dkd = dknots[subs]
    else:
        dkd = dknots[subs, :]
    B, dB = BernKnotDeriv(x, td, kd, dkd)
    # ap: r-nth derivative coefficients
    # dap: knots coefficients derivative
    Dr, td, kd, dDr, subs = DerivBKnotDeriv(knots, k, r, dknots)

    # B*Dr derivative
    p, m, n = dB.shape
    n = n + r

    # Compute dot multiplication from right to left
    #  ds = (B*dDr+dB*Dr)*alpha
    #  dalpha = lamb*(B*dDr+dB*Dr)
    lbdB = lamb.dot(B)

    Dralpha = Dr.dot(alpha)

    f = np.array([dB[:, i, :].reshape(-1) for i in range(m)])
    lbddB = lamb.dot(f).reshape(n - r, p)

    # localizate
    dDralpha = np.zeros((n - r, p))
    lbdBdDr = np.zeros((n, p))
    dBDralpha = np.zeros((m, p))
    for j in range(p):
        dDralpha[:, j] = dDr[j, :, :].dot(alpha)
        lbdBdDr[:, j] = lbdB.dot(dDr[j, :, :])
        dBDralpha[:, j] = dB[j, :, :].dot(Dralpha)

    # left multiplication
    ds = B.dot(dDralpha)+dBDralpha
    # right product
    dalpha = Dr.transpose().dot(lbddB) + lbdBdDr
    # output dimensions
    if p == 1:
        ds = ds.reshape(len(ds))
        dalpha = dalpha.reshape(len(dalpha))

    return(ds, dalpha)  # End Function


# DerivBKnotDeriv(knots,k,r,dknots)
# Takes the r-nth derivative of the spline function with respect to the abscissa
# INPUTS
#  knots   knots array
#  k       k-1 spline order
#  r       derivative order
#  dknots  knots derivative direction
# OUTPUTS
#  Dr      original coefficients derivatives arrays
#  td      subknots that can be used to evaluate derivative
#  kd      spline order after apply derivative
#  dDr     derivative multi-dimensional array for each direction
#  subs    subknots index td=knots(subs)
def DerivBKnotDeriv(knots, k, r, dknots):
    knots,dknots = to_numpyarrays((knots, dknots))
    nk = len(knots)
    n = nk - k

    if dknots.ndim == 1:
        p = 1
        dknots = dknots.reshape((1, len(dknots)))
    else:
        p = dknots.shape[1]
        dknots = dknots.transpose()

    dout = []
    In = scs.eye(n)
    Zn = scs.bsr_matrix(np.zeros((n, n)))

    for v in range(p):
        Dr = deepcopy(In)
        dDr = deepcopy(Zn)

        for nu in range(r):
            ij = np.arange(nu+1, n, dtype=np.int)
            dk = knots[ij + k - nu - 1] - knots[ij]
            ddk = dknots[v, ij + k - nu - 1] - dknots[v, ij]
            h = (k - nu - 1) / dk
            dh = -h * ddk / dk
            d = n - nu - 1

            h = np.vstack((np.hstack((h, 0)) * -1, np.hstack((0, h))))
            dh = np.vstack((np.hstack((dh, 0)) * -1, np.hstack((0, dh))))

            HL = scs.spdiags(h, np.array([0, 1]), d, d + 1)
            dHL = scs.spdiags(dh, np.array([0, 1]), d, d + 1)

            dDr = dHL.dot(Dr) + HL.dot(dDr)
            Dr = HL.dot(Dr)
        dout.append(dDr.toarray())
    # Output vars
    subs = np.arange(r, len(knots) - r, dtype=np.int)
    td = knots[subs]
    kd = k - r
    Dr = Dr.toarray()
    if p == 1:
        dDr = np.zeros((1, dout[0].shape[0], dout[0].shape[1]))
        dDr[0, :, :] = dout[0]
    else:
        dDr = np.array(dout)
    return(Dr, td, kd, dDr, subs)  # End Function


# Deq,vec,Xeq = BuildDeqMat(knots,k,pntcon,alpha=[],lagrange=[],E=[])
# Build inequality constraints considering Deq*alpha=vec
# INPUTS
#  knots      knots array
#  k          k-1 spline order
#  pntcon     constraints dict {'p': derivative_order, 'x': x, 'v': dydx,'nc': len(x)}
#  alpha      optional spline coefficients
#  lagrange   
#  E          knots derivative direction
# OUTPUTS
#  Deq        linear constraints coefficients array
#  vec        constraints independent terms array
#  Xeq        normalized array that satisfy (Xeq*Deq)*alpha=Xeq*veq for numeric stability
#
# If alpha, lagrange and E are input, knots derivative are returned
def BuildDeqMat(knots, k, pntcon, alpha=None, lagrange=None, E=None):
    if alpha is None:
        alpha = []
    if lagrange is None:
        lagrange = []
    if E is None:
        E = []

    knots, alpha, lagrange, E = to_numpyarrays((knots, alpha, lagrange, E))

    # Fix E size
    if E.shape[-1] == 1:
        E = E.flatten(1)

    n = len(knots) - k
    # shape restrictions number
    l = len(pntcon)

    # Build equality D array
    # this result depends of the knots location
    m = np.sum([x['nc'] for x in pntcon])  # restrictions number

    # Build D array
    if len(alpha) == 0:
        nzD = m * k  # non-zero number of elements

        # iterate over derivative restrictions
        # build indexes and the linear constraints value
        ilast = 0
        row = np.zeros(nzD, dtype=np.int)
        col = np.zeros(nzD, dtype=np.int)
        val = np.zeros(nzD, dtype=np.float)
        row = np.zeros(nzD, dtype=np.int)
        xn = np.zeros(m)
        rowstar = 0
        # singtol = np.spacing(1)**2
        for i in range(l):
            s = pntcon[i]
            Dp = Bspline(s['x'], knots, k, s['p'])
            c, r = np.where(Dp.transpose() != 0)
            dp = Dp[r, c]
            nz = len(dp)
            mi = Dp.shape[0]

            # fix
            r = rowstar + r
            idx = ilast + np.arange(nz, dtype=np.int)
            row[idx] = r
            col[idx] = c
            val[idx] = dp
            dn = np.linalg.norm(Dp, ord=2)
            xn[rowstar+np.arange(mi, dtype=np.int)] = 1.0 / dn
            rowstar += mi
            ilast += nz
        # Build lower constraints array
        if ilast < nzD:
            idx = np.arange(ilast, dtype=np.int)
            Deq = scs.csr_matrix((val[idx], (row[idx], col[idx])), shape=(m, n))
        else:
            Deq = scs.csr_matrix((val, (row, col)), shape=(m, n))

        Xeq = scs.spdiags(xn, 0, m, m).toarray()
        # rhs inequality constraints
        vec = np.hstack([X['v'] for X in pntcon])  # concatenate constraints
        Deq = Deq.toarray()  # convert sparse to array

    else:  # build derivative D array with respect knots
        # iterate over derivative constraints
        # build indexes and the linear constraints values
        if E.ndim == 1:
            ne = 1
            DL = np.zeros(n)
            DR = np.zeros(m)
        else:
            ne = E.shape[1]
            DL = np.zeros((n, ne))
            DR = np.zeros((m, ne))
        rowstart = 0

        for i in range(l):
            s = pntcon[i]
            p = s['p']
            mi = s['nc']
            idx = rowstart + np.arange(mi, dtype=np.int)
            u = lagrange[idx]
            c = BsplineKnotDeriv(s['x'], knots, k, alpha, p, E, u)
            if ne == 1:
                DR[idx] = c[0]
                DL += c[1]  # cumulative constraint gradient
            else:
                DR[idx, :] = c[0]
                DL += c[1]  # cumulative constraint gradient
            rowstart += mi
        # return knots derivative as first argument
        Deq = deepcopy(DR)
        # return knots dual derivative
        vec = DL.reshape((n, ne), order='F')
        Xeq = np.array([])

    return(Deq, vec, Xeq)  # End Function


# B,ueq,alpha,yfit,r,dB = ModelMat(x,knots,k,smoothing,dknots=[],derivate=False)
# Compute several variables for the estimation of the model
# INPUTS
#  x           values for the spline evaluation
#  knots       free knots array
#  k           k-1 spline order
#  smothing    dictionary with problem options and data
#  dknots      [optional] array with derivate direction
#  derivate    [boolean, optional] if it is True, compute dB, in other case dB=[]
# OUTPUTS
#  B          Bernstein coefficients 
#  ueq        Lagrange multipliers of equality constrints 
#  alpha      spline coefficients
#  yfit       fited spline
#  r          residuals of yfited
#  dB         derivate of Bernstein coefficients
def ModelMat(x, knots, k, smoothing, dknots=None, derivate=False):
    if dknots is None:
        dknots = []
    x, knots, dknots = to_numpyarrays((x, knots, dknots))

    if derivate:
        if len(dknots) == 0:
            dknots = smoothing['E']
        B, dB = BernKnotDeriv(x, knots, k, dknots)
    else:
        B = Bernstein(x, knots, k)
        dB = np.array([])

    # Constrained regression
    alpha, ueq = ConL2Fit(smoothing['ytarget'], B, smoothing['Deq'], smoothing['veq'], smoothing['Xeq'])


    # Fit model to data
    yfit = B.dot(alpha)
    # residual
    r = smoothing['ytarget'] - yfit
    # Output variables
    return(B, ueq, alpha, yfit, r, dB)


# ConL2Fit(y,B,Deq,veq,Xeq)
# Solve the fixed-knot least-squares fitting problem under shape-constraints by using quadratic programming
# OUTPUTS
#  alpha    optimal solution
#  ueq      Lagrange multiplier for equality constraints
def ConL2Fit(y, B, Deq, veq, Xeq):
    y, B, Deq, veq, Xeq = to_numpyarrays((y, B, Deq, veq, Xeq))

    # Hesian and gradient residuals
    H = B.transpose().dot(B)
    g = -(y.dot(B))
    # normalized constraints
    Aeq = Xeq.dot(Deq)
    beq = Xeq.dot(veq)
    # Solve quadratic programing
    s0 = np.zeros(g.shape)
    H = RestoreSPMat(H)
    alpha, ueq = quadprog(H, g, Aeq, beq, s0)
    # Check Lagrange multipliers sign
    s = (ueq.dot(Deq)).dot(H.dot(alpha) + g)
    ueq = np.sign(s) * ueq
    ueq = Xeq.dot(ueq)
    return(alpha, ueq)


# BuildJacobian(x,knots,k,smoothing,pntcon,dt=[])
# Build the Jacobian matrix (with respect to the free knots) matrix for the Gauss-Newton iteration
# INPUTS
#  x           array with x values to evaluate the spline
#  knots       free knots array
#  k           k-1 spline order
#  smoothing   dictionary with problem variables
#  pntcon      list of dictionaries with options of derivate constranints
#  dt          [optional] array with the knot derivate direction
# OUTPUTS
#  r          is the fit residual
#  J          Jacobian of r with respect to the free knots
#  yfit       fitted spline
def BuildJacobian(x, knots, k, smoothing, pntcon, dt=None):
    if dt is None:
        dt = []
    x, knots, dt = to_numpyarrays((x, knots, dt))
    # Update constraints
    smoothing = UpdateConstraints(smoothing, knots, pntcon)
    if len(dt) == 0:
        dt = smoothing['E']
    if dt.shape[-1] == 1:
        dt = dt.flatten(1)
    # Compute the basis and derivatives with respect to free-knot
    B, ueq, alpha, yfit, r, dB = ModelMat(x, knots, k, smoothing, dknots=dt, derivate=True)

    l, m, n = dB.shape
    # Shape-constrained Jacobian (Kaufman)
    Deq = deepcopy(smoothing['Deq'])

    # Derivative of the active constraints wrt alpha
    R = -Deq
    # Derivative of the active constraints wrt knots
    mpntcon = np.sum([con['nc'] for con in pntcon])
    upntcon = ueq[:mpntcon]
    Gamma, Rtu, Xeq_eq  = BuildDeqMat(knots, k, pntcon, alpha=alpha,
                                      lagrange=upntcon, E=dt)
    Gamma = -Gamma
    # null space of R
    N = null(R)[0]
    BN = B.dot(N)
    BNi = np.linalg.pinv(BN)  # pseudoinverse

    if dt.ndim > 1:  # multiple
        a = -alpha
        dBa = dB * a[None, None, :]
        if l == 1:
            dBa = np.sum(dBa, axis=2).flatten(1)
        else:
            dBa = np.sum(dBa, axis=2).transpose().reshape((m, l), order='F')
        Ri = np.linalg.pinv(R)  # pseudoinverse
        dB2 = B.dot(Ri.dot(Gamma))
        Psi = dBa + dB2
        # Kaufman matrix
        Ay = Psi - BN.dot(BNi.dot(Psi))
        # Transposed Kauffman
        rdB = r[None, :, None] * dB
        if l == 1:
            rdB = np.sum(rdB, axis=1).flatten(1)
        else:
            rdB = np.sum(rdB, axis=1).transpose().reshape((n, l), order='F')
        K = rdB + Rtu
        Phi = K.transpose().dot(N).dot(BNi).transpose()
        # Golub-Peyreyra
        J = Ay - Phi

    else:  # single directional derivative (typical when called from simul)
        # Fix dimensions
        dB = dB[0, :, :]

        dBa = dB.dot(-alpha)
        Ri = np.linalg.pinv(R)  # pseudoinverse
        dB2 = B.dot(Ri.dot(Gamma))
        Psi = dBa + dB2
        # Fix dimension
        Psi = Psi.flatten(1)
        Rtu = Rtu.flatten(1)
        # Kaufman matrix
        Ay = Psi - BN.dot(BNi.dot(Psi))
        # Transposed Kauffman
        Kt = (r.dot(dB)) + Rtu
        Phi = (Kt.dot(N)).dot(BNi)
        # Golub-Peyreyra
        J = Ay - Phi
    return(r, J, yfit)  # End BuildJacobian


# Solve reduced problem by quadratic programing
def GaussNewtonStep(t, p, r, J, C, h):
    t, r, J, C, h = to_numpyarrays((t, r, J, C, h))
    p = np.array(p, dtype=int)
    # Hessian and residual gradient vector for Qp
    H = RestoreSPMat(J.transpose().dot(J))
    g = J.transpose().dot(r)
    tf = t[p]  # free knots
    hr = h - C.dot(tf)  # new constraints
    s0 = np.zeros(g.shape)
    x = quadprog(H, g, G=-C, h=-hr, s0=s0)[0]
    return(x, g)  # End GaussNewtonStep()


# x,ueq = quadprog(H,g,A,b,s0)
# Quadratic programing to solve
#       min[ (1/2)*x'*H*x + g'*x ]
# subject to
#       G*x <= h
#       A*x = b
# INPUTS
#  S0      initial values
# OUTPUTS
#  x       solution of the problem
#  ueq     Lagrange multipliers of the equality constraints
def quadprog(H, g, A=None, b=None, s0=None, G=None, h=None):
    # Transformar entradas
    if A is None: A = []
    if b is None: b = []
    if s0 is None: s0 = []
    if G is None: G = []
    if h is None: h = []
    H, g, A, b, s0, G, h = to_numpyarrays((H, g, A, b, s0, G, h))
    # (1/2)*x'*H*x + g'*x
    H = cvomatrix(H, tc='d')
    g = cvomatrix(g, tc='d')
    op = 1
    # G*x <= h
    if len(G) != 0 and len(h) != 0:
        if len(h) == 1:
            G = cvomatrix(G, (1, len(G)), tc='d')
        else:
            G = cvomatrix(G, tc='d')
        h = cvomatrix(h, tc='d')
        op = 2
    else:
        G = []
        h = []
    # A*x = b
    if len(A) != 0 or len(b) != 0:
        if len(b) == 1:
            A = cvomatrix(A, (1, len(A)), tc='d')
        else:
            A = cvomatrix(A, tc='d')
        b = cvomatrix(b, tc='d')
        op = 1
    else:
        A = []
        b = []
    # initial values
    s0 = cvomatrix(s0, tc='d')
    # solve quadratic problem
    options = {'show_progress': False,
               'maxiters': 20}
    cvoptions.update(options)
    if op == 1:
        sol = quadsolver(H, g, A=A, b=b, initvals=s0)
    elif op == 2:
        sol = quadsolver(H, g, G=G, h=h, initvals=s0)
    # output variables
    x = np.array(sol['x']).flatten(1)  # result
    ueq = np.array(sol['y']).flatten(1)  # equality lagrange multipliers
    return(x, ueq)  # End quadprog()


# Make presumably SPMat to be numerical stable
def RestoreSPMat(H):
    if len(H) > 1:
        # make symmetric array
        H = 0.5 * (H.transpose() + H)
        try:
            # Eigenvalues with smallest algebraic value
            smallestev = eigsh(H, 1, which='SA', tol=1e-3)
            smallestev = smallestev[0]
        except:
            smallestev = 0.
        largestev = np.linalg.norm(H, ord=2)
        abstol = 1e-6 * largestev
        if smallestev <= abstol:
            # add and small value to improve the constraints
            deflation = 2 * max(-smallestev, abstol)
            # H = H + diag(deflation*ones(l,1))
            H[range(H.shape[0]), range(H.shape[1])] += deflation
    # Output variables
    return(H)  # End of RestoreSPMat()


# UpdateConstraints(smoothing,knots,pntcon=[])
# Update the constraint matrices and rhs when knots positions change
def UpdateConstraints(smoothing, knots, pntcon=None):
    if pntcon is None: pntcon = []
    if len(pntcon) == 0:
        pntcon = deepcopy(smoothing['pntcon'])
    smoothing['t'] = knots
    k = smoothing['k']
    Deq, veq, Xeq = BuildDeqMat(knots, k, pntcon)
    new_values = {'pntcon': pntcon,
                  'Deq': Deq,
                  'veq': veq,
                  'Xeq': Xeq}
    smoothing.update(new_values)
    return(smoothing)  # End of UpdateConstraints()


# Rank and null space of a square matrix
def null(a, rtol=1e-15):
    u,s,v = np.linalg.svd(a)
    rank = (s > rtol*s[0]).sum()
    nullspace = v[rank:].T.copy()
    return(nullspace, rank)  # End Function


# t,p,nremoved,validknots,knotidx,threshold = KnotRemoval(x,y,t,k,smoothing,p,firstcall,knotremoval_factor,validknots,pntcon,threshold=None)
# Try the remove the redundant knots
def KnotRemoval(x, y, t, k, smoothing, p, validknots, pntcon, threshold, knotremoval_factor):
    knotidx = smoothing['knotidx']
    nremoved = 0  # number of knot removed
    # Define subfunction to return a temporary state when the knot is removed
    def remove(iremoved):
        tr = deepcopy(t)
        pr = deepcopy(p)
        tr = np.delete(tr, p[iremoved])
        knotidx = deepcopy(smoothing['knotidx'])
        knotidx = np.delete(knotidx, p[iremoved])
        pr = np.hstack((p[:iremoved], pr[iremoved+1:]-1))
        sr = InitPenalization(y, tr, k, smoothing['d'], pr,
                              knotidx, pntcon)
        return(tr, pr, sr, knotidx)

    # Loop knot remove
    while True:
        # Compute the residual

        yfit, alpha, r = fit(x, y, t, k, smoothing)
        rmsresidu = (r.transpose().dot(r)/len(r))**0.5

        # Save the threshold in persistent variable for later use
        if threshold == None:
           threshold = knotremoval_factor * rmsresidu
        # Oops, residual increases too much when this knot would be removed
        if rmsresidu >= threshold:
            break
        else:
            validknots = deepcopy(smoothing)
            validknots['t'] = t
        # Nothing Left
        if len(p) == 0:
            break
        # Find the knot so that the fit residual is smallest after it has been removed
        minrms = 1e20
        # Loop on all knots
        for ir in range(len(p)):
            tr, pr, sr, trash = remove(ir)
            yfit, alpha, r = fit(x, y, tr, k, sr)
            rmsir = (r.transpose().dot(r)/len(r))**0.5
            if rmsir < minrms:  # keep track the best
                minrms = rmsir
                iremoved = ir
        # Removed selected knot
        t, p, smoothing, knotidx = remove(iremoved)
        nremoved += 1
    return(t, p, nremoved, validknots, knotidx, threshold)  # End KnotRemoval()


# yfit,alpha,r = fit(x,y,t,k,smoothing)
def fit(x, y, t, k, smoothing):
    # Update the constraints matrices corresponding to new knot positions
    smoothing = UpdateConstraints(smoothing, t)
    B, ueq, alpha, yfit, r, dB = ModelMat(x, t, k, smoothing)
    if len(B) == 0:
        yfit = np.full(y.shape, np.nan)
    else:
        yfit = yfit[range(len(y))]
    r = y - yfit
    return(yfit, alpha, r)  # End fit()


# newpntcon = ScalePntcon(pntcon,xscale,normfun)
# Scale the pntcon abscissa/derivative values to accomodate the fact that
# we change the scale of the abscissa x
def ScalePntcon(pntcon, xscale, normfun):
    l = len(pntcon)  # number of constrints
    newpntcon = []
    for i in range(l):
        s = deepcopy(pntcon[i])
        p = s['p']
        # linearly map to [0,1]
        x = normfun(np.array(s['x']))
        nc = len(x)
        a = xscale**p
        v = np.array(s['v'])
        s['x'] = x
        s['v'] = a * v  # scaled derivative
        s['nc'] = nc
        newpntcon.append(s)
    return(newpntcon)  # End ScalePntcon


# gammaopt,g0 = linearsearch(x,t0,p,s,k,smoothing,pntcon)
# gammaopt is the "optimal" step such that t0 + gammaopt*s decreases
# in a "good" way
# Find alpha that satisfies strong Wolfe conditions
def linearsearch(x, t0, p, s, k, smoothing, pntcon):
    # Adjustable constant
    delta = 1./8.
    # Constants used in two Wolfe's tests
    rm1 = delta
    rm2 = 0.9
    l = len(p)
    io = 1
    nap = 1
    napmax = 10 * l
    gmin = 0
    gmax = 1
    gopt = 15./16.
    gamma = 0

    simparams = (x, t0, p, s, k, smoothing, pntcon)

    f0, g0, trash = simul(gamma, *simparams)
    n = 1
    d = 1
    fpn = deepcopy(g0)
    gammaopt, trash, fopt, gopt, logic, trash1 = nlis0(n, gamma, f0, fpn, gopt, gmin, gmax,
                                                       d, g0, rm2, rm1, io, nap, napmax, simparams)
    if not(fopt < f0):
        gammaopt = 0
    # Output params
    return(gammaopt, g0)  # End linearsearch()


# f,g,yfit = simul(gamma,x,t0,p,s,k,smoothing,pntcon)
# Simulation gateway for the linearsearch nlis0
def simul(gamma, x, t0, p, s, k, smoothing, pntcon):
    # direction where the derivative will be computed
    dt = expand_freeknots(s, t0, p)
    t = t0 + gamma * dt
    # Update the constraints matrices corresponding to new knot positions
    smoothing = UpdateConstraints(smoothing, t, pntcon)
    r, J, yfit = BuildJacobian(x, t, k, smoothing, pntcon, dt)

    f = 0.5 * r.transpose().dot(r)  # square of L2 residual
    g = J.transpose().dot(r)  # Gradient of f (derivative with respect to gamma)
    # Output variables
    return(f, g, yfit)


# ps = procsa(x1,x2)
# Generic L2 scalar product
def procsa(x1, x2):
    ps = x1.transpose().dot(x2)
    return(ps)


# t = dcube(t,f,fp,ta,fa,fpa,tlower,tupper)
# Using f and fp at t and ta, computes new t by cubic formula
# safeguarded inside [tlower,tupper]
def dcube(t, f, fp, ta, fa, fpa, tlower, tupper):
    # Convert inputs to float numbers
    t, f, fp, ta, fa = float(t), float(f), float(fp), float(ta), float(fa)
    fpa, tlower, tupper = float(fpa), float(tlower), float(tupper)
    # Init
    z1 = fp + fpa - 3. * (fa - f) / (ta - t)
    b = z1 + fp
    # First compute the discriminant (without overflow)
    if abs(z1) <= 1:
        discri = z1 * z1 - fp * fpa
    else:
        discri = fp / z1
        discri = discri * fpa
        discri = z1 - discri
        if not(any([z1 >= 0, discri >= 0])):
            discri = z1 * discri
        else:
            discri= -1.
    if discri < 0:
        if fp < 0:
            t = tupper
        else:
            t = tlower
        return(t)
    # Discriminant nonnegative, compute solution (without overflow)
    discri = discri**0.5
    if t - ta < 0:
        discri = -discri
    sign = (t - ta) / abs(t - ta)
    if b * sign > 0:
        t = t + fp * (ta - t)/(b + discri)
    else:
        den = z1 + b + fpa
        anum = b - discri
        if abs((t - ta) * anum) < (tupper - tlower) * abs(den):
            t = t + anum * (ta - t) / den
        else:
            t = tupper
    t = min([max(t, tlower), tupper])
    return(t)  # End of dcube


# t,xn,fn,g,logic,nap = nlis0(n,xn,fn,fpn,t,tmin,tmax,d,g,amd,amf,io,nap,napmax,simulparams)
# Linear search for minimization of h(t):=f(x + t.d), t>=0;
# cubic interpolation + Wolfe's tests
# INPUT:
#     n: dimension of the full space
#     xn(n): current vector
#     fn: current cost function
#     fpn: derivative of h(t) at t=0
#     t: first guess of t
#     tmin, tmax: t-bracket interval
#     d(n): direction of linear search
#     g(n): current gradient vector (of f at xn)
#     amd, amf: constants for Wolfe's tests
#     imp, io: level and of file identifier for output printing
#     nap: current number of simulator calling
#     napmax: maximum number of allowed simulator calling
#     simulparams: tuple with input arguments for simul function (x,t0,p,s,k,smoothing,pntcon)
#  OUTPUT:
#     t: new t-value
#     xn(n): new current vector
#     fn: new function value
#     g(n): new gradient
#     logic =
#        0          line search OK
#        1          minimization stucks
#        4          nap > napmax
#        5          user stopping
#        6          function and gradient do not agree
#        < 0        contrainte implicite active
#
#     nap: updated number of simulator calling
def nlis0(n, xn, fn, fpn, t, tmin, tmax, d, g, amd, amf, io, nap, napmax, simulparams):

    if not(n > 0 and fpn < 0 and t > 0 and tmax > 0 and
           amf > 0 and amd > amf and amd < 1):
        logic = 6
        return(t, xn, fn, g, logic, nap)

    tesf = amf * fpn;
    tesd = amd * fpn
    barmin = 0.01
    barmul = 3
    barmax = 0.3
    barr = barmin
    td = 0
    tg = 0
    fg = fn
    fpg = fpn
    ta = 0
    fa = fn
    fpa = fpn
    #
    t = max([t, tmin])
    if t > tmax:
        tmin = tmax
    #
    while fn + t * fpn >= fn + 0.9 * t * fpn:
        t = 2.0 * t
    # Label 30
    indica = 1
    logic = 0
    if t > tmax:
        t = tmax
        logic = 1
    #        
    x = xn + t * d
    # Main Loop
    while True:
        nap += 1
        if nap > napmax:
            logic = 4
            fn = fg
            xn = xn + tg * d
            return(t, xn, fn, g, logic, nap)
        indic = 4
        # Call the simulator
        f, g, trash = simul(x, *simulparams)

        ps = d * g
        fp = ps

        # First Wolfe's test
        interpflag = False #ok
        ffn = f - fn
        if ffn > t * tesf:
            td = t
            fd = f
            fpd = fp
            indicd = indic
            logic = 0
            interpflag = True
        else:
        # Perform second Wolfe's test
            if fp > tesd or logic != 0:
                if fp > tesd:
                    logic = 0
                fn = f
                xn = x
                return(t, xn, fn, g, logic, nap)
            tg = t
            fg = f
            fpg = fp
            interpflag = td != 0
        # Extrapolation
        if not(interpflag):  # no interpolation
            taa = t
            gauche = (1.0 + barmin) * t
            droite = 10 * t
            t = dcube(t, f, fp, ta, fa, fpa, gauche, droite)
            ta = taa
            if t >= tmax:
                logic = 1
                t = tmax
        else:  # interpolation
            if indica <= 0:
                ta = t
                t = 0.9 * tg + 0.1 * td
            else:
                test = barr * (td-tg)
                gauche = tg + test
                droite = td - test
                taa = t
                t = dcube(t, f, fp, ta, fa, fpa, gauche, droite);
                ta = taa
                if t > gauche and t < droite:
                    barr = barmin
                else:
                    barr = min([barmul * barr, barmax])
        #
        fa = f
        fpa = fp
        indica = indic

        if td == 0:
            exitloop = False
        elif td - tg < tmin:
            exitloop = True
        else:
            z = xn + t * d
            exitloop = z == xn or z == x
        #
        if exitloop:
            logic = 6
            if indicd < 0:
                logic = indicd
            if tg != 0:
                fn = fg
                xn = xn + tg * d
            return(t, xn, fn, g, logic, nap)
        #
        x = xn + t * d
        # End loop
    # End nlis0()
