
# sph_cython.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True, profile=True


import numpy as np
cimport numpy as np
from libc.math cimport sqrt, exp, pow

# Define constant for PI
cdef double PI = 3.141592653589793

########################################################################
# 1. Pairwise Separations
########################################################################

def getPairwiseSeparations(double[:, :] ri, double[:, :] rj):
    """
    Get pairwise separations between two sets of coordinates.
    Parameters:
      ri : an M x 3 array of positions.
      rj : an N x 3 array of positions.
    Returns:
      dx, dy, dz : each an M x N array of separations.
    """
    cdef int M = ri.shape[0]
    cdef int N = rj.shape[0]
    cdef int i, j

    cdef np.ndarray[double, ndim=2] dx = np.empty((M, N), dtype=np.double)
    cdef np.ndarray[double, ndim=2] dy = np.empty((M, N), dtype=np.double)
    cdef np.ndarray[double, ndim=2] dz = np.empty((M, N), dtype=np.double)
    cdef double[:, :] dx_view = dx
    cdef double[:, :] dy_view = dy
    cdef double[:, :] dz_view = dz

    for i in range(M):
        for j in range(N):
            dx_view[i, j] = ri[i, 0] - rj[j, 0]
            dy_view[i, j] = ri[i, 1] - rj[j, 1]
            dz_view[i, j] = ri[i, 2] - rj[j, 2]
    return dx, dy, dz

########################################################################
# 2. Gaussian Kernel (W)
########################################################################

def W(double[:, :] x, double[:, :] y, double[:, :] z, double h):
    """
    Gaussian Smoothing kernel (3D) computed elementwise.
    Parameters:
      x, y, z : 2D arrays of positions (typically differences).
      h       : smoothing length.
    Returns:
      w : array of the same shape as x, with the kernel evaluated elementwise.
    """
    cdef int m = x.shape[0]
    cdef int n = x.shape[1]
    cdef int i, j
    cdef np.ndarray[double, ndim=2] w = np.empty((m, n), dtype=np.double)
    cdef double[:, :] w_view = w
    cdef double r, norm

    norm = pow(1.0/(h*sqrt(PI)), 3)
    for i in range(m):
        for j in range(n):
            r = sqrt(x[i, j]*x[i, j] + y[i, j]*y[i, j] + z[i, j]*z[i, j])
            w_view[i, j] = norm * exp(- (r*r) / (h*h))
    return w

########################################################################
# 3. Gradient of Gaussian Kernel (gradW)
########################################################################

def gradW(double[:, :] x, double[:, :] y, double[:, :] z, double h):
    """
    Gradient of the Gaussian Smoothing kernel (3D) computed elementwise.
    Parameters:
      x, y, z : 2D arrays of positions (typically differences).
      h       : smoothing length.
    Returns:
      wx, wy, wz : gradients (each an array of same shape as x).
    """
    cdef int m = x.shape[0]
    cdef int n = x.shape[1]
    cdef int i, j
    cdef np.ndarray[double, ndim=2] wx = np.empty((m, n), dtype=np.double)
    cdef np.ndarray[double, ndim=2] wy = np.empty((m, n), dtype=np.double)
    cdef np.ndarray[double, ndim=2] wz = np.empty((m, n), dtype=np.double)
    cdef double[:, :] wx_view = wx
    cdef double[:, :] wy_view = wy
    cdef double[:, :] wz_view = wz
    cdef double r, n_val
    cdef double factor = pow(h, 5) * pow(PI, 1.5)  # h^5 * (PI)^(3/2)
    
    for i in range(m):
        for j in range(n):
            r = sqrt(x[i, j]*x[i, j] + y[i, j]*y[i, j] + z[i, j]*z[i, j])
            n_val = -2.0 * exp(- (r*r) / (h*h)) / factor
            wx_view[i, j] = n_val * x[i, j]
            wy_view[i, j] = n_val * y[i, j]
            wz_view[i, j] = n_val * z[i, j]
    return wx, wy, wz

########################################################################
# 4. Density Calculation (getDensity)
########################################################################

def getDensity(double[:, :] r, double[:, :] pos, double m, double h):
    """
    Calculate density at sampling locations r from particles at pos.
    Parameters:
      r   : an M x 3 array (sampling positions).
      pos : an N x 3 array (particle positions).
      m   : particle mass.
      h   : smoothing length.
    Returns:
      rho : an array of length M with the computed densities.
    """
    cdef int M = r.shape[0]
    cdef int N = pos.shape[0]
    cdef int i, j
    cdef np.ndarray[double, ndim=1] rho = np.zeros(M, dtype=np.double)
    cdef double diff0, diff1, diff2, r_val, norm, w_val

    norm = pow(1.0/(h*sqrt(PI)), 3)
    for i in range(M):
        for j in range(N):
            diff0 = r[i, 0] - pos[j, 0]
            diff1 = r[i, 1] - pos[j, 1]
            diff2 = r[i, 2] - pos[j, 2]
            r_val = sqrt(diff0*diff0 + diff1*diff1 + diff2*diff2)
            w_val = norm * exp(-(r_val*r_val)/(h*h))
            rho[i] += m * w_val
    return rho

########################################################################
# 5. Equation of State (getPressure)
########################################################################

def getPressure(np.ndarray[double, ndim=1] rho, double k, double n):
    """
    Compute pressure using a polytropic equation of state.
    Parameters:
      rho : density (1D array).
      k   : equation-of-state constant.
      n   : polytropic index.
    Returns:
      P : pressure (1D array).
    """
    cdef int M = rho.shape[0]
    cdef int i
    cdef np.ndarray[double, ndim=1] P = np.empty(M, dtype=np.double)
    for i in range(M):
        P[i] = k * pow(rho[i], 1.0 + 1.0/n)
    return P

########################################################################
# 6. Acceleration Calculation (getAcc)
########################################################################

def getAcc(double[:, :] pos, double[:, :] vel, double m, double h,
           double k, double n, double lmbda, double nu):
    """
    Calculate acceleration for each SPH particle.
    Parameters:
      pos   : N x 3 array of positions.
      vel   : N x 3 array of velocities.
      m     : particle mass.
      h     : smoothing length.
      k     : equation-of-state constant.
      n     : polytropic index.
      lmbda : external force constant.
      nu    : viscosity.
    Returns:
      a : N x 3 array of accelerations.
    """
    cdef int N = pos.shape[0]
    cdef int i, j
    cdef np.ndarray[double, ndim=1] rho = getDensity(pos, pos, m, h)  # Using pos for both sampling and particle positions
    cdef np.ndarray[double, ndim=1] P = getPressure(rho, k, n)
    
    cdef np.ndarray[double, ndim=2] dx, dy, dz
    dx, dy, dz = getPairwiseSeparations(pos, pos)
    
    cdef np.ndarray[double, ndim=2] dWx, dWy, dWz
    dWx, dWy, dWz = gradW(dx, dy, dz, h)
    
    cdef np.ndarray[double, ndim=2] a = np.empty((N, 3), dtype=np.double)
    cdef double[:, :] a_view = a
    cdef double term
    for i in range(N):
        # Initialize acceleration components
        a_view[i, 0] = 0.0
        a_view[i, 1] = 0.0
        a_view[i, 2] = 0.0
        for j in range(N):
            term = (P[i]/(rho[i]*rho[i]) + P[j]/(rho[j]*rho[j]))
            a_view[i, 0] -= m * term * dWx[i, j]
            a_view[i, 1] -= m * term * dWy[i, j]
            a_view[i, 2] -= m * term * dWz[i, j]
        # Subtract external force and viscosity
        a_view[i, 0] -= lmbda * pos[i, 0] + nu * vel[i, 0]
        a_view[i, 1] -= lmbda * pos[i, 1] + nu * vel[i, 1]
        a_view[i, 2] -= lmbda * pos[i, 2] + nu * vel[i, 2]
    return a
