# optimized_funcs.pyx
import numpy as np
cimport numpy as cnp
from libc.math cimport exp, pow, M_PI

def optimized_n(cnp.ndarray[cnp.float64_t, ndim=2] r, double h):
    """
    Optimized calculation for:
    n = -2 * np.exp( -r**2 / h**2) / h**5 / (np.pi)**(3/2)
    """
    cdef int M = r.shape[0]
    cdef int N = r.shape[1]
    cdef int i, j
    cdef cnp.ndarray[cnp.float64_t, ndim=2] n = np.zeros((M, N), dtype=np.float64)

    for i in range(M):
        for j in range(N):
            n[i, j] = -2 * exp(-pow(r[i, j], 2) / pow(h, 2)) / pow(h, 5) / pow(M_PI, 1.5)

    return n


def optimized_separations(cnp.ndarray[cnp.float64_t, ndim=2] rix,
                          cnp.ndarray[cnp.float64_t, ndim=2] riy,
                          cnp.ndarray[cnp.float64_t, ndim=2] riz,
                          cnp.ndarray[cnp.float64_t, ndim=2] rjx,
                          cnp.ndarray[cnp.float64_t, ndim=2] rjy,
                          cnp.ndarray[cnp.float64_t, ndim=2] rjz):
    """
    Optimized matrix subtraction:
    dx = rix - rjx.T
    dy = riy - rjy.T
    dz = riz - rjz.T
    """
    cdef int M = rix.shape[0]
    cdef int N = rjx.shape[0]
    cdef int i, j

    cdef cnp.ndarray[cnp.float64_t, ndim=2] dx = np.zeros((M, N), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] dy = np.zeros((M, N), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] dz = np.zeros((M, N), dtype=np.float64)

    for i in range(M):
        for j in range(N):
            dx[i, j] = rix[i, 0] - rjx[j, 0]
            dy[i, j] = riy[i, 0] - rjy[j, 0]
            dz[i, j] = riz[i, 0] - rjz[j, 0]

    return dx, dy, dz


def optimized_density(cnp.ndarray[cnp.float64_t, ndim=2] dx,
                      cnp.ndarray[cnp.float64_t, ndim=2] dy,
                      cnp.ndarray[cnp.float64_t, ndim=2] dz,
                      double m, double h):
    """
    Optimized density calculation:
    rho = np.sum(m * W(dx, dy, dz, h), 1).reshape((M,1))
    """
    cdef int M = dx.shape[0]
    cdef int N = dx.shape[1]
    cdef int i, j
    cdef double density, w
    cdef cnp.ndarray[cnp.float64_t, ndim=2] rho = np.zeros((M, 1), dtype=np.float64)

    for i in range(M):
        density = 0
        for j in range(N):
            w = m * exp(-pow(dx[i, j]**2 + dy[i, j]**2 + dz[i, j]**2, 2) / pow(h, 2))
            density += w
        rho[i, 0] = density 

    return rho
