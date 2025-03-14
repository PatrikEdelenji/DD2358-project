import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # Force the use of Qt5Agg backend
import matplotlib.pyplot as plt
from scipy.special import gamma

# If not running under kernprof, define a dummy decorator.
try:
    profile
except NameError:
    def profile(func):
        return func

@profile
def W(x, y, z, h):
    """
    Gaussian Smoothing kernel (3D)
    x     is a vector/matrix of x positions
    y     is a vector/matrix of y positions
    z     is a vector/matrix of z positions
    h     is the smoothing length
    w     is the evaluated smoothing function
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    w = (1.0 / (h * np.sqrt(np.pi)))**3 * np.exp(-r**2 / h**2)
    return w

@profile
def gradW(x, y, z, h):
    """
    Gradient of the Gaussian Smoothing kernel (3D)
    x, y, z  are vectors/matrices of positions
    h        is the smoothing length
    Returns: wx, wy, wz (gradients)
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    n = -2 * np.exp(-r**2 / h**2) / (h**5 * (np.pi)**(3/2))
    wx = n * x
    wy = n * y
    wz = n * z
    return wx, wy, wz

@profile
def getPairwiseSeparations(ri, rj):
    """
    Get pairwise separations between 2 sets of coordinates.
    ri: an M x 3 matrix of positions.
    rj: an N x 3 matrix of positions.
    Returns: dx, dy, dz (each M x N matrices of separations)
    """
    M = ri.shape[0]
    N = rj.shape[0]
    # Reshape positions for broadcasting
    rix = ri[:, 0].reshape((M, 1))
    riy = ri[:, 1].reshape((M, 1))
    riz = ri[:, 2].reshape((M, 1))
    rjx = rj[:, 0].reshape((N, 1))
    rjy = rj[:, 1].reshape((N, 1))
    rjz = rj[:, 2].reshape((N, 1))
    dx = rix - rjx.T
    dy = riy - rjy.T
    dz = riz - rjz.T
    return dx, dy, dz

@profile
def getDensity(r, pos, m, h):
    """
    Get Density at sampling locations from SPH particle distribution.
    r   : an M x 3 matrix of sampling locations.
    pos : an N x 3 matrix of SPH particle positions.
    m   : particle mass.
    h   : smoothing length.
    Returns: rho, an M x 1 vector of densities.
    """
    dx, dy, dz = getPairwiseSeparations(r, pos)
    rho = np.sum(m * W(dx, dy, dz, h), axis=1).reshape((-1, 1))
    return rho

@profile
def getPressure(rho, k, n):
    """
    Equation of State.
    rho : vector of densities.
    k   : equation of state constant.
    n   : polytropic index.
    Returns: Pressure P.
    """
    P = k * rho**(1 + 1/n)
    return P

@profile
def getAcc(pos, vel, m, h, k, n, lmbda, nu):
    """
    Calculate the acceleration on each SPH particle.
    pos   : N x 3 matrix of positions.
    vel   : N x 3 matrix of velocities.
    m     : particle mass.
    h     : smoothing length.
    k     : equation-of-state constant.
    n     : polytropic index.
    lmbda : external force constant.
    nu    : viscosity.
    Returns: a, N x 3 matrix of accelerations.
    """
    N = pos.shape[0]
    rho = getDensity(pos, pos, m, h)
    P = getPressure(rho, k, n)
    dx, dy, dz = getPairwiseSeparations(pos, pos)
    dWx, dWy, dWz = gradW(dx, dy, dz, h)
    ax = -np.sum(m * (P/rho**2 + P.T/(rho.T**2)) * dWx, axis=1).reshape((N, 1))
    ay = -np.sum(m * (P/rho**2 + P.T/(rho.T**2)) * dWy, axis=1).reshape((N, 1))
    az = -np.sum(m * (P/rho**2 + P.T/(rho.T**2)) * dWz, axis=1).reshape((N, 1))
    a = np.hstack((ax, ay, az))
    a -= lmbda * pos   # external potential force
    a -= nu * vel      # viscosity
    return a

@profile
def main(N=400):
    """ SPH simulation """
    # Simulation parameters
    N = N          # Number of particles
    t = 0            # Current time
    tEnd = 5        # End time
    dt = 0.04        # Timestep
    M = 2            # Star mass
    R = 0.75         # Star radius
    h = 0.1          # Smoothing length
    k = 0.1          # Equation of state constant
    n = 1            # Polytropic index
    nu = 1           # Damping/viscosity
    plotRealTime = True  # Flag for plotting as simulation proceeds

    # Generate Initial Conditions
    np.random.seed(42)
    lmbda = (2 * k * (1 + n) * np.pi**(-3/(2*n)) *
             (M * gamma(5/2+n) / R**3 / gamma(1+n))**(1/n) / R**2)
    m = M / N                    # Particle mass
    pos = np.random.randn(N, 3)  # Random initial positions
    vel = np.zeros_like(pos)     # Initial velocities (zero)
    
    # Calculate initial accelerations
    acc = getAcc(pos, vel, m, h, k, n, lmbda, nu)
    
    # Number of timesteps
    Nt = int(np.ceil(tEnd / dt))
    
    # Prepare figure for real-time plotting
    fig = plt.figure(figsize=(4, 5), dpi=80)
    fig.canvas.manager.set_window_title("SPH Simulation")
    fig.canvas.manager.window.setGeometry(0, 0, 400, 600)
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0:2, 0])
    ax2 = plt.subplot(grid[2, 0])
    
    rr = np.zeros((100, 3))
    rlin = np.linspace(0, 1, 100)
    rr[:, 0] = rlin
    rho_analytic = lmbda / (4 * k) * (R**2 - rlin**2)
    
    # Simulation Main Loop
    for i in range(Nt):
        # Leapfrog integration: half-kick, drift, update acceleration, half-kick.
        vel += acc * dt / 2
        pos += vel * dt
        acc = getAcc(pos, vel, m, h, k, n, lmbda, nu)
        vel += acc * dt / 2
        t += dt
        
        # Get density for plotting
        rho = getDensity(pos, pos, m, h)
        
        # Plot in real time (or at the final timestep)
        if plotRealTime or (i == Nt - 1):
            plt.sca(ax1)
            plt.cla()
            cval = np.minimum((rho - 3) / 3, 1).flatten()
            plt.scatter(pos[:, 0], pos[:, 1], c=cval, cmap=plt.cm.autumn,
                        s=10, alpha=0.5)
            ax1.set(xlim=(-1.4, 1.4), ylim=(-1.2, 1.2))
            ax1.set_aspect('equal', 'box')
            ax1.set_xticks([-1, 0, 1])
            ax1.set_yticks([-1, 0, 1])
            ax1.set_facecolor((0.1, 0.1, 0.1))
            
            plt.sca(ax2)
            plt.cla()
            ax2.set(xlim=(0, 1), ylim=(0, 3))
            ax2.set_aspect(0.1)
            plt.plot(rlin, rho_analytic, color='gray', linewidth=2)
            rho_radial = getDensity(rr, pos, m, h)
            plt.plot(rlin, rho_radial, color='blue')
            plt.pause(0.001)
    
    return 0

if __name__ == "__main__":
    main(400)
