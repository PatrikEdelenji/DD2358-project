
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # Force the use of Qt5Agg backend
import matplotlib.pyplot as plt
from scipy.special import gamma
import cProfile, pstats
import sph_cython

# If not running under kernprof, define a dummy decorator.
try:
    profile
except NameError:
    def profile(func):
        return func

@profile
def main(N = 400):
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
    acc = sph_cython.getAcc(pos, vel, m, h, k, n, lmbda, nu)
    
    # Number of timesteps
    Nt = int(np.ceil(tEnd / dt))
    
    # Prepare figure for real-time plotting
    fig = plt.figure(figsize=(4, 5), dpi=80)
    fig.canvas.manager.set_window_title("SPH Simulation with Cython")
    fig.canvas.manager.window.setGeometry(400, 0, 400, 600)
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
        acc = sph_cython.getAcc(pos, vel, m, h, k, n, lmbda, nu)
        vel += acc * dt / 2
        t += dt
        
        # Get density for plotting
        rho = sph_cython.getDensity(pos, pos, m, h)
        
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
            rho_radial = sph_cython.getDensity(rr, pos, m, h)
            plt.plot(rlin, rho_radial, color='blue')
            plt.pause(0.001)
    
    # Add labels/legend and save figure
    # plt.sca(ax2)
    # plt.xlabel('radius')
    # plt.ylabel('density')
    # plt.savefig('sph.png', dpi=240)
    # plt.show()
    
    return 0

if __name__ == "__main__":
    # part = [400,2000,4000,8000]
    part = [4000]
    for i in part:
        print(f"Profiling for {i} particles..")
        profiler = cProfile.Profile()
        profiler.enable()  # Start profiling
        main(i)
        profiler.disable()  # Stop profiling

        # Create a Stats object and sort the results by cumulative time.
        stats = pstats.Stats(profiler).sort_stats("cumtime")
        stats.print_stats(8)
