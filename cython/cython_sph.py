import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.special import gamma
import time
from functools import wraps
from collections import defaultdict

from Cython.Build.Inline import cython_inline
import optimized_funcs # type: ignore

import statistics
import cProfile


"""
Create Your Own Smoothed-Particle-Hydrodynamics Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate the structure of a star with SPH
"""

timing_stats = defaultdict(lambda: {"count": 0, "total": 0, "min": float('inf'), "max": float('-inf')})


def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time_ns() * 1e-9
        result = fn(*args, **kwargs)
        t2 = time.time_ns() * 1e-9
        elapsed = t2 - t1

        # Update statistics
        stats = timing_stats[fn.__name__]
        stats["count"] += 1
        stats["total"] += elapsed
        stats["min"] = min(stats["min"], elapsed)
        stats["max"] = max(stats["max"], elapsed)

        print_timing_stats()
        return result

    return measure_time

def print_timing_stats():
    """ Print timing summary for all measured functions """
    print("\nFunction Execution Summary:")
    for func_name, stats in timing_stats.items():
        avg_time = stats["total"] / stats["count"] if stats["count"] > 0 else 0
        print(f"{func_name}: Min: {stats['min']:.6f}s, Max: {stats['max']:.6f}s, Avg: {avg_time:.6f}s, Calls: {stats['count']}")

# @profile
def W( x, y, z, h ):
	"""
    Gausssian Smoothing kernel (3D)
	x     is a vector/matrix of x positions
	y     is a vector/matrix of y positions
	z     is a vector/matrix of z positions
	h     is the smoothing length
	w     is the evaluated smoothing function
	"""
	
	r = np.sqrt(x**2 + y**2 + z**2)
	
	w = (1.0 / (h*np.sqrt(np.pi)))**3 * np.exp( -r**2 / h**2)
	
	return w
	
# @profile
def gradW( x, y, z, h ):
	"""
	Gradient of the Gausssian Smoothing kernel (3D)
	x     is a vector/matrix of x positions
	y     is a vector/matrix of y positions
	z     is a vector/matrix of z positions
	h     is the smoothing length
	wx, wy, wz     is the evaluated gradient
	"""
	
	r = np.sqrt(x**2 + y**2 + z**2)
	
 
 
	n = optimized_funcs.optimized_n(r, h)  
 
 
	wx = n * x
	wy = n * y
	wz = n * z
	
	return wx, wy, wz
	
# @profile
def getPairwiseSeparations( ri, rj ):
	"""
	Get pairwise desprations between 2 sets of coordinates
	ri    is an M x 3 matrix of positions
	rj    is an N x 3 matrix of positions
	dx, dy, dz   are M x N matrices of separations
	"""
	
	M = ri.shape[0]
	N = rj.shape[0]
	
	# positions ri = (x,y,z)
	rix = ri[:,0].reshape((M,1))
	riy = ri[:,1].reshape((M,1))
	riz = ri[:,2].reshape((M,1))
	
	# other set of points positions rj = (x,y,z)
	rjx = rj[:,0].reshape((N,1))
	rjy = rj[:,1].reshape((N,1))
	rjz = rj[:,2].reshape((N,1))
	
	# matrices that store all pairwise particle separations: r_i - r_j
 
	return optimized_funcs.optimized_separations(rix, riy, riz, rjx, rjy, rjz)
	
# @profile
def getDensity( r, pos, m, h ):
	"""
	Get Density at sampling loctions from SPH particle distribution
	r     is an M x 3 matrix of sampling locations
	pos   is an N x 3 matrix of SPH particle positions
	m     is the particle mass
	h     is the smoothing length
	rho   is M x 1 vector of densities
	"""
	
	# M = r.shape[0] moved to cython file
	
	dx, dy, dz = getPairwiseSeparations( r, pos );
	
	rho = optimized_funcs.optimized_density(dx, dy, dz, m, h)
	
	return rho
	
# @profile
def getPressure(rho, k, n):
	"""
	Equation of State
	rho   vector of densities
	k     equation of state constant
	n     polytropic index
	P     pressure
	"""
	
	P = k * rho**(1+1/n)
	
	return P
	
# @profile
def getAcc( pos, vel, m, h, k, n, lmbda, nu ):
	"""
	Calculate the acceleration on each SPH particle
	pos   is an N x 3 matrix of positions
	vel   is an N x 3 matrix of velocities
	m     is the particle mass
	h     is the smoothing length
	k     equation of state constant
	n     polytropic index
	lmbda external force constant
	nu    viscosity
	a     is N x 3 matrix of accelerations
	"""
	
	N = pos.shape[0]
	
	# Calculate densities at the position of the particles
	rho = getDensity( pos, pos, m, h )
	
	# Get the pressures
	P = getPressure(rho, k, n)
	
	# Get pairwise distances and gradients
	dx, dy, dz = getPairwiseSeparations( pos, pos )
	dWx, dWy, dWz = gradW( dx, dy, dz, h )
	
	# Add Pressure contribution to accelerations
	ax = - np.sum( m * ( P/rho**2 + P.T/rho.T**2  ) * dWx, 1).reshape((N,1))
	ay = - np.sum( m * ( P/rho**2 + P.T/rho.T**2  ) * dWy, 1).reshape((N,1))
	az = - np.sum( m * ( P/rho**2 + P.T/rho.T**2  ) * dWz, 1).reshape((N,1))
	
	# pack together the acceleration components
	a = np.hstack((ax,ay,az))
	
	# Add external potential force
	a -= lmbda * pos
	
	# Add viscosity
	a -= nu * vel
	
	return a
	

# @profile
def main():
	""" SPH simulation """
	start_time1 = time.time()
	# Simulation parameters
	N         = 400    # Number of particles
	t         = 0      # current time of the simulation
	tEnd      = 5     # time at which simulation ends   - REDUCED TIME TO 5
	dt        = 0.04   # timestep
	M         = 2      # star mass
	R         = 0.75   # star radius
	h         = 0.1    # smoothing length
	k         = 0.1    # equation of state constant
	n         = 1      # polytropic index
	nu        = 1      # damping
	plotRealTime = True # switch on for plotting as the simulation goes along
	
	# Generate Initial Conditions
	np.random.seed(42)            # set the random number generator seed
	
	lmbda = 2*k*(1+n)*np.pi**(-3/(2*n)) * (M*gamma(5/2+n)/R**3/gamma(1+n))**(1/n) / R**2  # ~ 2.01
	m     = M/N                    # single particle mass
	pos   = np.random.randn(N,3)   # randomly selected positions and velocities
	vel   = np.zeros(pos.shape)
	
	# calculate initial gravitational accelerations
	acc = getAcc( pos, vel, m, h, k, n, lmbda, nu )
	
	# number of timesteps
	Nt = int(np.ceil(tEnd/dt))
	
	# prep figure
	fig = plt.figure(figsize=(4,5), dpi=80)
	grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
	ax1 = plt.subplot(grid[0:2,0])
	ax2 = plt.subplot(grid[2,0])
	rr = np.zeros((100,3))
	rlin = np.linspace(0,1,100)
	rr[:,0] =rlin
	rho_analytic = lmbda/(4*k) * (R**2 - rlin**2)
	
	# Simulation Main Loop
	for i in range(Nt):
		# (1/2) kick
		vel += acc * dt/2
		
		# drift
		pos += vel * dt
		
		# update accelerations
		acc = getAcc( pos, vel, m, h, k, n, lmbda, nu )
		
		# (1/2) kick
		vel += acc * dt/2
		
		# update time
		t += dt
		
		# get density for plotting
		rho = getDensity( pos, pos, m, h )
		
		# plot in real time
		if plotRealTime or (i == Nt-1):
			plt.sca(ax1)
			plt.cla()
			cval = np.minimum((rho-3)/3,1).flatten()
			plt.scatter(pos[:,0],pos[:,1], c=cval, cmap=plt.cm.autumn, s=10, alpha=0.5)
			ax1.set(xlim=(-1.4, 1.4), ylim=(-1.2, 1.2))
			ax1.set_aspect('equal', 'box')
			ax1.set_xticks([-1,0,1])
			ax1.set_yticks([-1,0,1])
			ax1.set_facecolor('black')
			ax1.set_facecolor((.1,.1,.1))
			
			plt.sca(ax2)
			plt.cla()
			ax2.set(xlim=(0, 1), ylim=(0, 3))
			ax2.set_aspect(0.1)
			plt.plot(rlin, rho_analytic, color='gray', linewidth=2)
			rho_radial = getDensity( rr, pos, m, h )
			plt.plot(rlin, rho_radial, color='blue')
			plt.pause(0.001)
	    
	end_time1 = time.time()	
	secs = end_time1 - start_time1
	print("Whole process took", secs, "seconds")
 
	# add labels/legend
	plt.sca(ax2)
	plt.xlabel('radius')
	plt.ylabel('density')
	
	# Save figure
	plt.savefig('sph.png',dpi=240)
	plt.show()


 
	return 0
	



if __name__== "__main__":
	main()



