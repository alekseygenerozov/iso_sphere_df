
import h5py
import numpy as np
from numba import jit,njit
from numba import vectorize

import sys
import cgs_const as cgs
from scipy.stats import maxwell
##Physical constants
G= 43007.1
M_scale=1e10*cgs.M_sun
r_scale=1e3*cgs.pc
v_scale=(cgs.G/G*M_scale/r_scale)**0.5
##Velocity dispersion of isothermal sphere. (Converted to code units)
v0=2e7/v_scale

def gen_ran_vector(n):
	'''
	Generate n isotropic vectors.
	'''
	phi=np.random.uniform(0.,2.*np.pi, size=n)
	ct=np.random.uniform(-1,1,size=n)
	st=(1.-ct**2.)**0.5
	return np.transpose([st*np.cos(phi), st*np.sin(phi), ct])

def gen_iso_sphere(n, v0, rmax=10.):
	'''
	Generate particle velocities and positions for an isothermal sphere
	'''
	pos=np.zeros([n, 3])
	vel=np.zeros([n, 3])

	r=np.random.uniform(0,rmax, size=n)
	v=maxwell.rvs(scale=v0, size=n)
	pos=gen_ran_vector(n)*np.expand_dims(r, axis=1)
	vel=gen_ran_vector(n)*np.expand_dims(v, axis=1)
	##Total mass of iso sphere (from integrating density....)
	mtot=4.*np.pi*v0**2./(2*np.pi*G)*rmax
	return pos, vel, mtot/n


############################################
@vectorize('float64(float64)')
def smooth(q):
	'''
	Smoothing kernel
	'''
	if q<0.5:
		return 32/3.*q**3.-192/5.*q**5.+32.*q**6.
	elif q<1:
		return -1./15.+64./3.*q**3.-48.*q**4.+192./5.*q**5.-32./3.*q**6.
	else:
		return 1.

# @jit('float64[:](float64[:], float64[:], float64, float64[:,:], float64[:,:])')
def get_acc(pos, vel, M, m, pos_all, vel_all):
	'''
	Calculate acceleration of particle of mass M, 
	with postion pos and velocity vel due other particles
	whose positions and velocities are stores in pos_all
	and vel_all
	'''
	##Relative velocity 
	Vi=vel-vel_all
	##Relative positions
	ri=pos-pos_all
	##Distances (The operation below is essentially a row-by-row dot product.)
	ri_mag=((ri*ri).sum(axis=1, dtype=np.float64))**0.5
	Vi_mag=((Vi*Vi).sum(axis=1, dtype=np.float64))**0.5
	##Smoothing function
	smooths=smooth(ri_mag/1e-5)

	rhati=ri/np.expand_dims(ri_mag, axis=1)
	vhati=Vi/np.expand_dims(Vi_mag, axis=1)
	##auxilliary quantities in df formula
	tmp=(rhati*vhati).sum(axis=1)
	tmp=(rhati-vhati*np.expand_dims(tmp, axis=1))
	bi=ri_mag*((tmp*tmp).sum(axis=1, dtype=np.float64))**0.5
	alpha_i=bi*Vi_mag**2./(G*M)
	##Smoothing kernel
	pre1=smooths*G*m/ri_mag**2.
	pre2=alpha_i*bi/(1.+alpha_i**2.)/ri_mag*smooths*G*m/ri_mag**2.
	##normal acceleration and df acceleration (note sign of the normal acceleration is opposite from Ma et al. 2021. However this way makes for physical sense.)
	aext=-(rhati*np.expand_dims(pre1, axis=1)).sum(axis=0, dtype=np.float64)
	adf=-(vhati*np.expand_dims(pre2, axis=1)).sum(axis=0, dtype=np.float64)

	return aext+adf

##############################################
##Generate synthetic isothermal sphere
pos_all, vel_all, m=gen_iso_sphere(int(8e6), v0)
# ##Initial condition. Circular orbit
pos=np.array([1.,0,0])
vel=np.array([0,v0*2.**0.5,0])
#Total time, stating time, time-step (command line argument)
t_max=0.5
t=0
dt=float(sys.argv[1])

pos_test=[]
while t<t_max:
	##Distance to all of the other particles...
	acc1=get_acc(pos, vel, 1e8/1e10, m, pos_all, vel_all)
	acc=acc1
	##Leapfrog
	vel+=acc*dt
	pos+=vel*dt
	# pos_test.append(pos)
	t+=dt
	print('{0} {1} {2} {3} {4} {5} {6}'.format(t, pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]))
	sys.stdout.flush()
##############################################

