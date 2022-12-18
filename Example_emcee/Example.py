import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import os
import emcee
import corner
import math

SigmaNeff=0.15 # Global
NSamples=100000
unit_la=8
param_a=[0.0, 10.0]
param_la=[6, 12]
param_lk=[1, 23]
param_ls=[-2, 1]
nwalkers=16

# ---- 
TabFile='data/dN_table.h5'
h5f=h5py.File(TabFile)
Tab=h5f['/dN_table'] # Global
TabSize=h5f['/Size']
# Interp Tab axis sizes
na=int(TabSize[0]) # Global
ns=int(TabSize[1]) # Global
nk=int(TabSize[2]) # Global
# Interp Tab axis range
la1=2 # Global
la2=12 # Global
ls1=-2 # Global
ls2=1 # Global
lk1=1 # Global
lk2=23.2227 # Global
# Interp Tab resolution
dla=(la2-la1)/(na-1) # Global
dls=(ls2-ls1)/(ns-1) # Global
dlk=(lk2-lk1)/(nk-1) # Global

def dNeff(A,ls,lk):
	' Python interpolation function for matlab dNeff '
	
	global Tab, na, ns, nk, la1, la2, ls1, ls2, lk1, lk2, SigmaNeff, dla, dls, dlk
	# ensure inputs are float
	if A < 0:
		A = 1E-6
	la=math.log10(A)+unit_la
	x=float(la)
	y=float(lk)
	z=float(ls)
	
	# Find neighboring indexes
	IdxA1=math.floor((la-la1)/dla)
	IdxK1=math.floor((lk-lk1)/dlk)
	IdxS1=math.floor((ls-ls1)/dls)
	if IdxA1 < 0:
		IdxA1=0
	if IdxA1 >= na-1:
		IdxA1=na-2
	if IdxK1 < 0:
		IdxK1=0
	if IdxK1 >= nk-1:
		IdxK1=nk-2
	if IdxS1 < 0:
		IdxS1=0
	if IdxS1 >= ns-1:
		IdxS1=ns-2
	IdxA2=IdxA1+1
	IdxK2=IdxK1+1
	IdxS2=IdxS1+1

	x1=la1+dla*IdxA1;
	x2=la1+dla*IdxA2;
	y1=lk1+dlk*IdxK1;
	y2=lk1+dlk*IdxK2;
	z1=ls1+dls*IdxS1;
	z2=ls1+dls*IdxS2;

	f11=Tab[IdxA1,IdxK1,IdxS1]
	f12=Tab[IdxA1,IdxK2,IdxS1]
	f21=Tab[IdxA2,IdxK1,IdxS1]
	f22=Tab[IdxA2,IdxK2,IdxS1]

	h11=Tab[IdxA1,IdxK1,IdxS2]
	h12=Tab[IdxA1,IdxK2,IdxS2]
	h21=Tab[IdxA2,IdxK1,IdxS2]
	h22=Tab[IdxA2,IdxK2,IdxS2]

	f1=(f21-f11)*(y-y1)/(y2-y1)+f11
	f2=(f22-f12)*(y-y1)/(y2-y1)+f12
	N1=(f2-f1)*(x-x1)/(x2-x1)+f1
	
	h1=(h21-h11)*(y-y1)/(y2-y1)+h11
	h2=(h22-h12)*(y-y1)/(y2-y1)+h12
	N2=(h2-h1)*(x-x1)/(x2-x1)+h1
	
	N=(N2-N1)*(z-z1)/(z2-z1)+N1
	return N

def log_likelihood(theta):
	'This is lnL, does not seem to matter even if I used one-sided'
	A, ls, lk = theta
	global SigmaNeff
	dN=dNeff(A,ls,lk)
	if dN < 0 or A < 0:
		return -np.inf
	else:
		return -0.5 * (dN**2)/(SigmaNeff**2)

def log_prior(theta):
	A, ls, lk = theta
	#if A < 1.0E-6:
	#	la=unit_la - 6
	#else:
	#	la=math.log10(A)+unit_la
	#if param_la[0] < la < param_la[1] and param_ls[0] < ls < param_ls[1] and param_lk[0] < lk < param_lk[1]:
	if param_a[0] < A < param_a[1] and param_ls[0] < ls < param_ls[1] and param_lk[0] < lk < param_lk[1]:
		return 0.0
	else:
		return -np.inf

def log_probability(theta):
	lp = log_prior(theta)
	if not np.isfinite(lp):
		return -np.inf
	else:
		return lp + log_likelihood(theta)

pos=[0.0, param_ls[0], param_lk[0]] + 1e-4 * np.random.randn(nwalkers, 3)

nwalkers, ndim = pos.shape
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)

sampler.run_mcmc(pos, NSamples, progress=True);

ig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["$10^{-8} A$", "$log_{10}\sigma$","$log_{10}k_{bh}$"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
ig.savefig('p1.png') 

flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print(flat_samples.shape)

fig = corner.corner(
    flat_samples, 
    labels=labels,
    color='b',
    show_titles=True,
    levels=(0.95,), # for 2D
    quantiles=(0.0,0.95), # for 1D
    bins=40,
    smooth=1,
    smooth1d=1,
    )
fig.savefig("p2.png")