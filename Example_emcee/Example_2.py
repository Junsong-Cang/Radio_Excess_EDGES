import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import os
import emcee
import corner
import math

nwalkers = 16
NSamples = 1000

FakeData = np.genfromtxt("Fake_Data.txt")
Sigma=0.5

def Signal(a,w):
	global FakeData
	x=FakeData[:,0]
	# y = a*np.sin(w*x+0.5)
	# os.system('echo -- >> log.txt')
	y = a*x+w
	return y

def log_likelihood(theta):
	'This is lnL'
	global FakeData, Sigma
	y_data=FakeData[:,1]
	a, w = theta
	y=Signal(a,w)
	return -0.5 * np.sum(((y-y_data)**2)/(Sigma**2))

def log_prior(theta):
	a, w = theta
	if 0 < a < 10 and 0 < w < 10:
		return 0.0
	else:
		return -np.inf

def log_probability(theta):
	lp = log_prior(theta)
	if not np.isfinite(lp):
		return -np.inf
	else:
		return lp + log_likelihood(theta)

pos=[0.1, 0.1] + 1e-4 * np.random.randn(nwalkers, 2)
nwalkers, ndim = pos.shape
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
sampler.run_mcmc(pos, NSamples, progress=True)
ig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)

samples = sampler.get_chain()
labels = ["$a$", "$\omega$"]
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