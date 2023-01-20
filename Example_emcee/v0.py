import matplotlib.pyplot as plt
from multiprocessing import Pool
import h5py
import time
import numpy as np
import os
import emcee

nwalkers = 6
ndim = 3

p1_range = {'min':-10,'max':10, 'start':1}
p2_range = {'min':-10,'max':10, 'start':1}
p3_range = {'min':-10,'max':10, 'start':1}

n_samples = 10000
ChainFile = "data/Chains.h5"
TxtChainFile = "data/Example_EMCEE.txt"
Getdist_Range_File = "data/Example_EMCEE.range"
ConvergeFile='data/Status.h5'

# -- Convergence Stats --
# Check convergence every N iterations
Check_Interv = 10
# Convergence Ctiteria
Converge_Thresh = 100
PrintGetdist = True

# ---- Get Fake Data ----
p1=2
p2=-3
p3=1.2

def Model(p1,p2,p3,x):
	return p1+p2*x+p3*x**2

x_data=np.arange(0,3,0.1)
y_data=Model(p1,p2,p3,x_data)
sigma_data=0.2 + 0.4*np.random.rand(len(x_data))

# ---- Define Likelihood and Prior ----
def log_likelihood(theta):
	'This is lnL'
	global x_data, y_data, sigma_data
	p1, p2, p3 = theta
	y=Model(p1,p2,p3,x_data)
	return -0.5 * np.sum(((y-y_data)**2)/(sigma_data**2))

def log_prior(theta):
	p1, p2, p3 = theta
	global p1_range, p2_range, p3_range
	if p1_range['min'] < p1 < p1_range['max'] and p2_range['min'] < p2 < p2_range['max'] and p3_range['min'] < p3 < p3_range['max']:
		return 0.0
	else:
		return -np.inf

def log_probability(theta):
    global TxtChainFile, PrintGetdist
    LogP = log_prior(theta) + log_likelihood(theta)
    if PrintGetdist:
        F=open(TxtChainFile,'a')
        Wight = 0.9999
        p1, p2, p3 = theta
        Chi2 = -2 * LogP
        print("{0:.5E}".format(Wight), "    {0:.5E}".format(Chi2), "    {0:.5E}".format(p1), "    {0:.5E}".format(p2), "    {0:.5E}".format(p3), file=F)
        F.close()
    return LogP

# ---- Initialise ----
try:
    os.remove(ChainFile)
    # os.remove(ConvergeFile)
    os.remove(TxtChainFile)
    os.remove(Getdist_Range_File)
except FileNotFoundError:
	pass
Start_Location=[p1_range['start'], p2_range['start'], p3_range['start']] + 1e-4 * np.random.randn(nwalkers, ndim)
backend = emcee.backends.HDFBackend(ChainFile)
# backend.reset(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,backend=backend)
sampler.run_mcmc(Start_Location, n_samples, progress=True)
