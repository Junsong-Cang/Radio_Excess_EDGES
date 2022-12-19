import h5py
import time
import numpy as np
import os
import emcee

nwalkers = 4
ndim = 2
n_samples = 100000
ChainFile = "Chains.h5"
ConvergeFile='Status.h5'

# ---- Convergence Stats ----
# Check convergence every N iterations
Check_Interv = 10
# Convergence Ctiteria
Converge_Thresh = 100

FakeData = np.genfromtxt("Fake_Data.txt")
Sigma=0.5

# ---- Initialise
try:
	os.remove(ChainFile)
	os.remove(ConvergeFile)
except FileNotFoundError:
	pass

def Signal(a,w):
	global FakeData
	x=FakeData[:,0]
	# y = a*np.sin(w*x+0.5)
	# os.system('echo -- >> log.txt')
	y = a*x+w
	#time.sleep(0.01)
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

coords=[0.1, 0.1] + 1e-4 * np.random.randn(nwalkers, ndim)
backend = emcee.backends.HDFBackend(ChainFile)
backend.reset(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,backend=backend)

# We'll track how the average autocorrelation time estimate changes
index = 0
autocorr = np.empty(n_samples)
# This will be useful to testing convergence
old_tau = np.inf

# Now we'll sample for up to n_samples steps
for sample in sampler.sample(coords, iterations=n_samples, progress=True):
    # Only check convergence every 100 steps
    if sampler.iteration % Check_Interv:
        continue

    # Compute the autocorrelation time so far
    # Using tol=0 means that we'll always get an estimate even
    # if it isn't trustworthy
    tau = sampler.get_autocorr_time(tol=0)
    autocorr[index] = np.mean(tau)
    index += 1

    h5f=h5py.File(ConvergeFile, 'w')
    h5f.create_dataset('autocorr', data=autocorr)
    h5f.create_dataset('index', data=index)
    h5f.close()

    # Check convergence
    converged = np.all(tau * Converge_Thresh < sampler.iteration)
    converged &= np.all(np.abs(old_tau - tau) / tau < 1/Converge_Thresh)
    if converged:
        break
    old_tau = tau
