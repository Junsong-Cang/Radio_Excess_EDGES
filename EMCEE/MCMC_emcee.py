import h5py
import time
import numpy as np
import os
import emcee
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import matplotlib.pyplot as plt
import py21cmfast as p21c

nwalkers = 4
n_samples = 100000

ndim = 2
fR_Range = {'min':1,'max':6}
LX_Range = {'min':38,'max':44}

ChainFile = "Chains.h5"
ConvergeFile='Status.h5'

# ---- Convergence Stats ----
# Check convergence every N iterations
Check_Interv = 10
# Convergence Ctiteria
Converge_Thresh = 100

# -------- Start --------
Data = np.genfromtxt("Data_EDGES_2.txt")

# z=Data[:,7]
# Er=Data[:,8]
# T21=Data[:,5]

try:
  os.remove(ChainFile)
  os.remove(ConvergeFile)
except FileNotFoundError:
  pass

def Get_LightCone(fR=0,L_X=40):
  global Data
  z=Data[:,7]
  zmin=np.min(z)
  zmax=np.max(z)
  os.system('echo ---- >> log.txt')

  user_params = p21c.UserParams(
    BOX_LEN = 500,
    HII_DIM = 20,
    N_THREADS = 10
  )

  astro_params = p21c.AstroParams(
    F_STAR10 = -0.8,
    F_STAR7_MINI = -2.0,
    M_TURN = 7.1,
    NU_X_THRESH = 200,
    t_STAR = 0.3,
    fR = fR,
    L_X = L_X
  )

  flag_options = p21c.FlagOptions(
    USE_MASS_DEPENDENT_ZETA = True,
    USE_TS_FLUCT = True,
    INHOMO_RECO = True,
    USE_RADIO_ACG = True
  )
  lc = p21c.run_lightcone(astro_params=astro_params, flag_options=flag_options,redshift=zmin, max_redshift=zmax)
  return lc

def Get_T21(fR=0,L_X=40):
  global Data
  z=Data[:,7]
  lc = Get_LightCone(fR, L_X)
  return spline(lc.node_redshifts, lc.global_brightness_temp)(z)


def log_likelihood(theta):
  'This is lnL'
  fR, L_X = theta
  global Data
  Sigma=Data[:,8]
  T21_data=Data[:,5]
  T21=Get_T21(fR,L_X)
  return -0.5 * np.sum(((T21-T21_data)**2)/(Sigma**2))

def log_prior(theta):
  fR, L_X = theta
  if fR_Range['min'] < fR < fR_Range['max'] and LX_Range['min'] < L_X < LX_Range['max']:
    return 0.0
  else:
    return -np.inf

def log_probability(theta):
  lp = log_prior(theta)
  if not np.isfinite(lp):
    return -np.inf
  else:
    return lp + log_likelihood(theta)

# -------- MCMC --------
fR_Mid=(fR_Range['min']+fR_Range['max'])/2
LX_Mid=(LX_Range['min']+LX_Range['max'])/2
coords=[fR_Mid, LX_Mid] + 1e-4 * np.random.randn(nwalkers, ndim)
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

