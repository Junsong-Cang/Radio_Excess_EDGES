# Sample EDGES likelihood using EMCEE using my interpretation of Chi2
from edges_estimate.likelihoods import LinearFG
from edges_cal.modelling import LinLog
import attr
import h5py
# yabf: for MCMC
from yabf import Component, Parameter
import py21cmfast as p21c
import numpy as np
import os
import emcee
from functools import cached_property 
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import time
import random
import shutil

nwalkers = 4
n_samples = 100000
ndim = 2
fR_Range = {'min':1,'max':7,'start':5.05}
LX_Range = {'min':37,'max':45,'start':42.0}

File_Root = "EMCEE_V1/Pop_II_Test"
# ChainFile = "Chains.h5"
# ConvergeFile='Status.h5'

# ---- Convergence Stats ----
# Check convergence every N iterations
Check_Interv = 10
# Convergence Ctiteria
Converge_Thresh = 100

# ---- Initialise ----
ChainFile = 'data/'+File_Root+'_EMCEE_Chains.h5'
ConvergeFile = 'data/'+File_Root+'_EMCEE_Status.h5'
Getdist_ChainFile = 'data/'+File_Root+'_EMCEE.txt'
Getdist_Range_File =  'data/'+File_Root+'_EMCEE.range'

print('ChainFile = ', ChainFile)
print('ConvergeFile = ', ConvergeFile)
print('Getdist_ChainFile = ', Getdist_ChainFile)
print('Getdist_Range_File = ', Getdist_Range_File)

data = np.genfromtxt("data/Data_EDGES.txt")
try:
    os.remove(ChainFile)
    os.remove(ConvergeFile)
    os.remove(TxtChainFile)
    os.remove(Getdist_Range_File)
except FileNotFoundError:
	pass
F=open(Getdist_Range_File,'w')
print("fR       ", "{0:.5E}".format(fR_Range['min']),  "    {0:.5E}".format(fR_Range['max']), file=F)
print("LX       ", "{0:.5E}".format(LX_Range['min']),  "    {0:.5E}".format(LX_Range['max']), file=F)
F.close()

# ---- Define Functions ----
@attr.s
class AbsorptionProfile(Component):
    provides = ["eor_spectrum"]
    # Obviously, put in the parameters of 21cmFAST below. Each can take a min and max. 
    # The parameters defined here are just the "possible" parameters to fit, they
    # don't define the actively fit parameters in the likelihood itself. 
    # As a pre-flight test, let's fit fR and L_X param first
    # CJS: How does yabf know whether fR and L_X are astro (and not cosmo)?
    base_parameters = [
        Parameter(name="fR", fiducial=5.05, min=1.0, max=7.0, latex="f_{R}"),
        Parameter(name="L_X", fiducial=42.0, min=37.0, max=45.0, latex="L_x"),
    ]

    observed_redshifts: np.ndarray = attr.ib(kw_only=True, eq=attr.cmp_using(eq=np.array_equal))
    
    # Add other initialization parameters you require here, using attr.ib(). Eg.
    user_params: p21c.UserParams = attr.ib()
    cosmo_params: p21c.CosmoParams = attr.ib()
    flag_options: p21c.FlagOptions = attr.ib()
    astro_params: p21c.AstroParams = attr.ib()    
    run_lightcone_kwargs: dict = attr.ib(factory=dict)
    # set default in attr.ib()
    cache_loc = attr.ib(default="/home/dm/watson/21cmFAST-data/cache/")

    # You probably want to run the initial conditions etc and cache them...
    @cached_property
    def initial_conditions(self):
        return p21c.initial_conditions(
            user_params=self.user_params, 
            cosmo_params=self.cosmo_params, 
            direc=self.cache_loc
        )

    def calculate(self, ctx, **params):
        # This is the thing that actually has to calculate the global signal.
        # "params" is a dictionary of {name: value}, where the "name" is one of the 
        # names defined in your base_parameters.
        
        # So, something like this...
        self.astro_params.update(**params)
        lc = p21c.run_lightcone(
            astro_params=self.astro_params, 
            init_box = self.initial_conditions, 
            flag_options=self.flag_options,
            direc=self.cache_loc, 
            redshift=self.observed_redshifts.min(), 
            max_redshift=self.observed_redshifts.max(),
            **self.run_lightcone_kwargs
        )
        # spline requires z to be in increasing order
        z = lc.node_redshifts[-1:0:-1]
        T21 = lc.global_brightness_temp[-1:0:-1]
        return spline(z, T21)(self.observed_redshifts)

    def spectrum(self, ctx, **params):
        # Don't change this.
        return ctx["eor_spectrum"]

def log_likelihood(theta):
    global data, fR_Range, LX_Range
    fR, L_X = theta
    freq = data[:, 0]
    wght = data[:, 1]
    tsky = data[:, 2]
    freq = freq[wght>0]
    tsky = tsky[wght>0]
    
    # Let's fix these params around the best-guess settings
    user_params = p21c.UserParams(
        BOX_LEN = 150,
        HII_DIM = 20, # Should be at least 50 for the official run
        N_THREADS = 10
        )
    astro_params = p21c.AstroParams(
        F_STAR10 = -0.8,
        F_STAR7_MINI = -2.0,
        M_TURN = 7.1,
        NU_X_THRESH = 200,
        t_STAR = 0.3
        )
    flag_options = p21c.FlagOptions(
        USE_MASS_DEPENDENT_ZETA = True,
        USE_TS_FLUCT = True,
        INHOMO_RECO = True,
        USE_RADIO_ACG = True
    )
    
    eor = AbsorptionProfile(
        observed_redshifts = 1420/freq - 1,
        user_params = user_params,
        cosmo_params = p21c.CosmoParams(),
        flag_options = flag_options,
        astro_params = astro_params,
        params = {
            'fR': {'min': fR_Range['min'], 'max': fR_Range['max']}, 
            'L_X':{'min': LX_Range['min'],'max': LX_Range['max']}
        }, # these are the params that are actually fit. The names have to be in the `base_parameters` above
        cache_loc = '/home/dm/watson/21cmFAST-data/cache/',
        run_lightcone_kwargs = {"ZPRIME_STEP_FACTOR": 1.03,"write":False}
        )

    fg_model = LinLog(n_terms=5)

    my_likelihood = LinearFG(freq=freq, t_sky=tsky, var=0.03**2, fg=fg_model, eor=eor)

    # Then call the likelihood like this:
    LnL = my_likelihood.partial_linear_model.logp(params=[fR, L_X]) # params here should be fiducials for params you want to fit
    return LnL

def log_prior(theta):
  global fR_Range, LX_Range
  fR, L_X = theta
  if fR_Range['min'] < fR < fR_Range['max'] and LX_Range['min'] < L_X < LX_Range['max']:
    return 0.0
  else:
    return -np.inf

def log_probability(theta):
  lp = log_prior(theta)
  if not np.isfinite(lp):
    LogP = -np.inf
  else:
    LogP = lp + log_likelihood(theta)
  # Print out getdist chains in equal weight
  Chi2 = -2 * LogP
  F=open(TxtChainFile,'a')
  Weight = 1.0
  fR, LX = theta
  print("{0:.5E}".format(Weight), "    {0:.5E}".format(Chi2), "    {0:.5E}".format(fR), "    {0:.5E}".format(LX), file=F)
  F.close()
  return LogP

Start_Location=[fR_Range['start'], LX_Range['start']] + 1e-4 * np.random.randn(nwalkers, ndim)
backend = emcee.backends.HDFBackend(ChainFile)
backend.reset(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,backend=backend)
# We'll track how the average autocorrelation time estimate changes
index = 0
autocorr = np.empty(n_samples)
# This will be useful to testing convergence
old_tau = np.inf

# Now we'll sample for up to n_samples steps
for sample in sampler.sample(Start_Location, iterations=n_samples, progress=True):
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
