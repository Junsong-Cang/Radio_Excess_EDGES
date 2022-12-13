from edges_estimate.likelihoods import LinearFG
from edges_cal.modelling import LinLog
import attr
from yabf import Component, Parameter
import py21cmfast as p21c
import numpy as np
from functools import cached_property
from scipy.interpolate import InterpolatedUnivariateSpline as spline


@attr.s
class AbsorptionProfile(Component):
  provides = ["eor_spectrum"]
  # Obviously, put in the parameters of 21cmFAST below. Each can take a min and max
  # The parameters defined here are just the "possible" parameters to fit, they
  # don't define the actively fit parameters in the likelihood itself.
  base_parameters = [Parameter("HII_EFF_FACTOR", fiducial = 30.0, min=0, latex = r"\zeta"),
  base_parameters = [Parameter("t_STAR", fiducial = 0.5, min=0.01, latex = "T_{star}"),
  ]

  observed_redshifts: np.ndarray = attr.ib(kw_only = True, eq = attr.cmp_using(eq = np.array_equal))
  # add other initialisation parameters you require here, using attr.ib(). Eg.
  user_params: p21c.UserParams = attr.ib()
  astro_params: p21c.AstroParams = attr.ib()
  cosmo_params: p21c.CosmoParams = attr.ib()
  flag_options: p21c.FlagOptions = attr.ib()
  # zprime_step_factor = attr.ib(default=1.03)
  cache_loc: str = attr.ib()
  run_lightcone_kwargs: dict=attr.ib(factory=dict)

  # You probably want to run the initial conditions etc and cache them...
  @cached_property
  def initial_conditions(self):
    return p21c.initial_conditions(
      user_params=self.user_params,
      cosmo_params=self.cosmo_params,
      direc=self.cache_loc)

  def calculate(self,ctx,**params):
    # This is the thing that actually has to calculate the global signal.
    # "params" is a directory of {name:value}, where the "name" is one of the names defined in your base parameters.
    self.astro_params.update(**params)
    lc=p21c.run_lightcone(
      astro_params=self.astro_params,
      init_box=self.initial_conditions,
      flag_options=self.flag_options,
      direc=self.cache_loc,
      redshift=self.observed_redshifts.min(),
      max_redshift=self.observed_redshifts.max(),
      ZPRIME_STEP_FACTOR = self.zprime_step_factor,
      **self.run_lightcone_kwargs
      )
    # Spline interpolate the node_redshifts to find T21 at the observed redshift
    return spline(lc.node_redshifts,lc.global_brightness_temp)(self.observed_redshifts)

  def spectrum(self,ctx,**params):
    # Don't change this
    return ctx["eor_spectrum"]

if __name__=='__main__':

  data = np.genfromtxt("my_edges_data_file.txt")
  freq = data[:,0]
  tsky = data[:,1] # t_sky?
  wght = data[:,2] # This should be 0 and 1

  freq = freq[wght>1]
  tsky = tsky[wght>1]

  # The component that actually calculates the signal
  eor = AbsorptionProfile(
    observed_redshifts = 1420/freq-1,
    user_params=p21c.UserParams(...),
    cosmo_params=p21c.CosmoParams(),
    flag_options=p21c.FlagOptions(),
    # The following are the params that are actually fitted, must be pre-defined in class AbsorptionProfile(Component)
    # The names have to be in the 'base_parameters' above
    params = {'HII_EFF_FACTOR':{'min':25,'max':40.0}},
    cache_loc='.'
    run_lightcone_kwargs = {
      "ZPRIME_STEP_FACTOR":1/03,
    }
  )
  
  fg_model = LinLog(m_terms = 5)

  # contains the data info
  my_likelihood = LinearFG(freq,t_sky,sigma=0.03,fg=fg_model,eor=eor)

  # Then call this likelihood like this:
  my_likelihood.logp(params=[30.0]) # params here is a list in order of the params you defined, you can also pass a dict to make it more explicit



