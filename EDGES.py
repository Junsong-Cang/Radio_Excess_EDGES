# Magrinalise EDGES nuisance params

from edges_estimate.likelihoods import LinearFG
from edges_cal.modelling import LinLog
import attr

# yabf: for MCMC
from yabf import Component, Parameter
from yabf.samplers import polychord
import py21cmfast as p21c
import numpy as np
from functools import cached_property 
from scipy.interpolate import InterpolatedUnivariateSpline as spline

print('Do conda activate base first')

@attr.s
class AbsorptionProfile(Component):
    provides = ["eor_spectrum"]
    # Obviously, put in the parameters of 21cmFAST below. Each can take a min and max. 
    # The parameters defined here are just the "possible" parameters to fit, they
    # don't define the actively fit parameters in the likelihood itself. 
    # As a pre-flight test, let's fit fR and L_X param first
    # CJS: How does yabf know whether fR and L_X are astro (and not cosmo)?
    base_parameters = [
        Parameter(name="fR", fiducial=2.0, min=1.0, max=6.0, latex="f_{R}"),
        Parameter(name="L_X", fiducial=37.0, min=35.0, max=42.0, latex="L_x"),
    ]

    observed_redshifts: np.ndarray = attr.ib(kw_only=True, eq=attr.cmp_using(eq=np.array_equal))
    
    # Add other initialization parameters you require here, using attr.ib(). Eg.
    user_params: p21c.UserParams = attr.ib()
    cosmo_params: p21c.CosmoParams = attr.ib()
    flag_options: p21c.FlagOptions = attr.ib()
    astro_params: p21c.AstroParams = attr.ib()    
    run_lightcone_kwargs: dict = attr.ib(factory=dict)
    # cache_loc: str = attr.ib()
    # set default in attr.ib()
    # cache_loc: str = '/home/dm/watson/21cmFAST-data/'
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

if __name__ == '__main__':

    data = np.genfromtxt("Data_EDGES.txt")
    freq = data[:, 0]
    wght = data[:, 1]
    tsky = data[:, 2]
    freq = freq[wght>0]
    tsky = tsky[wght>0]
    
    # Let's fix these params around the best-guess settings
    user_params = p21c.UserParams(
        BOX_LEN = 150,
        HII_DIM = 50, # Should be at least 50 for the official run
        N_THREADS = 1
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
            'fR': {'min': 2.0, 'max': 6.0}, 
            'L_X':{'min':37.0,'max':42.0}
        }, # these are the params that are actually fit. The names have to be in the `base_parameters` above
        cache_loc = '/home/dm/watson/21cmFAST-data/cache/',
        run_lightcone_kwargs = {"ZPRIME_STEP_FACTOR": 1.03}
        )

    fg_model = LinLog(n_terms=5)

    my_likelihood = LinearFG(freq=freq, t_sky=tsky, var=0.03**2, fg=fg_model, eor=eor)

    # Then call the likelihood like this:
    # my_likelihood.logp(params=[2.0, 37.0])  # params here is a list in order of the params you defined in the eor model. You can also pass a dict to make it more explicit.
    
    a = my_likelihood.partial_linear_model.logp(params=[2, 37.0]) # params here should be fiducials for params you want to fit
    print(a)

    sampler = polychord(
        my_likelihood.partial_linear_model,                  # The actual likelihood to sample from
        save_full_config = False,                            # Otherwise would save a YAML file that is hard to read.
        output_dir = "Chains",       # Directory in which to save all the output chains.
        output_prefix = "PopII_Test",                        # A prefix for all files output.
        sampler_kwargs = {                                   # Anything that can be passed to PolychordSettings,
            "nlives": 256                                    # see https://github.com/PolyChord/PolyChordLite/pypolychord/settings.py#L5
        }
    )

    # Actually run the sampling. You'll get a bunch of files in the output dir, that can
    # be read by getdist.
    samples = sampler.sample()               