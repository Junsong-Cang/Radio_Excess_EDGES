# Get inputs and define power spectra function
import numpy as np
import h5py
import os
from powerbox.tools import get_power
import py21cmfast as p21c

cmd=input()
exec(cmd)
#FileName
#L_X
#fR

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

# ---- Run 21cmFAST ----
LC_Quantities = ('brightness_temp','Trad_box')
GLB_Quantities = ('brightness_temp','Trad_box')

Data=p21c.run_lightcone(
    redshift=13.0,
    max_redshift=28.0,
    user_params=user_params,
    astro_params=astro_params,
    flag_options=flag_options,
    lightcone_quantities=LC_Quantities,
    global_quantities=GLB_Quantities
    )
Data.save(FileName)