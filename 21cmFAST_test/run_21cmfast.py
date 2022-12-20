import py21cmfast as p21c

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

lc = p21c.run_lightcone(astro_params=astro_params, flag_options=flag_options,redshift=13, max_redshift=28)
lc.save('/home/dm/watson/21cmFAST-data/test_lc.h5')
