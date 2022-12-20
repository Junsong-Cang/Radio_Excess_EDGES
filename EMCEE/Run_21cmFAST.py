# Get inputs and define power spectra function
import numpy as np
import h5py
import os
from powerbox.tools import get_power
import py21cmfast as p21c

Stop=0
# Definning useful params
default='a very long string'
F=f=N=n=False
T=t=Y=y=True

while Stop==0:
    try:
        cmd=input()
        exec(cmd)
    except EOFError:
        Stop=1

# ------ Set other params to defaults ------
U0=p21c.UserParams._defaults_;
C0=p21c.CosmoParams._defaults_;
A0=p21c.AstroParams._defaults_;
F0=p21c.FlagOptions._defaults_;

# ---- CosmoParams ----
if SIGMA_8==default:
    SIGMA_8=C0['SIGMA_8'];
if hlittle==default:
    hlittle=C0['hlittle']
if OMm==default:
    OMm=C0['OMm']
if POWER_INDEX==default:
    POWER_INDEX=C0['POWER_INDEX']

# ---- UserParams ----
if BOX_LEN==default:
    BOX_LEN=U0['BOX_LEN']
if DIM==default:
    DIM=U0['DIM']
if HII_DIM==default:
    HII_DIM=U0['HII_DIM']
if USE_FFTW_WISDOM==default:
    USE_FFTW_WISDOM=U0['USE_FFTW_WISDOM']
if HMF==default:
    HMF=U0['HMF']
if USE_RELATIVE_VELOCITIES==default:
    USE_RELATIVE_VELOCITIES=U0['USE_RELATIVE_VELOCITIES']
if POWER_SPECTRUM==default:
    POWER_SPECTRUM=U0['POWER_SPECTRUM']
if N_THREADS==default:
    N_THREADS=U0['N_THREADS']
if PERTURB_ON_HIGH_RES==default:
    PERTURB_ON_HIGH_RES=U0['PERTURB_ON_HIGH_RES']
if NO_RNG==default:
    NO_RNG=U0['NO_RNG']
if USE_INTERPOLATION_TABLES==default:
    USE_INTERPOLATION_TABLES=U0['USE_INTERPOLATION_TABLES']
if FAST_FCOLL_TABLES==default:
    FAST_FCOLL_TABLES=U0['FAST_FCOLL_TABLES']
if USE_2LPT==default:
    USE_2LPT=U0['USE_2LPT']
if MINIMIZE_MEMORY==default:
    MINIMIZE_MEMORY=U0['MINIMIZE_MEMORY']

# ---- AstroParams ----
if HII_EFF_FACTOR==default:
    HII_EFF_FACTOR=A0['HII_EFF_FACTOR']
if F_STAR10==default:
    F_STAR10=A0['F_STAR10']
if F_STAR7_MINI==default:
    F_STAR7_MINI=A0['F_STAR7_MINI']
if ALPHA_STAR==default:
    ALPHA_STAR=A0['ALPHA_STAR']
if ALPHA_STAR_MINI==default:
    ALPHA_STAR_MINI=A0['ALPHA_STAR_MINI']
if F_ESC10==default:
    F_ESC10=A0['F_ESC10']
if F_ESC7_MINI==default:
    F_ESC7_MINI=A0['F_ESC7_MINI']
if ALPHA_ESC==default:
    ALPHA_ESC=A0['ALPHA_ESC']
if M_TURN==default:
    M_TURN=A0['M_TURN']
if R_BUBBLE_MAX==default:
    R_BUBBLE_MAX=A0['R_BUBBLE_MAX']
if ION_Tvir_MIN==default:
    ION_Tvir_MIN=A0['ION_Tvir_MIN']
if L_X==default:
    L_X=A0['L_X']
if L_X_MINI==default:
    L_X_MINI=A0['L_X_MINI']
if NU_X_THRESH==default:
    NU_X_THRESH=A0['NU_X_THRESH']
if X_RAY_SPEC_INDEX==default:
    X_RAY_SPEC_INDEX=A0['X_RAY_SPEC_INDEX']
if X_RAY_Tvir_MIN==default:
    X_RAY_Tvir_MIN=A0['X_RAY_Tvir_MIN']
if F_H2_SHIELD==default:
    F_H2_SHIELD=A0['F_H2_SHIELD']
if t_STAR==default:
    t_STAR=A0['t_STAR']
if N_RSD_STEPS==default:
    N_RSD_STEPS=A0['N_RSD_STEPS']
if A_LW==default:
    A_LW=A0['A_LW']
if BETA_LW==default:
    BETA_LW=A0['BETA_LW']
if A_VCB==default:
    A_VCB=A0['A_VCB']
if BETA_VCB==default:
    BETA_VCB=A0['BETA_VCB']
# Now set these Radio/PBH Params
if fR==default:
    fR=A0['fR']
if aR==default:
    aR=A0['aR']
if fR_mini==default:
    fR_mini=A0['fR_mini']
if aR_mini==default:
    aR_mini=A0['aR_mini']
if mbh==default:
    mbh=A0['mbh']   
if fbh==default:
    fbh=A0['fbh']
if bh_aR==default:
    bh_aR=A0['bh_aR']
if bh_fX==default:
    bh_fX=A0['bh_fX']
if bh_fR==default:
    bh_fR=A0['bh_fR']
if bh_lambda==default:
    bh_lambda=A0['bh_lambda']
if bh_Eta==default:
    bh_Eta=A0['bh_Eta']
if bh_spin==default:
    bh_spin=A0['bh_spin']
if Radio_Zmin==default:
    Radio_Zmin=A0['Radio_Zmin']

# ---- FlagOptions ----
if USE_HALO_FIELD==default:
    USE_HALO_FIELD=F0['USE_HALO_FIELD']
if USE_MINI_HALOS==default:
    USE_MINI_HALOS=F0['USE_MINI_HALOS']
if USE_MASS_DEPENDENT_ZETA==default:
    USE_MASS_DEPENDENT_ZETA=F0['USE_MASS_DEPENDENT_ZETA']
if SUBCELL_RSD==default:
    SUBCELL_RSD=F0['SUBCELL_RSD']
if INHOMO_RECO==default:
    INHOMO_RECO=F0['INHOMO_RECO']
if USE_TS_FLUCT==default:
    USE_TS_FLUCT=F0['USE_TS_FLUCT']
if M_MIN_in_Mass==default:
    M_MIN_in_Mass=F0['M_MIN_in_Mass']
if PHOTON_CONS==default:
    PHOTON_CONS=F0['PHOTON_CONS']
if FIX_VCB_AVG==default:
    FIX_VCB_AVG=F0['FIX_VCB_AVG']
if USE_RADIO_ACG==default:
    USE_RADIO_ACG=F0['USE_RADIO_ACG']
if USE_RADIO_MCG==default:
    USE_RADIO_MCG=F0['USE_RADIO_MCG']
if USE_RADIO_PBH==default:
    USE_RADIO_PBH=F0['USE_RADIO_PBH']
if USE_HAWKING_RADIATION==default:
    USE_HAWKING_RADIATION=F0['USE_HAWKING_RADIATION']


# ------ Setting params ------
CosmoParams=p21c.CosmoParams(
    SIGMA_8=SIGMA_8,
    hlittle=hlittle,
    OMm=OMm,
    POWER_INDEX=POWER_INDEX
    )

UserParams=p21c.UserParams(
    BOX_LEN=BOX_LEN,
    DIM=DIM,
    HII_DIM=HII_DIM,
    USE_FFTW_WISDOM=USE_FFTW_WISDOM,
    HMF=HMF,
    USE_RELATIVE_VELOCITIES=USE_RELATIVE_VELOCITIES,
    POWER_SPECTRUM=POWER_SPECTRUM,
    N_THREADS=N_THREADS,
    PERTURB_ON_HIGH_RES=PERTURB_ON_HIGH_RES,
    NO_RNG=NO_RNG,
    USE_INTERPOLATION_TABLES=USE_INTERPOLATION_TABLES,
    FAST_FCOLL_TABLES=FAST_FCOLL_TABLES,
    USE_2LPT=USE_2LPT,
    MINIMIZE_MEMORY=MINIMIZE_MEMORY
    )

AstroParams=p21c.AstroParams(
    HII_EFF_FACTOR=HII_EFF_FACTOR,
    F_STAR10=F_STAR10,
    F_STAR7_MINI=F_STAR7_MINI,
    ALPHA_STAR=ALPHA_STAR,
    ALPHA_STAR_MINI=ALPHA_STAR_MINI,
    F_ESC10=F_ESC10,
    F_ESC7_MINI=F_ESC7_MINI,
    ALPHA_ESC=ALPHA_ESC,
    M_TURN=M_TURN,
    R_BUBBLE_MAX=R_BUBBLE_MAX,
    ION_Tvir_MIN=ION_Tvir_MIN,
    L_X=L_X,
    L_X_MINI=L_X_MINI,
    NU_X_THRESH=NU_X_THRESH,
    X_RAY_SPEC_INDEX=X_RAY_SPEC_INDEX,
    X_RAY_Tvir_MIN=X_RAY_Tvir_MIN,
    F_H2_SHIELD=F_H2_SHIELD,
    t_STAR=t_STAR,
    N_RSD_STEPS=N_RSD_STEPS,
    A_LW=A_LW,
    BETA_LW=BETA_LW,
    A_VCB=A_VCB,
    BETA_VCB=BETA_VCB,
    fR=fR,
    aR=aR,
    fR_mini=fR_mini,
    aR_mini=aR_mini,
    mbh=mbh,
    fbh=fbh,
    bh_aR=bh_aR,
    bh_fX=bh_fX,
    bh_fR=bh_fR,
    bh_lambda=bh_lambda,
    bh_Eta=bh_Eta,
    bh_spin=bh_spin,
    Radio_Zmin=Radio_Zmin
    )

FlagOptions=p21c.FlagOptions(
    USE_HALO_FIELD=USE_HALO_FIELD,
    USE_MINI_HALOS=USE_MINI_HALOS,
    USE_MASS_DEPENDENT_ZETA=USE_MASS_DEPENDENT_ZETA,
    SUBCELL_RSD=SUBCELL_RSD,
    INHOMO_RECO=INHOMO_RECO,
    USE_TS_FLUCT=USE_TS_FLUCT,
    M_MIN_in_Mass=M_MIN_in_Mass,
    PHOTON_CONS=PHOTON_CONS,
    FIX_VCB_AVG=FIX_VCB_AVG,
    USE_RADIO_ACG=USE_RADIO_ACG,
    USE_RADIO_MCG=USE_RADIO_MCG,
    USE_RADIO_PBH=USE_RADIO_PBH,
    USE_HAWKING_RADIATION=USE_HAWKING_RADIATION
    )

# ---- Run 21cmFAST ----
LC_Quantities = ('brightness_temp','Trad_box')
GLB_Quantities = ('brightness_temp','Trad_box')

Data=p21c.run_lightcone(
    redshift=redshift,
    max_redshift=max_redshift,
    cosmo_params=CosmoParams,
    user_params=UserParams,
    astro_params=AstroParams,
    flag_options=FlagOptions,
    lightcone_quantities=LC_Quantities,
    global_quantities=GLB_Quantities
    )
Data.save(FileName)