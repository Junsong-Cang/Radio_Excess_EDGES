import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('agg')
import os
import emcee
import corner
import shutil

ChainFile = "../Chains/Chains.h5"
ConvergeFile = "Status.h5"
labels = ["$\mathrm{log}_{10}f_{\mathrm{R}}$", "${\mathrm{log}}_{10}L_{\mathrm{X}}$"]

Check_Interv = 10
Converge_Thresh = 100

# ---- Initialise ----
ChainFile_Swap = "Chains_Swap.h5"
shutil.copyfile(ChainFile,ChainFile_Swap)

reader = emcee.backends.HDFBackend(ChainFile_Swap)
tau = reader.get_autocorr_time(tol=0)
samples = reader.get_chain(flat=True)

# Get Triangular plot
fig = corner.corner(
    samples, 
    labels=labels,
    color='b',
    show_titles=True,
    levels=(0.95,), # for 2D
    quantiles=(0.0,0.95), # for 1D
    bins=40,
    smooth=1,
    smooth1d=1,
    )
fig.savefig("Triangular_Stat.png")

os.system('open Triangular_Stat.png')

# Clean up
os.remove(ChainFile_Swap)
print(samples.shape)
