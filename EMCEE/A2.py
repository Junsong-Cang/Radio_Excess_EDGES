import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib
import getdist.plots
import os
import emcee
import corner
import shutil

ChainFile = "../data/EMCEE_V1/Pop_II_Test_EMCEE_Chains.h5"
ConvergeFile = "../data/EMCEE_V1/Pop_II_Test_EMCEE_Status.h5"
labels = ["$\mathrm{log}_{10}f_{\mathrm{R}}$", "${\mathrm{log}}_{10}L_{\mathrm{X}}$"]

Check_Interv = 10
Converge_Thresh = 100

# ---- Initialise ----
ChainFile_Swap = "Chains_Swap.h5"
ConvergeFile_Swap = "Status_Swap.h5"
shutil.copyfile(ChainFile,ChainFile_Swap)
shutil.copyfile(ConvergeFile,ConvergeFile_Swap)

reader = emcee.backends.HDFBackend(ChainFile_Swap)
tau = reader.get_autocorr_time(tol=0)
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
# Clean up
os.remove(ChainFile_Swap)
os.remove(ConvergeFile_Swap)

posterior = samples
g = getdist.plots.getSubplotPlotter()
g.triangle_plot(posterior, filled=True)
g.export('posterior.pdf')