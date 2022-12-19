import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('agg')
import os
import emcee
import corner
import shutil

ChainFile = "Chains.h5"
ConvergeFile = "Status.h5"
labels = ["$a$", "$\omega$"]

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

# Get Convergence Stats
f = h5py.File(ConvergeFile_Swap, 'r')
index = np.array(f['index'])
autocorr = np.array(f['autocorr'])
f.close()

n = Check_Interv * np.arange(1, index + 1)
y = autocorr[:index]

fig,ax=plt.subplots()
plt.plot(n, n / Converge_Thresh, "--k")
plt.plot(n, y)
plt.xlim(0, n.max())
plt.ylim(0, y.max() + 0.1 * (y.max() - y.min()))
plt.xlabel("number of steps")
plt.ylabel(r"mean $\hat{\tau}$");
fig.savefig('Convergence.png',bbox_inches='tight',dpi=100)
plt.close()

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
os.system('open Convergence.png')


# Clean up
os.remove(ChainFile_Swap)
os.remove(ConvergeFile_Swap)
print(samples.shape)
print(np.sum(index))
