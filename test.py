import numpy as np
data = np.genfromtxt("Data_EDGES.txt")
print(data[:,0],sep='\n')

'''
freq = data[:, 0]
tsky = data[:, 1]
wght = data[:, 2]
freq = freq[wght>1]
tsky = tsky[wght>1]
'''