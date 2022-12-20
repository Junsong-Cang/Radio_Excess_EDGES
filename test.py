import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline

x=np.arange(10,0,-1)
y=np.sin(x)
x3=x[-1:0:-1]
#x2=np.arange(0,10,2)
#y2=spline(x,y)(x2)
plt.plot(x,y)
#plt.plot(x2,y2)
print(x3[0])
print(x[0])
