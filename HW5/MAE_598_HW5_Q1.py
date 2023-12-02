import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

x1=np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
x2=np.flip(x1)
print (x1)
print (x2)
p=np.array([28.1,34.4,36.7,36.9,36.8,36.7,36.5,35.4,32.9,27.7,17.5])
a=np.array([[8.07131,1730.63,233.426],[7.43155,1554.679,240.337]])
T=20
p_water=10 ** (a[0,0] - a[0,1]/ (T+a[0,2]))
p_dio= 10 ** (a[1,0]-a[1,1]/(T+a[1,2]))

def obj_function(para,x1,x2,p):
    A12=para[0]
    A21=para[1]
    p_pred=x1 * np.exp(A12*(A21*x2/(A12*x1+A21*x2))**2)*p_water + x2*np.exp(A21*(A12*x1/(A12*x1+A21*x2))**2)*p_dio

    return np.sum((p_pred-p)**2)

params=[0.5,0.5]

result=minimize(obj_function,params,args=(x1,x2,p),method='BFGS')

optimize_para = result.x

print (result)

print ('The optimized value for A12 and A21 are:')
print (optimize_para)
print ('The loss of the function is:')
print(result.fun)

A12=optimize_para[0]
A21=optimize_para[1]
p_pred=x1 * np.exp(A12*(A21*x2/(A12*x1+A21*x2))**2)*p_water + x2*np.exp(A21*(A12*x1/(A12*x1+A21*x2))**2)*p_dio

plt.plot(x1,p_pred)
plt.plot(x1,p)
plt.show()