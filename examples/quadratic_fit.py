import numpy as np
import matplotlib.pyplot as plt

#generate "synthetic" dataset
m1_true=-2.0; m2_true=0.25; m3_true=1.0
x = np.linspace(-5,5,100)
d = m1_true + m2_true*x + m3_true*x**2

#add random "noise" to the data
noise = np.random.uniform(0.0,5.0,len(d))
d += noise

#plot data that we would like to fit
plt.scatter(x,d,c='C0')
plt.xlim([-6,6])
plt.xlabel('x')
plt.ylabel('d')
plt.show()

#design the G matrix
N = len(d)            #the number of observations
M = 3                 #the number of model parameters
G = np.zeros((N,M))   #initialize empty matrix
G[:,0] = 1.0          #constant term
G[:,1] = x            #linear term
G[:,2] = x**2         #quadratic term

#solve the linear system system
sol = np.linalg.lstsq(G,d) #numpy least squares solver
m_lstsq = sol[0]      #array of model parameters

#'predict' the data from the least squares model
d_pre = np.dot(G,m_lstsq) #d_pre = G*m_lstsq

#plot the results
plt.scatter(x,d,c='C0')
plt.plot(x,d_pre,c='k',linewidth=2)
plt.xlim([-6,6])
plt.xlabel('x')
plt.ylabel('d')
plt.show()

