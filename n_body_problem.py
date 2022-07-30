"""
N Body Problem - Numerical Solution and Representation

This Function solves the N Body Problem using Range-Kutta-Fehlberg Models for Numerical Integration

Author: Lucas Casaril
        eng@lucascasaril.me
"""

import numpy as np
from numpy.linalg import norm
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from motion import dydt
from rkf_method import rkf_integration

G = 6.67259e-20 #Universal Gravitational Constant (km^3/kg/s^2)
t0 = 0

#Input Data: 
m1 = 1e26 # First Body's Mass - kg - Blue
m2 = 1e24 # Second Body's Mass - kg - Black
m3 = 1e5 # Third Body's Mass - kg - Red
m4 = 1e5 # Fourth Body's Mass - kg - Gray
m5 = 1e5 # Fifth Body's Mass - kg - Green

m = [m1, m2, m3, m4, m5]
N = 5 # Number of Bodies

tf = 1000 # Time of Simulation - seconds
h = 1 # Steps within time interval - Number of iterations is going to be = tf*(1/h)
tol = 1e-6 # Tolerance for the Runge-Kutta-Fehlberg Method
lenght = int(tf*(1/h))

#Initial Condition 
R1_0 = [0, 0, 0] # Initial Position of the First Body (km) 
R2_0 = [7000, 0, 0] # Initial Position of the Second Body (km)
R3_0 = [7500, 0, 0] # Initial Position of the Third Body (km)
R4_0 = [2000, 0, 0] # Initial Position of the Fourth Body (km)
R5_0 = [4000, 0, 0] # Initial Position of the Fifth Body (km)

V1_0 = [0, 0, 10] # Initial Velocity of the First Body (km/s)
V2_0 = [0, 30, 10] # Initial Velocity of the Second Body (km/s)
V3_0 = [0, 20, 10] # Initial Velocity of the Third Body (km/s)
V4_0 = [0, 60, 10] # Initial Velocity of the Fourth Body (km/s)
V5_0 = [0, 40, 10] # Initial Velocity of the Fifth Body (km/s)

# Initial Condition - Vector (6Nx1)
y0 = np.concatenate((R1_0, R2_0, R3_0, R4_0, R5_0, V1_0, V2_0, V3_0, V4_0, V5_0), axis=None)

#### ATTENTION - Only use on method at a time !!

# Calling the Numerical Integration Solver (Runge-Kutta-Fehlberg)
y_result = rkf_integration(dydt, t0, tf, y0, m, tol, h)
y_result = np.array(y_result)

# Using the built-in solvers in Python:
#t = np.linspace(t0, tf, lenght)
#y_result = odeint(dydt, y0, t, args=(m,)) # Need to change the calling "def dydt(y, t):


# At this point, the plot has to be adjusted for the number of bodies
X1 = y_result[:, 0]
Y1 = y_result[:, 1]
Z1 = y_result[:, 2]

X2 = y_result[:, 3]
Y2 = y_result[:, 4]
Z2 = y_result[:, 5]

X3 = y_result[:, 6]
Y3 = y_result[:, 7]
Z3 = y_result[:, 8]

X4 = y_result[:, 9]
Y4 = y_result[:, 10]
Z4 = y_result[:, 11]

X5 = y_result[:, 12]
Y5 = y_result[:, 13]
Z5 = y_result[:, 14]

'''
# Center the Mass at each time step used:
XG = np.zeros (len(X1)); YG = np.zeros (len(X1)); ZG = np.zeros (len(X1))

for i in range(len(X1)):

    XG[i] = ((m1*X1[i] + m2*X2[i] + m3*X3[i] + m4*X4[i] + m5*X5[i])/(m1 + m2 + m3 + m4 + m5))
    YG[i] = ((m1*Y1[i] + m2*Y2[i] + m3*Y3[i] + m4*Y4[i] + m5*Y5[i])/(m1 + m2 + m3 + m4 + m5))
    ZG[i] = ((m1*Z1[i] + m2*Z2[i] + m3*Z3[i] + m4*Z4[i] + m5*Z5[i])/(m1 + m2 + m3 + m4 + m5))
'''
# Ploting the Answer

dataSet1 = np.array([X1, Y1, Z1])  # Combining the position coordinates
dataSet2 = np.array([X2, Y2, Z2])  # Combining the position coordinates
dataSet3 = np.array([X3, Y3, Z3])  # Combining the position coordinates
dataSet4 = np.array([X4, Y4, Z4])  # Combining the position coordinates
dataSet5 = np.array([X5, Y5, Z5])  # Combining the position coordinates
#dataSetG = np.array([XG, YG, ZG])  # Combining the position coordinates
numDataPoints = len(X1)

def animate_func(num): # Aqui dentro tem que ter as várias chamadas das orbitas

    ax.clear()  # Clears the figure to update the line, point,   
                # title, and axes

    
    # Updating Trajectory Line (num+1 due to Python indexing)
    ax.plot3D(dataSet1[0, :num+1], dataSet1[1, :num+1], dataSet1[2, :num+1], c='blue')
    ax.plot3D(dataSet2[0, :num+1], dataSet2[1, :num+1], dataSet2[2, :num+1], c='black')
    ax.plot3D(dataSet3[0, :num+1], dataSet3[1, :num+1], dataSet3[2, :num+1], c='red')
    ax.plot3D(dataSet4[0, :num+1], dataSet4[1, :num+1], dataSet4[2, :num+1], c='gray')
    ax.plot3D(dataSet5[0, :num+1], dataSet5[1, :num+1], dataSet5[2, :num+1], c='green')
    #ax.plot3D(dataSetG[0, :num+1], dataSetG[1, :num+1], dataSetG[2, :num+1], c='red')
    
    #ax.plot_surface(0, 0, 0, color='blue', alpha=0.7)

    # Updating Point Location 
    ax.scatter(dataSet1[0, num-1], dataSet1[1, num-1], dataSet1[2, num-1], c='blue', marker='o')
    ax.scatter(dataSet2[0, num-1], dataSet2[1, num-1], dataSet2[2, num-1], c='black', marker='o')
    ax.scatter(dataSet3[0, num-1], dataSet3[1, num-1], dataSet3[2, num-1], c='red', marker='o')
    ax.scatter(dataSet4[0, num-1], dataSet4[1, num-1], dataSet4[2, num-1], c='gray', marker='o')
    ax.scatter(dataSet5[0, num-1], dataSet5[1, num-1], dataSet5[2, num-1], c='green', marker='o')

    # Adding Constant Origin
    #ax.plot3D(0,0,0, c='blue', marker='o')

    #Setting Axes Limits
    ax.set_xlim3d([-1e4, 1e4])    
    ax.set_ylim3d([-1e4, 1e4])
    ax.set_zlim3d([0, 9e3])

    plt.title('5-Body Problem - Simulation')
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')

    ax.view_init(45, 45)

# Plotting the Animation
#numDataPoints = numDataPoints/1
fig = plt.figure()
ax = plt.axes(projection='3d')
line_ani = animation.FuncAnimation(fig, animate_func, interval=1, frames=numDataPoints)   # Inteval can be the speed of the animation
plt.show()

'''
# Saving the Animation
#f = r"/home/casaril/Desktop/animate_func1.gif"
#writergif = animation.PillowWriter(fps=numDataPoints/10)
#line_ani.save(f, writer=writergif)
'''