"""
N Body Problem - Equations of Motions

This Function creates the equations of motions for the N Body Problem, using Python's Vectorization.

Author: Lucas Casaril
        eng@lucascasaril.me
"""

import numpy as np
from numpy.linalg import norm

#def dydt(t , y, m):
def dydt(y, t, m): #For the use of the odeint integrator

     G = 6.67259e-20 #Universal Gravitational Constant (km^3/kg/s^2)

     mass = m

     # We need to create a matrix that in each row is the x,y,z position of that body
     position_matrix = np.array([y[0], y[1], y[2]])
     cont = 3
     for i in range(len(mass)-1):
          position = np.array([y[cont], y[cont+1], y[cont+2]])
          position_matrix = np.vstack((position_matrix, position))
          cont = cont + 3

     # We create array to concatenate all x positions, all y positions and all z positions of the bodies
     x_position = position_matrix[:,0:1]
     y_position = position_matrix[:,1:2]
     z_position = position_matrix[:,2:3]

     x_t = np.transpose(x_position)
     y_t = np.transpose(y_position)
     z_t = np.transpose(z_position)

     # Usind python's vectorization to create a matrix
     vector_x = x_t - x_position 
     vector_y = y_t - y_position 
     vector_z = z_t - z_position

     # We use a differential distance dr to help resolt of the inverse when the numbers are too small and to big
     dr = 0.0001
     r3_inv = (vector_x**2 + vector_y**2 + vector_z**2 + dr**2)
     r3_inv[r3_inv>0] = r3_inv[r3_inv>0]**(-1.5)

     ax = G * (vector_x * mass) * r3_inv
     ax = [sum(e) for e in ax]

     ay = G * (vector_y * mass) * r3_inv
     ay = [sum(e) for e in ay]

     az = G * (vector_z * mass) * r3_inv
     az = [sum(e) for e in az]

     accel = []
     for i in range(len(ax)):
         accel.append(ax[i])
         accel.append(ay[i])
         accel.append(az[i])
     accel = np.array(accel)

     # Eu preciso fazer com que o dydt tenha primeiro as velocidades:
     velocities = np.array_split(y, 2)
     velocities = velocities[1]

     dydt_new = np.concatenate((velocities, accel), axis = None)

     # Returning the vector with Velocity and Acceleration of the Bodies 
     return dydt_new

