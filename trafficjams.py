import numpy as np
import matplotlib.pyplot as plt

#parameters

N = 100         #total number of vehicles
L = 200         #Length
a = 1           #sensitivity
s = 2           #safety distance
iterations = 1000
Startspeed = 1


tanhs=np.tanh(s) #calculating it once so you dont have to calculate it every time during the adjustment function.
def AdjustmentFunction(delta_P):
    return (np.tanh(delta_P-s)+tanhs)

def initCars():
    #creating the general positioning and the matrix to store it
    position = np.zeros(shape=(N,iterations)) #vertical different cars positions, horizontal different iterations.
    for i in range(N):
        position[i,0]=(L/N)*i #Each one uses the equalibrium solution
    #adding the deviation


    #velocity storage
    velocity = np.zeros(shape=(N,iterations)) #vertical different cars velocities, horizontal different iterations.
    for i in range(N):
        velocity[i,0]=Startspeed

    #headway storage
    headway = np.zeros(shape=(N,iterations)) #vertical different cars headways, horizontal different iterations.
    for i in range(N):
        headway[i,0]=position[i+1,0]-position[i,0]