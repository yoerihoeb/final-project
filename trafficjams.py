import numpy as np
import matplotlib.pyplot as plt

#parameters
N = 100             #total number of vehicles
L = 200             #Length
a = 1               #sensitivity
s = 2               #safety distance
iterations = 300
Startspeed = 1

graphdetail = 4     #the amount of detail, higher is less detail

tanhs=np.tanh(s) #calculating it once so you dont have to calculate it every time during the adjustment function.
def V(delta_P):
    return (np.tanh(delta_P-s)+tanhs)

#Steady state flow constants
b = float(L)/float(N)
c = V(b)
f = 1 - np.tanh(b)**2 

#storage arrays
position = np.zeros(shape=(N,iterations))   #vertical different cars positions, horizontal different iterations.
headway = np.zeros(shape=(N,iterations))    #vertical different cars headways, horizontal different iterations.
velocity = np.zeros(shape=(N,iterations))   #vertical different cars velocities, horizontal different iterations.
acceleration = np.zeros(shape=(N,iterations)) #vertical different cars velocities, horizontal different iterations.



def aCalc(car,time): #calculates the acceleration for a given car and time
    return a*(V(headway[car,time])-velocity[car,time])

def initCars():
    #position initialization
    for i in range(N): 
        position[i,0]=(L/N)*i #Each one uses the equalibrium solution
    
    #adding the deviation
    position[0,0] += 0.001

    #headway initialization
    for i in range(N):
        headway[i,0]=(position[(i+1)%N,0]-position[(i)%N,0])%L

    #velocity initialization
    for i in range(N):
        velocity[i,0]=Startspeed
    
    #acceleration initialization
    for i in range(N):
        acceleration[i,0]=aCalc(i,0)


def main():
    initCars()
    for i in range(1,iterations):
        for j in range(N): #loop to change the position and velocity for next cycle
            position[j,i]=(position[j,i-1]+velocity[j,i-1])%L
            velocity[j,i]=velocity[j,i-1]+acceleration[j,i-1]
        for j in range(N): #seperate loop for headway because position is needed for headway
            headway[j,i]=(position[(j+1)%N,i]-position[j,i])%L
        for j in range(N): #seperate loop for acceleration because headway and velocity are needed for acceleration
            acceleration[j,i]=aCalc(j,i)



main()
#displaying as a graph
fig, ax = plt.subplots(2,2)
fig.subplots_adjust(hspace=0.53, wspace=0.3)
fig.set_size_inches(12, 6)

t = np.arange(0, iterations, 1)

#position time graph
for i in range(N//graphdetail):
    ax[0,0].plot(t, position[i*graphdetail,t],c="k",ls="", marker="o",markersize=0.4)
ax[0,0].set_title("Car position")
ax[0,0].set_ylabel("Position")
ax[0,0].set_xlabel("Time")
ax[0,0].set_ylim(0,L)
ax[0,0].set_xlim(0,iterations)

#headway time graph
for i in range(N//graphdetail):
    ax[0,1].plot(t, headway[i*graphdetail,t],c="b",ls="-", linewidth=0.5)

avg_headway = np.mean(headway, axis=0)  # average over all cars, for each timestep
ax[0,1].plot(t, avg_headway, c="red", ls="--", linewidth=1.2, label="Average Headway")
ax[0,1].legend()

ax[0,1].set_title("Headway per car")
ax[0,1].set_ylabel("Headway")
ax[0,1].set_xlabel("Time")
ax[0,1].set_xlim(0,iterations)

#velocity time graph
for i in range(N//graphdetail):
    ax[1,0].plot(t, velocity[i*graphdetail,t], c="g",ls="-",linewidth=0.5)

avg_velocity = np.mean(velocity, axis=0)
ax[1,0].plot(t, avg_velocity, c="red", ls="--", linewidth=1.2, label="Average Velocity")
ax[1,0].legend()

ax[1,0].set_title("Velocity per car")
ax[1,0].set_ylabel("Velocity")
ax[1,0].set_xlabel("Time")
ax[1,0].set_ylim(0, 2)
ax[1,0].set_xlim(0, iterations)


carn = np.arange(0, N,1)        #carnumber

ax[1,1].plot(carn,velocity[carn,0], c="orange",ls="-",linewidth=1.3, label="t=0")
ax[1,1].plot(carn,velocity[carn,100], c="b",ls="-",linewidth=1.3, label="t=100")

ax[1,1].set_title("Velocity of car at t")
ax[1,1].set_ylabel("Velocity")
ax[1,1].set_xlabel("Car number")
ax[1,1].set_ylim(0, 2)
ax[1,1].set_xlim(0, N)
ax[1,1].legend()

plt.show()