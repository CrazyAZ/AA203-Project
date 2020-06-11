import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

I = 3.6e-3
K_p = 2000. # kg/m
K_d = K_p * 0.45 / np.pi
K_saturation = 0.75

setpoint = np.pi/2

def torque(theta, dtheta):
    return K_p * ((setpoint - theta)) - K_d * dtheta

def dynamics(t, x):
    theta = x[0]
    dtheta = x[1]
    tau = torque(theta, dtheta)
    return np.array([dtheta, tau / I])

def rise(t, x):
    return x[0] - setpoint

x0 = np.array([0, 0])

simulate = solve_ivp(dynamics, [0, 1], x0, max_step=0.01)

plt.plot(simulate.t, simulate.y[0,:])
plt.plot([simulate.t[0], simulate.t[-1]], [setpoint, setpoint], "--")
plt.show()