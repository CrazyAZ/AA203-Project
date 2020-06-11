import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
from scipy.integrate import solve_ivp
import math
import serial
import struct

ser = serial.Serial('/dev/tty.usbmodem74956901', 115200)
T = np.array([0.001 * t for t in range(1000)])
theta = np.ndarray((1000,))
setpoint = np.pi / 4 * np.random.rand()
print(setpoint)

ser.write(struct.pack("<f", -setpoint))

for t in range(1000):
    encoder_pos = struct.unpack('f', ser.read(4))[0]
    theta[t] = -encoder_pos

def smooth(y, size=50, order=2, deriv=0):

    n = len(y)
    m = size

    y = np.append([0 for _ in range(m)], y)
    y = np.append(y, [y[-1] for _ in range(m)])

    result = np.zeros(n)

    a = np.array([i * 0.001 for i in range(-m , m+1)])
    A = np.array([[a[i]**n for n in range(order+1)] for i in range(len(a))])
    Ai = np.linalg.pinv(A)

    for i in range(n):
        start, end = i , i + 2 * m + 1
        f = Ai[deriv]
        result[i] = np.dot(f, y[start:end])

    if deriv > 1:
        result *= math.factorial(deriv)

    return result
    
# stheta = smooth(theta)
# sdtheta = smooth(theta, deriv=1)
# sddtheta = smooth(theta, deriv=2)

# I = 3.6e-3
# error = setpoint - stheta
# dtheta_squared = np.square(sdtheta)

# K = cvx.Variable(3)
# cost = cvx.norm((K[0] * error - K[1] * sdtheta - K[2] * dtheta_squared) - I * sddtheta, p=1)
# prob = cvx.Problem(cvx.Minimize(cost))
# prob.solve()

# Kp = K.value[0]
# Kd = K.value[1]

# print(Kp)
# print(Kd)
# print(K.value[2])
# contoller = np.array([0.4757 * (setpoint - stheta[t]) - 0.0434 * sdtheta[t] for t in range(len(T))]) / I


# fig, axs = plt.subplots(3)
# axs[0].plot(T, stheta)
# axs[0].plot([T[0], T[-1]], [setpoint, setpoint], "--")
# axs[1].plot(T, sdtheta)
# axs[2].plot(T, sddtheta)
# axs[2].plot(T, contoller)
# plt.show()

I = 3.6e-3
Kp = 1.2
Kd = 0.09
Ks = 0.4

def contoller_dynamics(t, x):
    tau = Kp * Ks * np.tanh((setpoint - x[0]) / Ks) - Kd * x[1]
    return np.append(x[1], tau / I)

sim = solve_ivp(contoller_dynamics, [0, T[-1]], [0, 0], t_eval=T)

plt.plot(T, theta)
plt.plot(T, sim.y[0,:])
plt.plot([T[0], T[-1]], [setpoint, setpoint], "--")
plt.show()