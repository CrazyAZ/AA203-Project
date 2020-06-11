import numpy as np
from numpy.linalg import inv
from scipy.linalg import solve_continuous_are
from real_acrobot_env import AcrobotEnv
from extended_kalman_filter import update
import serial
import struct
import time

ser = serial.Serial('/dev/tty.usbmodem74956901', 115200)

x0 = [np.pi, 0.0, 0.0, 0.0]

Q = np.array([[10, 0, 0, 0],
              [0, 10, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

R = 50*np.eye(1)

env = AcrobotEnv()
goal_state = np.array([np.pi, 0.0, 0.0, 0.0])
f, A, B = env.linearized_dynamics(goal_state, 0)

P = solve_continuous_are(A, B, Q, R)
L = np.linalg.inv(R) @ B.T @ P
print(L)

Sigma = 0.01 * np.eye(4)
state_estimate = x0
C = env.C
Sw = env.state_noise_covariance
Sv = env.measurement_noise_covariance

env.render(state_estimate)

input("Press Enter to continue...")
prev_time = time.monotonic()
while True:
    
    curr_time = time.monotonic()
    dt = curr_time - prev_time
    if dt < 0.01:
        continue
    prev_time = curr_time
    

    u = - L @ (state_estimate - goal_state)
    if abs(u[0] - state_estimate[1]) < env.deadband:
        u[0] = state_estimate[1]
    u[0] = min(max(u[0], -np.pi / 2), np.pi / 2)

    ser.write(struct.pack("<f", u))

    encoder_pos = struct.unpack('f', ser.read(4))[0]
    state_estimate, Sigma = update(env, state_estimate, u, encoder_pos, Sigma, dt)
    env.render(state_estimate)

    print(u)
    print(state_estimate)
    print(encoder_pos)
    print()
    
env.close()