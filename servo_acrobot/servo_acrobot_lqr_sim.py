import numpy as np
from numpy.linalg import inv
from scipy.linalg import solve_discrete_are, solve_continuous_are
from servo_acrobot_env import AcrobotEnv
from extended_kalman_filter import update
import time

x0 = [np.pi, 0.0, 0.0, 0.0]

Q = np.eye(4)
R = np.eye(1)

env = AcrobotEnv()
goal_state = np.array([np.pi, 0, 0, 0])
f, A, B = env.linearized_dynamics(goal_state, 0)

P = solve_continuous_are(A, B, Q, R)
L = np.linalg.inv(R) @ B.T @ P

env.reset(state=x0)
env.render()
time.sleep(1)

Sigma = np.zeros((4,4))
state_estimate = x0
C = env.C
Sw = env.state_noise_covariance
Sv = env.measurement_noise_covariance


N = 1000
start = time.monotonic()
for t in range(N):
    u = - L @ (state_estimate - goal_state)
    print(u)
    state_measurement = env.step(u)

    state_estimate, Sigma = update(env, state_estimate, u, state_measurement, Sigma)

    print(state_estimate)
    print(env.state)
    print()
    
    env.render()
end = time.monotonic()
print(end-start)
env.close()

