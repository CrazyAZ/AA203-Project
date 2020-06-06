import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.special import expit #sigmoid function
from acrobot_env import AcrobotEnv
import time

x0 = [3.1, 0.0, 0.0, 0.0]

Q = np.eye(4)
R = np.eye(1)

env = AcrobotEnv()
goal_state = np.array([np.pi, 0, 0, 0])
goal_energy = env.energy(goal_state, 0)
f, A, B = env.linearized_dynamics(goal_state, 0)
print(A)

P = solve_continuous_are(A, B, Q, R)
L = np.linalg.inv(R) @ B.T @ P

k1 = 10
k2 = 4
k3 = 25

state = env.reset(state=x0)
env.render()
time.sleep(1)
u = 0

start = time.monotonic()
for _ in range(10000):
    E = env.energy(state, u)
    if abs(env.state[0] - np.pi) > 2: 
        print('Energy')
        u_bar = 2 * expit((E - goal_energy) * env.state[2]) - 1
        u = - k1 * env.state[1] - k2 * env.state[3] + k3 * u_bar
    else:
        u = - L @ (state - goal_state)
    state = env.step(u)
    env.render()
end = time.monotonic()
print(end-start)
env.close()

