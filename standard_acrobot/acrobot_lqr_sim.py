import numpy as np
from scipy.linalg import solve_continuous_are
from acrobot_env import AcrobotEnv
import time

x0 = [np.pi, 0.0, 0.0, 0.0]

Q = np.eye(4)
R = np.eye(1)

env = AcrobotEnv()
goal_state = np.array([np.pi, 0, 0, 0])
f, A, B = env.linearized_dynamics(goal_state, 0)

P = solve_continuous_are(A, B, Q, R)
L = np.linalg.inv(R) @ B.T @ P

print(L)

state = env.reset(state=x0)
env.render()
time.sleep(1)

start = time.monotonic()
for _ in range(1000):
    u = - L @ (state - goal_state)
    state = env.step(u)
    env.render()
end = time.monotonic()
print(end-start)
env.close()

