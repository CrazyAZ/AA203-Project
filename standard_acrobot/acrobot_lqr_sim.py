import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are
from acrobot_env import AcrobotEnv
import time

x0 = [np.pi, 0.0, 0.0, 0.0]

Q = np.eye(4)
R = np.eye(1)

env = AcrobotEnv()
goal_state = np.array([np.pi, 0, 0, 0])
f, A, B = env.linearized_dynamics(goal_state, 0)

Ad = np.eye(4) + env.dt * A
Bd = env.dt * B

P = solve_continuous_are(A, B, Q, R)
L = np.linalg.inv(R) @ B.T @ P

Pd = solve_discrete_are(Ad, Bd, Q, R)
Ld = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

print(L)
print(Ld)

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

