import numpy as np
from scipy.linalg import solve_discrete_are
from acrobot_env import AcrobotEnv
from ilqr_solution import ilqr_solution
import time

x0 = [0.0, 0.0, 0.0, 0.0]

Q = np.eye(4)
Qf = 1000 * np.eye(4)
R = np.eye(1)

env = AcrobotEnv()
goal_state = np.array([np.pi, 0, 0, 0])

swingup_steps = 200
env.dt = 0.017 * 3
u_bar1 = 10 * np.random.rand(swingup_steps, 1)
print("First")
_, u_bar1, _, _ = ilqr_solution(env, Q, R, Qf, goal_state, x0, u_bar1, swingup_steps)

swingup_steps = 600
env.dt = 0.017
u_bar = np.ndarray((swingup_steps, 1))
for t in range(200):
    u_bar[3*t] = u_bar1[t]
    u_bar[3*t+1] = u_bar1[t]
    u_bar[3*t+2] = u_bar1[t]

print("Combined")
x_bar, u_bar, l, L = ilqr_solution(env, Q, R, Qf, goal_state, x0, u_bar, swingup_steps)


f, A, B = env.linearized_dynamics(goal_state, 0)

A = np.eye(4) + env.dt * A
B = env.dt * B

P = solve_discrete_are(A, B, Q, R)
L_inf = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

state = env.reset(state=x0)
env.render()
time.sleep(1)

start = time.monotonic()
for t in range(1000):
    if t < swingup_steps:
        u = u_bar[t] + l[t] + L[t] @ (state - x_bar[t])
    else:
        u = - L_inf @ (state - goal_state)
    state = env.step(u)
    env.render()
end = time.monotonic()
print(end-start)
env.close()


