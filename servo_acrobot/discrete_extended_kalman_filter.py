import numpy as np
from numpy.linalg import inv


def update(env, x_prev, u, z, P_prev):


    dx, A, _ = env.linearized_dynamics(x_prev, u[0])
    A = np.eye(4) + env.dt * A
    x_pred = x_prev + env.dt * dx
    P_pred = A @ P_prev @ A.T + env.state_noise_covariance

    C = env.C
    K = P_pred @ C.T @ inv(C @ P_pred @ C.T + env.measurement_noise_covariance)
    x = x_pred + K @ (z - C @ x_pred)
    P = P_pred - K @ C @ P_pred

    return x, P


