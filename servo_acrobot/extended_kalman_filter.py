import numpy as np
from numpy.linalg import inv
from scipy.integrate import solve_ivp


def update(env, x_prev, u, z, P_prev):

    X = np.append(x_prev, P_prev)

    def dXdt(t, X):
        x_hat = X[:4]
        P = np.reshape(X[4:], (4,4))
        dx_hat, A, _ = env.linearized_dynamics(x_hat, u[0])
        dP = A @ P + P @ A.T + env.state_noise_covariance
        return np.append(dx_hat, dP)

    solve = solve_ivp(dXdt, [0, env.dt], X, t_eval=[env.dt])
    x_pred = solve.y[:4, 0]
    P_pred = np.reshape(solve.y[4:, 0], (4,4))

    C = env.C
    K = P_pred @ C.T @ inv(C @ P_pred @ C.T + env.measurement_noise_covariance)
    x = x_pred + K @ (z - C @ x_pred)
    P = P_pred - K @ C @ P_pred

    return (x, P)


