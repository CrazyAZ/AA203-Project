import numpy as np

def backward_riccati_recursion(P, p, A, B, Q, q, R, r):
    Qx = q + A.T @ p
    Qu = r + B.T @ p
    Qxx = Q + A.T @ P @ A
    Quu = R + B.T @ P @ B
    Qux = B.T @ P @ A
    
    l = -np.linalg.inv(Quu) @ Qu
    L = -np.linalg.inv(Quu) @ Qux
    
    p = Qx - L.T @ Quu @ l
    P = Qxx - L.T @ Quu @ L

    return (l, L, P, p)

def ilqr_solution(env, Q, R, Qf, goal, x0, u_bar, num_steps):
    n = 4
    m = 1

    l = np.zeros((num_steps, m))
    L = np.zeros((num_steps, m, n))

    x_bar = np.zeros((num_steps+1, n))
    x_bar[0] = x0
    u_bar_prev = 1000 * np.ones((num_steps, m))

    epsilon = 0.1

    for t in range(num_steps):
        x_bar[t+1] = env.f(x_bar[t], u_bar[t])
    x_bar_prev = np.copy(x_bar)

    while np.linalg.norm(u_bar - u_bar_prev) > epsilon:
        print(np.linalg.norm(u_bar - u_bar_prev))
        qf = Qf @ (x_bar[-1] - goal)
        P = Qf
        p = qf

        for t in range(num_steps-1, -1, -1):
            _, A, B = env.linearized_dynamics(x_bar[t], u_bar[t,0])
            A = np.eye(4) + env.dt * A
            B = env.dt * B

            q = Q @ (x_bar[t] - goal)
            r = R @ u_bar[t]

            lt, Lt, P, p = backward_riccati_recursion(P, p, A, B, Q, q, R, r)

            l[t] = lt
            L[t] = Lt
        
        u_bar_prev = np.copy(u_bar)

        for t in range(num_steps):
            u_bar[t] = u_bar[t] + l[t] + L[t] @ (x_bar[t] - x_bar_prev[t])
            x_bar[t+1] = env.f(x_bar[t], u_bar[t])
        x_bar_prev = np.copy(x_bar)
        

    return (x_bar, u_bar, l, L)

