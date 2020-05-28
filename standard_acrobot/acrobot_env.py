"""Acrobot model modified by Aaron Schultz for optimal control"""
import numpy as np
from scipy.integrate import solve_ivp
from numpy import sin, cos, pi


# Based on work by:
# __copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
# __credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
#                "William Dabney", "Jonathan P. How"]
# __license__ = "BSD 3-Clause"
# __author__ = "Christoph Dann <cdann@cdann.de>"

# SOURCE:
# https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py

class AcrobotEnv():

    """
    Acrobot is a 2-link pendulum with only the second joint actuated.
    Initially, both links point downwards. The goal is to swing the
    end-effector at a height at least the length of one link above the base.
    Both links can swing freely and can pass by each other, i.e., they don't
    collide when they have the same angle.
    **STATE:**
    The state consists of the sin() and cos() of the two rotational joint
    angles and the joint angular velocities :
    [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2].
    For the first link, an angle of 0 corresponds to the link pointing downwards.
    The angle of the second link is relative to the angle of the first link.
    An angle of 0 corresponds to having the same angle between the two links.
    A state of [1, 0, 1, 0, ..., ...] means that both links point downwards.
    **ACTIONS:**
    The action is either applying +1, 0 or -1 torque on the joint between
    the two pendulum links.
    **REFERENCE:**
    .. seealso::
        R. Sutton: Generalization in Reinforcement Learning:
        Successful Examples Using Sparse Coarse Coding (NIPS 1996)
    .. seealso::
        R. Sutton and A. G. Barto:
        Reinforcement learning: An introduction.
        Cambridge: MIT press, 1998.
    .. warning::
        This version of the domain uses the Runge-Kutta method for integrating
        the system dynamics and is more realistic, but also considerably harder
        than the original version which employs Euler integration,
        see the AcrobotLegacy class.
    """

    dt = 0.017

    LINK_LENGTH_1 = 1.  # [m]
    LINK_LENGTH_2 = 1.  # [m]
    LINK_MASS_1 = 1.  #: [kg] mass of link 1
    LINK_MASS_2 = 1.  #: [kg] mass of link 2
    LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
    LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
    LINK_MOI = 1.  #: moments of inertia for both links

    g = 9.8

    MAX_VEL_1 = 4 * pi
    MAX_VEL_2 = 9 * pi

    torque_noise_max = 0.

    coeff_friction = 0.1

    def __init__(self):
        self.viewer = None
        self.state = None

    def reset(self, state=None):
        if state == None:
            self.state = np.array([0.0, 0.0, 0.0, 0.0])
        else:
            self.state = state
        return self.state

    def dynamics(self, x, u):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        g = self.g
        theta1 = x[0]
        theta2 = x[1]
        dtheta1 = x[2]
        dtheta2 = x[3]
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * cos(theta2)) + I2
        d1 = m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * cos(theta2)) + I1 + I2
        phi2 = m2 * lc2 * g * sin(theta1 + theta2)
        phi1 = - m2 * l1 * lc2 * (dtheta2 ** 2 + 2 * dtheta2 * dtheta1) * sin(theta2) + (m1 * lc1 + m2 * l1) * g * sin(theta1) + phi2
        ddtheta2 = (u + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * sin(theta2) - phi2) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

        # ddtheta1 -= self.coeff_friction * dtheta1
        # ddtheta2 -= self.coeff_friction * dtheta2

        return (dtheta1, dtheta2, ddtheta1, ddtheta2)
    
    def linearized_dynamics(self, x_bar, u_bar):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        g = self.g
        theta1 = x_bar[0]
        theta2 = x_bar[1]
        dtheta1 = x_bar[2]
        dtheta2 = x_bar[3]
        tau = u_bar
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * cos(theta2)) + I2
        d1 = m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * cos(theta2)) + I1 + I2
        phi2 = m2 * lc2 * g * sin(theta1 + theta2)
        phi1 = - m2 * l1 * lc2 * (dtheta2 ** 2 + 2 * dtheta2 * dtheta1) * sin(theta2) + (m1 * lc1 + m2 * l1) * g * sin(theta1) + phi2
        ddtheta2_num = (tau + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * sin(theta2) - phi2)
        ddtheta2_den = m2 * lc2 ** 2 + I2 - d2 ** 2 / d1
        ddtheta2 = ddtheta2_num / ddtheta2_den
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

        dtau_dtheta1 = dtau_dtheta2 = dtau_ddtheta1 = dtau_ddtheta2 = 0
        dtau_du = 1

        dd2_dtheta2 = - m2 * l1 * lc2 * sin(theta2)
        dd1_dtheta2 = 2 * dd2_dtheta2
        dphi2_dtheta1 = dphi2_dtheta2 = m2 * lc2 * g * cos(theta1 + theta2)
        dphi1_dtheta1 = (m1 * lc1 + m2 * l1) * g * cos(theta1) + dphi2_dtheta1
        dphi1_dtheta2 = - m2 * l1 * lc2 * (dtheta2 ** 2 + 2 * dtheta2 * dtheta1) * cos(theta2) + dphi2_dtheta2
        dphi1_ddtheta1 = -2 * m2 * l1 * lc2 * dtheta2 * sin(theta2)
        dphi1_ddtheta2 = -2 * m2 * l1 * lc2 * (dtheta2 + dtheta1) * sin(theta2)
        
        ddtheta2_dtheta1 = (dtau_dtheta1 + d2 / d1 * dphi1_dtheta1 - dphi2_dtheta1) / ddtheta2_den

        ddtheta2_num_dtheta2 = dtau_dtheta2 + d2 /d1 * dphi1_dtheta2 + phi1 * (dd2_dtheta2 * d1 - d2 * dd1_dtheta2) / (d1 ** 2) - m2 * l1 * lc2 * dtheta2 ** 2 * cos(theta2) - dphi2_dtheta2
        ddtheta2_den_dtheta2 = - (2 * d2 * dd2_dtheta2 * d1 - dd1_dtheta2 * d2 ** 2) / d1 ** 2
        ddtheta2_dtheta2 = (ddtheta2_num_dtheta2 * ddtheta2_den - ddtheta2_num * ddtheta2_den_dtheta2) / ddtheta2_den ** 2

        ddtheta2_ddtheta1 = (dtau_ddtheta1 + d2 / d1 * dphi1_ddtheta1 - 2 * m2 * l1 * lc2 * dtheta1 * sin(theta2)) / ddtheta2_den
        ddtheta2_ddtheta2 = (dtau_ddtheta2 + d2 / d1 * dphi1_ddtheta2) / ddtheta2_den

        ddtheta1_dtheta1 = - (d2 * ddtheta2_dtheta1 + dphi1_dtheta1) / d1
        ddtheta1_dtheta2 = - ((dd2_dtheta2 * ddtheta2 + d2 * ddtheta2_dtheta2 + dphi1_dtheta2) * d1 - (d2 * dtheta2 + phi1) * dd1_dtheta2) / d1 ** 2
        ddtheta1_ddtheta1 = -(d2 * ddtheta2_ddtheta1 + dphi1_ddtheta1) / d1
        ddtheta1_ddtheta2 = -(d2 * ddtheta2_ddtheta2 + dphi1_ddtheta2) / d1

        ddtheta2_du = dtau_du / ddtheta2_den
        ddtheta1_du = - d2 * ddtheta2_du / d1

        f_bar = np.array([dtheta1, dtheta2, ddtheta1, ddtheta2])
        A = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [ddtheta1_dtheta1, ddtheta1_dtheta2, ddtheta1_ddtheta1, ddtheta1_ddtheta2],
                      [ddtheta2_dtheta1, ddtheta2_dtheta2, ddtheta2_ddtheta1, ddtheta2_ddtheta2]])
        B = np.array([[0],
                      [0],
                      [ddtheta1_du],
                      [ddtheta2_du]])

        return (f_bar, A, B)

    def f(self, x, u):
        s_augmented = np.append(x, u)

        solve = solve_ivp(self._dsdt, [0, self.dt], s_augmented, t_eval=[self.dt])

        xn = np.zeros(4)
        xn[0] = solve.y[0, 0]
        xn[1] = wrap(solve.y[1, 0], -pi, pi)
        xn[2] = bound(solve.y[2, 0], -self.MAX_VEL_1, self.MAX_VEL_1)
        xn[3] = bound(solve.y[3, 0], -self.MAX_VEL_2, self.MAX_VEL_2)

        return xn


    def step(self, u):
        x = self.state

        # Add noise to the force action
        if self.torque_noise_max > 0:
            u += self.np_random.uniform(-self.torque_noise_max, self.torque_noise_max)

        self.state = self.f(x, u)
        
        self.state += np.random.multivariate_normal([0, 0, 0, 0], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0.0001, 0], [0, 0, 0, 0.0001]])

        return self.state

    def _dsdt(self, t, x_augmented):
        u = x_augmented[-1]
        x = x_augmented[:-1]
        dx = self.dynamics(x, u)

        return np.append(dx, 0)

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500,500)
            bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2  # 2.2 for default
            self.viewer.set_bounds(-bound,bound,-bound,bound)

        if s is None: return None

        p1 = [-self.LINK_LENGTH_1 *
              cos(s[0]), self.LINK_LENGTH_1 * sin(s[0])]

        p2 = [p1[0] - self.LINK_LENGTH_2 * cos(s[0] + s[1]),
              p1[1] + self.LINK_LENGTH_2 * sin(s[0] + s[1])]

        xys = np.array([[0,0], p1, p2])[:,::-1]
        thetas = [s[0]- pi/2, s[0]+s[1]-pi/2]
        link_lengths = [self.LINK_LENGTH_1, self.LINK_LENGTH_2]

        for ((x,y),th,llen) in zip(xys, thetas, link_lengths):
            l,r,t,b = 0, llen, .1, -.1
            jtransform = rendering.Transform(rotation=th, translation=(x,y))
            link = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
            link.add_attr(jtransform)
            link.set_color(0,.8, .8)
            circ = self.viewer.draw_circle(.1)
            circ.set_color(.8, .8, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def wrap(x, m, M):
    """Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.
    Args:
        x: a scalar
        m: minimum possible value in range
        M: maximum possible value in range
    Returns:
        x: a scalar, wrapped
    """
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x

def bound(x, m, M=None):
    """Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].
    Args:
        x: scalar
    Returns:
        x: scalar, bound between min (m) and Max (M)
    """
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)