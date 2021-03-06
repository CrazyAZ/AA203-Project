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

    dt = 0.006

    LINK_LENGTH_1 = 0.305  # [m]
    LINK_LENGTH_2 = 0.35  # [m]
    LINK_MASS_1 = 0.130  #: [kg] mass of link 1
    LINK_MASS_2 = 0.088  #: [kg] mass of link 2
    LINK_COM_POS_1 = 0.21  #: [m] position of the center of mass of link 1
    LINK_COM_POS_2 = 0.185  #: [m] position of the center of mass of link 2
    LINK_MOI_1 = 7.6e-3  #: moment of inertia around pivot for link 1
    LINK_MOI_2 = 3.6e-3  #: moment of inertia around pivot for link 2

    g = 9.8

    KP = 1.2
    KD = 0.09 
    KS = 0.4

    MAX_VEL_1 = 4 * pi
    MAX_VEL_2 = pi / 0.45

    deadband = 0.0072

    torque_noise_max = 0.

    coeff_friction = 0.0

    state_noise_covariance = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0.000001, 0], [0, 0, 0, 0.0001]])

    C = np.array([[1, 0, 0, 0]])

    measurement_noise_covariance = 0.0001 * np.eye(1)

    def __init__(self):
        self.viewer = None
    
    def linearized_dynamics(self, x_bar, u_bar):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI_1
        I2 = self.LINK_MOI_2
        g = self.g
        kp = self.KP
        kd = self.KD
        ks = self.KS

        theta1 = x_bar[0]
        theta2 = x_bar[1]
        dtheta1 = x_bar[2]
        dtheta2 = x_bar[3]

        tau = kp * ks * np.tanh((u_bar - theta2) / ks) - kd * dtheta2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * cos(theta2)) + I2
        d1 = m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * cos(theta2)) + I1 + I2
        phi2 = m2 * lc2 * g * sin(theta1 + theta2)
        phi1 = - m2 * l1 * lc2 * (dtheta2 ** 2 + 2 * dtheta2 * dtheta1) * sin(theta2) + (m1 * lc1 + m2 * l1) * g * sin(theta1) + phi2
        ddtheta2_num = (tau + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * sin(theta2) - phi2)
        ddtheta2_den = m2 * lc2 ** 2 + I2 - d2 ** 2 / d1
        ddtheta2 = ddtheta2_num / ddtheta2_den
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

        dtau_dtheta1 = dtau_ddtheta1 = 0
        dtau_dtheta2 = -kp / np.cosh((u_bar - theta2) / ks)**2
        dtau_ddtheta2 = -kd
        dtau_du = -dtau_dtheta2

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
        ddtheta1_dtheta2 = - ((dd2_dtheta2 * ddtheta2 + d2 * ddtheta2_dtheta2 + dphi1_dtheta2) * d1 - (d2 * ddtheta2 + phi1) * dd1_dtheta2) / d1 ** 2
        ddtheta1_ddtheta1 = -(d2 * ddtheta2_ddtheta1 + dphi1_ddtheta1) / d1 - self.coeff_friction
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

        return f_bar, A, B

    def render(self, s, mode='human'):
        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(1000,1000)
            bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.04  # 2.2 for default
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
            l,r,t,b = 0, llen, .03, -.03
            jtransform = rendering.Transform(rotation=th, translation=(x,y))
            link = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
            link.add_attr(jtransform)
            link.set_color(0,.8, .8)
            circ = self.viewer.draw_circle(.03)
            circ.set_color(.8, .8, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

