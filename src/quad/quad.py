""" Implementation of the Simplified Simulator and its quadrotor dynamics.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

import os
import numpy as np
import casadi as cs 

from src.utils.utils import parse_xacro_file, v_dot_q, skew_symmetric
from src.utils.DirectoryConfig import DirectoryConfig as DirConfig

class Quadrotor:

    def __init__(self, quad_name, prop=False):
        """
        Initialization of the 3D quadrotor class
        :param noisy: Whether noise is used in the simulation
        :type noisy: bool
        :param drag: Whether to simulate drag or not.
        :type drag: bool
        :param payload: Whether to simulate a payload force in the simulation
        :type payload: bool
        :param motor_noise: Whether non-gaussian noise is considered in the motor inputs
        :type motor_noise: bool
        """
        self.custom_quad_param_loader(quad_name)

        if prop:
            self.x = cs.MX.sym('x', 13)
            self.u = cs.MX.sym('u', 4)
            self.quad_xdot_prop = self.dynamics(self.x, self.u)

    def custom_quad_param_loader(self, quad_name):

        quad_param_dir = DirConfig.QUAD_PARAM_DIR
        quad_param = os.path.join(quad_param_dir, quad_name, quad_name + '.xacro')

        # Get parameters for drone
        attrib = parse_xacro_file(quad_param)

        self.quad_name = quad_name

        self.base_mass = float(attrib['mass']) # mass of just the frame/base
        self.mass = (float(attrib['mass']) + float(attrib['mass_rotor']) * 4)
        self.J = np.array([float(attrib['body_inertia'][0]['ixx']),
                        float(attrib['body_inertia'][0]['iyy']),
                        float(attrib['body_inertia'][0]['izz'])])
        self.length = float(attrib['arm_length'])

        self.motor_constant = float(attrib["motor_constant"])
        if 'max_thrust' in attrib.keys() and attrib['max_thrust'] is not None:
            self.max_thrust = float(attrib['max_thrust'])
        else:
            self.max_thrust = float(attrib["max_rot_velocity"]) ** 2 * float(attrib["motor_constant"])
        self.c = float(attrib['moment_constant'])
        self.rotor_drag_coeff = float(attrib["rotor_drag_coefficient"])

        # Input constraints
        if 'max_input_value' in attrib.keys():
            self.max_input_value = float(attrib['max_input_value'])  # Motors at full thrust
        else:
            self.max_input_value = 1
        if 'min_input_value' in attrib.keys():
            self.min_input_value = float(attrib['min_input_value'])  # Motors turned off
        else:
            self.min_input_value = 0

        # x configuration
        if quad_name != "hummingbird":
            dx = float(attrib['rotor_dx'])
            dy = float(attrib['rotor_dy'])
            dz = float(attrib['rotor_dz'])

            self.x_f = np.array([dx, -dx, -dx, dx])
            self.y_f = np.array([-dy, -dy, dy, dy])
            self.z_l_tau = np.array([-self.c, self.c, -self.c, self.c])

        # + configuration
        else:
            self.x_f = np.array([self.length, 0, -self.length, 0])
            self.y_f = np.array([0, self.length, 0, -self.length])
            self.z_l_tau = -np.array([-self.c, self.c, -self.c, self.c])

        # Compute hover thrust
        self.hover_thrust = (self.mass * 9.81) / (self.max_thrust * 4)

        return 
    
    def dynamics(self, x, u, payload=None):
        """
        Symbolic dynamics of the 3D quadrotor model. The state consists on: [p_xyz, a_wxyz, v_xyz, r_xyz]^T, where p
        stands for position, a for angle (in quaternion form), v for velocity and r for body rate. The input of the
        system is: [u_1, u_2, u_3, u_4], i.e. the activation of the four thrusters.

        :return: CasADi function that computes the analytical differential state dynamics of the quadrotor model.
        Inputs: 'x' state of quadrotor (6x1) and 'u' control input (2x1). Output: differential state vector 'x_dot'
        (6x1)
        """
        if payload is not None:
            x_dot = cs.vertcat(self.p_dynamics(x), self.q_dynamics(x), self.v_dynamics(x, u, payload), self.w_dynamics(x, u))
            return cs.Function('x_dot', [x, u, payload], [x_dot], ['x', 'u', 'p'], ['x_dot'])
        else:
            x_dot = cs.vertcat(self.p_dynamics(x), self.q_dynamics(x), self.v_dynamics(x, u), self.w_dynamics(x, u))
            return cs.Function('x_dot', [x, u], [x_dot], ['x', 'u'], ['x_dot'])

    def p_dynamics(self, x):
        v = x[7:10]
        return v

    def q_dynamics(self, x):
        q = x[3:7]
        r = x[10:]
        return 1 / 2 * cs.mtimes(skew_symmetric(r), q)

    def v_dynamics(self, x, u, payload=None):
        q = x[3:7]

        g = cs.vertcat(0.0, 0.0, 9.81)
        f_thrust = u * self.max_thrust
        if payload is not None:
            a_thrust = cs.vertcat(0.0, 0.0, f_thrust[0] + f_thrust[1] + f_thrust[2] + f_thrust[3]) / (self.mass + payload)
        else:
            a_thrust = cs.vertcat(0.0, 0.0, f_thrust[0] + f_thrust[1] + f_thrust[2] + f_thrust[3]) / (self.mass)

        v_dynamics = v_dot_q(a_thrust, q) - g

        return v_dynamics

    def w_dynamics(self, x, u):
        r = x[10:]

        f_thrust = u * self.max_thrust

        y_f = cs.MX(self.y_f)
        x_f = cs.MX(self.x_f)
        c_f = cs.MX(self.z_l_tau)
        return cs.vertcat(
            (cs.mtimes(f_thrust.T, y_f) + (self.J[1] - self.J[2]) * r[1] * r[2]) / self.J[0],
            (-cs.mtimes(f_thrust.T, x_f) + (self.J[2] - self.J[0]) * r[2] * r[0]) / self.J[1],
            (cs.mtimes(f_thrust.T, c_f) + (self.J[0] - self.J[1]) * r[0] * r[1]) / self.J[2])

    def discretize_dynamics(self, t_horizon, m_steps_per_point=1):
        """
        Integrates the symbolic dynamics and cost equations until the time horizon using a RK4 method.
        :param t_horizon: time horizon in seconds
        :param m_steps_per_point: number of integrations steps
        :return: a symbolic function that computes the dynamics integration
        """

        # Fixed step Runge-Kutta 4 integrator
        dt = t_horizon / m_steps_per_point
        u = self.u
        x0 = self.x

        for _ in range(m_steps_per_point):
            k1 = self.quad_xdot_prop(x=x0, u=u)['x_dot']
            k2 = self.quad_xdot_prop(x=x0 + dt / 2 * k1, u=u)['x_dot']
            k3 = self.quad_xdot_prop(x=x0 + dt / 2 * k2, u=u)['x_dot']
            k4 = self.quad_xdot_prop(x=x0 + dt * k3, u=u)['x_dot']
            x_out = x0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return cs.Function('F', [x0, u], [x_out], ['x0', 'u'], ['xf'])
    

    def forward_prop(self, x_0, u_seq, t_horizon, m_int_steps=1):
        """
        Propagates forward the state estimate described by the mean vector x_0 and the covariance matrix covar, and a
        sequence of inputs for the system u_seq. These inputs can either be numerical or symbolic.

        :param x_0: initial mean state of the state probability density function. Vector of length m
        :param u_seq: sequence of flattened N control inputs. I.e. vector of size N*4
        :param t_horizon: time horizon corresponding to sequence of inputs
        :param m_int_steps: number of intermediate integration steps per control node.
        :return: The sequence of mean and covariance estimates for every corresponding input, as well as the computed
        cost for each stage.
        """
        if not isinstance(x_0, np.ndarray):
            x_0 = np.array(x_0)

        f_func = self.discretize_dynamics(t_horizon, m_int_steps)
        fk = f_func(x0=x_0, u=u_seq)
        xf = np.squeeze(np.array(fk['xf']))

        return xf

    def get_hover_thrust(self):
        return self.hover_thrust

    def get_base_mass(self):
        return self.base_mass
    
    def get_mass(self):
        return self.mass
    
    def get_max_thrust(self):
        return self.max_thrust