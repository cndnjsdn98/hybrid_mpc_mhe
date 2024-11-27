""" Implementation of the nonlinear optimizer for the data-augmented RHCE.

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

import casadi as cs
import numpy as np
from acados_template import AcadosModel

from src.utils.utils import skew_symmetric, v_dot_q
from src.utils.DirectoryConfig import DirectoryConfig as DirConfig

class QuadOptimizer:
    def __init__(self, quad, t_mpc=None, n_mpc=None, t_mhe=None, n_mhe=None, mhe_type=None,
                 node_mpc=False, node_mhe=False,
                 change_mass=0):
        """
        :param quad: quadrotor object
        :type quad: Quadrotor3D
        :param t_mpc: time horizon for MPC optimization
        :param n_mpc: number of optimization nodes in time horizon for MP
        :param t_mhe: time horizon for MHE optimization
        :param n_mhe: number of optimization nodes in time horizon for MHE
        :param q_mpc: diagonal of Qc matrix for LQR cost of MPC cost function. Must be a numpy array of length 12.
        :param qt_factor: integer to set the terminal cost of MPC as a factor/multiple of q_mpc.
        :param r_mpc: diagonal of Rc matrix for LQR cost of MPC cost function. Must be a numpy array of length 4.
        :param q_mhe: diagonal of Qe matrix for model mismatch cost of MHE cost function. Must be a numpy array of length 12.
        :param q0_factor: integer to set the arrival cost of MHE as a factor/multiple of q_mhe.
        :param r_mhe: diagonal of Re matrix for measurement mismatch cost of MHE cost function. Must be a numpy array of length ___.
        :param solver_options: Optional set of extra options dictionary for solvers.
        if not used
        """
        self.mhe_type = mhe_type 
        self.change_mass = change_mass
        self.node_mpc = node_mpc
        self.node_mhe = node_mhe


        self.T_mpc = t_mpc  # Time horizon for MPC
        self.N_mpc = n_mpc  # number of nodes within prediction horizon
        self.T_mhe = t_mhe  # Time horizon for MHE
        self.N_mhe = n_mhe  # number of nodes within estimation horizon

        self.quad = quad

        self.max_u = quad.max_input_value
        self.min_u = quad.min_input_value

        # Declare model variables
        self.p = cs.MX.sym('p', 3)  # position
        self.q = cs.MX.sym('q', 4)  # angle quaternion (wxyz)
        self.v = cs.MX.sym('v', 3)  # velocity
        self.r = cs.MX.sym('r', 3)  # angle rate
        # Full state vector (13-dimensional)
        self.x = cs.vertcat(self.p, self.q, self.v, self.r)
        self.state_dim = 13

        # Full differential state vector 
        self.p_dot = cs.MX.sym('p_dot', 3)
        self.q_dot = cs.MX.sym('q_dot', 4)
        self.v_dot_w = cs.MX.sym('v_dot_w', 3)
        self.r_dot = cs.MX.sym('r_dot', 3)
        self.x_dot = cs.vertcat(self.p_dot, self.q_dot, self.v_dot_w, self.r_dot)

        # Control input vector
        u1 = cs.MX.sym('u1')
        u2 = cs.MX.sym('u2')
        u3 = cs.MX.sym('u3')
        u4 = cs.MX.sym('u4')
        self.u = cs.vertcat(u1, u2, u3, u4)
        
        # Declare model noise variables for MHE
        self.w_p = cs.MX.sym('w_p', 3)  # position
        self.w_q = cs.MX.sym('w_q', 4)  # angle quaternion (wxyz)
        self.w_v = cs.MX.sym('w_v', 3)  # velocity
        self.w_r = cs.MX.sym('w_r', 3)  # angle rate
        self.w_a = cs.MX.sym('w_a', 3)  # linear acceleration
        self.w = cs.vertcat(self.w_p, self.w_q, self.w_v, self.w_r)

        # Declare additional model variables for MHE
        if self.change_mass != 0 and mhe_type=="dynamic":
            self.k_m = cs.MX.sym('km') # quadrotor mass
            self.m_dot = cs.MX.sym('km_dot')
            self.w_km = cs.MX.sym('w_km')
            self.km_hist = np.zeros((0, 1))

            self.param = cs.vertcat(self.k_m)
            self.param_dot = cs.vertcat(self.m_dot)
            self.w_param = cs.vertcat(self.w_km)
            self.n_param = 1
            self.param_init = np.array([0])
            self.param_lbx = np.array([0])
            self.param_ubx = np.array([20])
        else:
            self.k_m = 0.0
            self.param = np.zeros((0, 0))
            self.param_dot = np.zeros((0, 0))
            self.w_param = np.zeros((0, 0))
            self.n_param = 0

        # MPC param
        if self.change_mass != 0:
            self.load_m= cs.MX.sym('load_m') # quadrotor mass
            self.mpc_param = cs.vertcat(self.load_m)
        else:
            self.load_m = 0
            self.mpc_param = np.array([])

        # Model Disturbance terms for MHE and MPC with GP 
        if self.node_mhe:
            self.d = cs.MX.sym('d', 3)  # acceleration disturbance (from GP) 
            self.d_dot = cs.MX.sym('d_dot', 3)
            self.w_d = cs.MX.sym('w_d', 3)  # acceleration disturbance
            self.n_d = 3
            self.d_init = np.array([0, 0, 0])
        elif self.node_mpc:
            self.d = cs.MX.sym('d', 3)  # acceleration disturbance (from GP) 
            self.d_dot = np.zeros((0, 0))
            self.w_d = np.zeros((0, 0))
            self.n_d = 0
        else:
            self.d = np.zeros((0, 0)) 
            self.d_dot = np.zeros((0, 0))
            self.w_d = np.zeros((0, 0))
            self.n_d = 0

        # Model adjustments for MHE
        if mhe_type == "kinematic":
            self.a = cs.MX.sym('a', 3)  # IMU acc measurement
            self.a_dot = cs.MX.sym('a_dot', 3)
            # Full state vector (16-dimensional)
            self.x = cs.vertcat(self.x, self.a)
            self.x_dot = cs.vertcat(self.x_dot, self.a_dot)
            self.state_dim = 16
            # Full state noise vector (16-dimensional)
            self.w = cs.vertcat(self.w, self.w_a)
            # Update Full input state vector
            self.u = cs.vertcat()
            # Full measurement state vector
            self.y = cs.vertcat(self.p, self.r, self.a)
        elif mhe_type == "dynamic":
            # Full state vector (13-dimensional)
            self.x = cs.vertcat(self.x, self.d, self.param)
            self.x_dot = cs.vertcat(self.x_dot, self.d_dot, self.param_dot)
            if self.node_mhe:
                self.state_dim = 16
            else:
                self.state_dim = 13
            # Full state noise vector (13-dimensional)
            self.w = cs.vertcat(self.w, self.w_d)

            f_thrust = self.u * self.quad.max_thrust 
            self.a = cs.vertcat(0.0, 0.0, (f_thrust[0] + f_thrust[1] + f_thrust[2] + f_thrust[3])) / (self.quad.mass + self.k_m) #a_thrust
            # Full measurement state vector
            self.y = cs.vertcat(self.p, self.r, self.d)
        
        # The trigger variable is used to tell ACADOS to use the additional GP state estimate in the first optimization
        # node and the regular integrated state in the rest
        self.trigger_var = cs.MX.sym('trigger', 1)

        # Initialize objective function, 0 target state and integration equations
        self.L = None
        self.target = None
        
        # Declare model variables for GP prediction (only used in real quadrotor flight with EKF estimator).
        # Will be used as initial state for GP prediction as Acados parameters.
        self.gp_p = cs.MX.sym('gp_p', 3)
        self.gp_q = cs.MX.sym('gp_q', 4)
        self.gp_v = cs.MX.sym('gp_v', 3)
        self.gp_r = cs.MX.sym('gp_r', 3)
        self.gp_x = cs.vertcat(self.gp_p, self.gp_q, self.gp_v, self.gp_r)

        self.B_x = np.zeros((13, 3))
            
        # functions of the state and input vectors
        self.quad_xdot = {}
        self.quad_xdot_mhe = {}
        self.quad_xdot_prop = self.quad_dynamics(prop=True, mhe=not mhe_type is None, mhe_type=mhe_type)

        # Acados OCP solvers
        self.acados_mpc_solver = {}
        self.acados_mhe_solver = {}

        self.acados_models_dir = DirConfig.ACADOS_MODEL_DIR

    def acados_setup_model(self, nominal, model_name, mhe=False):
        """
        Builds an Acados symbolic models using CasADi expressions.
        :param model_name: name for the acados model. Must be different from previously used names or there may be
        problems loading the right model.
        :param nominal: CasADi symbolic nominal model of the quadrotor: f(self.x, self.u) = x_dot, dimensions 13x1.
        :param mhe: Boolean variable. True if model is for MHE, and False if model is for MPC. 
        :return: Returns a total of three outputs, where m is the number of GP's in the GP ensemble, or 1 if no GP:
            - A dictionary of m AcadosModel of the GP-augmented quadrotor
            - A dictionary of m CasADi symbolic nominal dynamics equations with GP mean value augmentations (if with GP)
        :rtype: dict, dict, cs.MX
        """

        def fill_in_acados_model(x, x_dot, u, w, p, dynamics, name, mhe):

            f_impl = x_dot - dynamics

            # Dynamics model
            model = AcadosModel()
            model.f_expl_expr = dynamics
            model.f_impl_expr = f_impl
            model.x = x
            model.xdot = x_dot
            if mhe:
                model.u = w
                model.p = cs.vertcat(p, u)
                model.name = "mhe"
                # Quaternion normalization constraint
                # model.con_h_expr = self.q[0]**2 + self.q[1]**2 + self.q[2]**2 + self.q[3]**2
            else:
                model.u = u
                model.p = p
                model.name = "mpc"

            return model

        acados_models = {}
        dynamics_equations = {}

        # Run GP inference if GP's available               
        if self.node_mpc and not mhe:
            # The corrections are passed in prior to the ocp_mpc.
            # These states are used to transform the corrections to world reference frame.
            # First tranformation will be done using the gp_x that will be passed in as params,
            # however the rest of the transformation will be done using the regular integrated states.     
            # gp_x = self.gp_x * self.trigger_var + self.x * (1 - self.trigger_var)

            # Transform back to world reference frame
            # gp_means = v_dot_q(self.d, gp_x[3:7])   
            gp_means = v_dot_q(self.d, self.x[3:7])

            # Add GP mean prediction parameters and additive state noise if model is for MHE
            dynamics_equations[0] = nominal + cs.mtimes(self.B_x, gp_means)

            x_dot_ = self.x_dot
            x_ = self.x
            w_ = self.w
            dynamics_ = dynamics_equations[0]
            i_name = model_name + "_gpy"

            # params = cs.vertcat(self.mpc_param, self.gp_x, self.d, self.trigger_var)
            params = cs.vertcat(self.mpc_param, self.d)
            acados_models[0] = fill_in_acados_model(x=x_, x_dot=x_dot_, u=self.u, w=w_, p=params, dynamics=dynamics_, name=i_name, mhe=mhe)

        elif self.node_mhe and mhe:
            # The corrections are passed in prior to the ocp_mpc.
            # These states are used to transform the corrections to world reference frame.
            # First tranformation will be done using the gp_x that will be passed in as params,
            # however the rest of the transformation will be done using the regular integrated states.     

            # Transform back to world reference frame
            gp_means = v_dot_q(self.d, self.q)   

            # Add GP mean prediction parameters and additive state noise if model is for MHE
            dynamics_equations[0] = nominal + cs.mtimes(self.B_x, gp_means)

            x_dot_ = self.x_dot
            x_ = self.x
            w_ = self.w
            u_ = self.u
            dynamics_ = dynamics_equations[0]
            i_name = model_name + "_gpy"

            # params = cs.vertcat(self.w_v)
            params = []
            acados_models[0] = fill_in_acados_model(x=x_, x_dot=x_dot_, u=u_, w=w_, p=params, dynamics=dynamics_, name=i_name, mhe=mhe)
        else:
            # No available GP so return nominal dynamics and add additive state noise if model is for MHE
            dynamics_equations[0] = nominal
            x_dot_ = self.x_dot
            x_ = self.x
            u_ = self.u
            w_ = self.w
            if mhe:
                param = []
            else:
                param = self.mpc_param

            dynamics_ = nominal

            acados_models[0] = fill_in_acados_model(x=x_, x_dot=x_dot_, u=u_, w=w_, p=param, dynamics=dynamics_, name=model_name, mhe=mhe)

        return acados_models, dynamics_equations

    def quad_dynamics(self, mhe=False, mhe_type="kinematic", prop=False):
        """
        Symbolic dynamics of the 3D quadrotor model. The state consists on: [p_xyz, a_wxyz, v_xyz, r_xyz]^T, where p
        stands for position, a for angle (in quaternion form), v for velocity and r for body rate. The input of the
        system is: [u_1, u_2, u_3, u_4], i.e. the activation of the four thrusters.

        :return: CasADi function that computes the analytical differential state dynamics of the quadrotor model.
        Inputs: 'x' state of quadrotor (6x1) and 'u' control input (2x1). Output: differential state vector 'x_dot'
        (6x1)
        """
        if mhe:
            if mhe_type == "kinematic":
                x_dot_mhe = cs.vertcat(self.p_dynamics(mhe=mhe), self.q_dynamics(mhe=mhe), 
                                        self.v_dynamics(mhe=mhe), self.w_dynamics(mhe=mhe))
                x_dot_mhe = cs.vertcat(x_dot_mhe, np.zeros(3) + self.w_a)
                return cs.Function('x_dot_mhe', [self.x, self.u, self.w], [x_dot_mhe], ['x', 'u', 'w'], ['x_dot'])
            elif mhe_type == "dynamic":
                x_dot_mhe = cs.vertcat(self.p_dynamics(mhe=mhe), self.q_dynamics(mhe=mhe), 
                                        self.v_dynamics(mhe=mhe), self.w_dynamics(mhe=mhe))
                x_dot_mhe = cs.vertcat(x_dot_mhe, np.zeros(self.n_d)+self.w_d, np.zeros(self.n_param))
                return cs.Function('x_dot_mhe', [self.x, self.u, self.w], [x_dot_mhe], ['x', 'u', 'w'], ['x_dot'])
        elif prop:
            x_dot = cs.vertcat(self.p_dynamics(), self.q_dynamics(), self.v_dynamics(), self.w_dynamics())
            return cs.Function('x_dot', [self.x, cs.vertcat(self.u, self.mpc_param)], [x_dot], ['x', 'u'], ['x_dot'])
        else:
            x_dot = cs.vertcat(self.p_dynamics(), self.q_dynamics(), self.v_dynamics(), self.w_dynamics())
            return cs.Function('x_dot', [self.x, self.u, self.mpc_param], [x_dot], ['x', 'u', 'p'], ['x_dot'])

    def p_dynamics(self, mhe=False):
        if mhe:
            return self.v + self.w_p
        else:
            return self.v

    def q_dynamics(self, mhe=False):
        if mhe:
            return 1 / 2 * cs.mtimes(skew_symmetric(self.r), self.q) + self.w_q
        else:
            return 1 / 2 * cs.mtimes(skew_symmetric(self.r), self.q)

    def v_dynamics(self, mhe=False):
        """
        :param
        """
        g = cs.vertcat(0.0, 0.0, 9.81)
        if mhe:
            v_dynamics = v_dot_q(self.a, self.q) - g + self.w_v
        else:
            f_thrust = self.u * self.quad.max_thrust
            a_thrust = cs.vertcat(0.0, 0.0, f_thrust[0] + f_thrust[1] + f_thrust[2] + f_thrust[3]) / (self.quad.mass + self.load_m)

            v_dynamics = v_dot_q(a_thrust, self.q) - g

        return v_dynamics

    def w_dynamics(self, mhe=False):
        if mhe:
            return cs.vertcat(0, 0, 0) + self.w_r
        else:
            f_thrust = self.u * self.quad.max_thrust

            y_f = cs.MX(self.quad.y_f)
            x_f = cs.MX(self.quad.x_f)
            c_f = cs.MX(self.quad.z_l_tau)
            return cs.vertcat(
                (cs.mtimes(f_thrust.T, y_f) + (self.quad.J[1] - self.quad.J[2]) * self.r[1] * self.r[2]) / self.quad.J[0],
                (-cs.mtimes(f_thrust.T, x_f) + (self.quad.J[2] - self.quad.J[0]) * self.r[2] * self.r[0]) / self.quad.J[1],
                (cs.mtimes(f_thrust.T, c_f) + (self.quad.J[0] - self.quad.J[1]) * self.r[0] * self.r[1]) / self.quad.J[2])

    def discretize_dynamics(self, t_horizon, m_steps_per_point=1):
        """
        Integrates the symbolic dynamics and cost equations until the time horizon using a RK4 method.
        :param t_horizon: time horizon in seconds
        :param m_steps_per_point: number of integrations steps
        :return: a symbolic function that computes the dynamics integration
        """

        # Fixed step Runge-Kutta 4 integrator
        dt = t_horizon / m_steps_per_point
        u = cs.vertcat(self.u, self.mpc_param)
        x0 = self.x

        for _ in range(m_steps_per_point):
            k1 = self.quad_xdot_prop(x=self.x, u=u)['x_dot']
            k2 = self.quad_xdot_prop(x=self.x + dt / 2 * k1, u=u)['x_dot']
            k3 = self.quad_xdot_prop(x=self.x + dt / 2 * k2, u=u)['x_dot']
            k4 = self.quad_xdot_prop(x=self.x + dt * k3, u=u)['x_dot']
            x_out = self.x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return cs.Function('F', [x0, u], [x_out], ['x0', 'p'], ['xf'])
    

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
        fk = f_func(x0=x_0, p=u_seq)
        xf = np.squeeze(np.array(fk['xf']))

        return xf

