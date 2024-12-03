#!/usr/bin/env python

""" Implementation of the nonlinear optimizer for the data-augmented MHE.

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
import sys
import casadi as cs
import numpy as np
import threading
import rospy
import l4casadi as l4c

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from src.quad_opt.quad import custom_quad_param_loader
from src.utils.utils import v_dot_q, skew_symmetric

from src.gp.GPyModelWrapper import GPyModelWrapper
class QuadOptimizerMPC:
    def __init__(self, quad, t_mpc=1, n_mpc=10, 
                 q_mpc=None, qt_factor=None,r_mpc=None, 
                 quad_name=None, solver_options=None, 
                 use_nn=False, nn_params={},
                 compile_acados=True):
        """
        :param quad: quadrotor object
        :type quad: Quadrotor3D
        :param t_mpc: time horizon for MPC optimization
        :param n_mpc: number of optimization nodes in time horizon for MP
        :param q_mpc: diagonal of Qc matrix for LQR cost of MPC cost function. Must be a numpy array of length 12.
        :param qt_factor: integer to set the terminal cost of MPC as a factor/multiple of q_mpc.
        :param r_mpc: diagonal of Rc matrix for LQR cost of MPC cost function. Must be a numpy array of length 4.
        :param B_x: dictionary of matrices that maps the outputs of the gp regressors to the state space.
        :param mpc_gp_regressors: Gaussian Process ensemble for correcting the nominal model
        :param solver_options: Optional set of extra options dictionary for solvers.
        """
        # MPC Params
        self.T = t_mpc
        self.N = n_mpc
        # Quad
        self.quad = quad
        self.max_u = quad.max_input_value
        self.min_u = quad.min_input_value
        
        # Use Data Driven (Neural Network) models
        self.use_nn = use_nn
        self.nn_params = nn_params
        if self.use_nn:
            self.model_name = self.nn_params['model_name']
            self.model_type = self.nn_params['model_type']
            self.input_features = self.nn_params['input_features']
            self.nn_input_idx = self.nn_params['nn_input_idx']
            self.output_features = self.nn_params['output_features']
            self.nn_output_idx = self.nn_params['nn_output_idx']
            self.correction_mode = self.nn_params['correction_mode']
            self.nn_model = self.nn_params['nn_model']
            if self.correction_mode == "online":
                self.nn_model = l4c.L4CasADi(self.nn_mdoel, device="cpu")

        # Init Casadi variables
        self.init_cs_vectors()

        # Initi variables
        self.x_opt = np.zeros((self.N + 1, 13))
        self.u_opt = np.zeros((self.N, 4))
        self.opt_dt = 0

        # functions of the state and input vectors used for propagating states 
        # to compute model errors
        self.quad_xdot_prop = self.quad_dynamics()

        # Nominal model equations symbolic function (no GP)
        self.quad_xdot_nominal = self.quad_dynamics()

        # Build full model for MPC. Will have 13 variables. self.dyn_x contains the symbolic variable that
        # should be used to evaluate the dynamics function. It corresponds to self.x if there are no GP's, or
        # self.x_with_gp otherwise
        acados_model, dynamics = self.acados_setup_model(
            self.quad_xdot_nominal(x=self.x, u=self.u, p=self.mpc_param)['x_dot'])
        
        # Convert dynamics variables to functions of the state and input vectors
        self.quad_xdot = cs.Function('x_dot', [self.x, self.u, self.mpc_param], [dynamics], ['x', 'u', 'p'], ['x_dot'])

        # Weighted squared error loss function q = (p_xyz, a_xyz, v_xyz, r_xyz), r = (u1, u2, u3, u4)
        if q_mpc is None:
            q_mpc = np.array([10, 10, 10, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        if r_mpc is None:
            r_mpc = np.array([0.1, 0.1, 0.1, 0.1])
        if qt_factor is None:
            qt_factor = 0
        print("\n###########################################################################################")
        print("Q_control          = ", q_mpc)
        print("Q_control_Terminal = ", q_mpc * qt_factor)
        print("R_control          = ", r_mpc)
        print("###########################################################################################\n")
        # Add one more weight to the rotation (use quaternion norm weighting in acados)
        q_mpc = np.concatenate((q_mpc[:3], np.mean(q_mpc[3:6])[np.newaxis], q_mpc[3:]))   

        # ### Setup and compile Acados OCP solvers ### #
        ocp_mpc = self.create_mpc_solver(acados_model, q_mpc, qt_factor, r_mpc, solver_options)

        json_file_mpc = os.path.join(self.acados_models_dir, "mpc", acados_model.name + '.json')
        self.acados_mpc_solver = AcadosOcpSolver(ocp_mpc, json_file=json_file_mpc)

    def init_cs_vectors(self):
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

        # Model Corrections using NN
        self.mpc_param = np.array([])
        if self.use_nn:
            if self.correction_mode == "offline":
                self.nn_corr = cs.MX.sym('d', len(self.nn_output_idx))
                # Transform Velocity corrections to world frame
                if "v" in self.output_features:
                    v_states = [7, 8, 9]
                    v_idx = [i for i, x in enumerate(self.nn_output_idx) if x in v_states]
                    vb_corr = self.nn_corr[v_idx]
                    vw_corr = v_dot_q(vb_corr, self.x[3:7])
                    self.nn_corr[v_idx] = vw_corr
                self.mpc_param = cs.vertcat(self.nn_corr)
            elif self.correction_mode == "online":
                # Model variables for Online predictions
                # Used as initial state for NN prediction
                nn_input = cs.vertcat()
                input = cs.vertcat()
                if "q" in self.input_features:
                    self.nn_q = cs.MX.sym('nn_q', 4)
                    nn_input = cs.vertcat(nn_input, self.nn_q)
                    input = cs.vertcat(input, self.q)
                if "v" in self.input_features:
                    self.nn_v = cs.MX.sym('nn_v', 3)
                    nn_input = cs.vertcat(nn_input, self.nn_v)
                    input = cs.vertcat(input, self.v)
                if "w" in self.input_features:
                    self.nn_r = cs.MX.sym('nn_r', 3)
                    nn_input = cs.vertcat(nn_input, self.nn_r)
                    input = cs.vertcat(input, self.r)
                if "u" in self.input_features:
                    nn_u1 = cs.MX.sym('nn_u1')
                    nn_u2 = cs.MX.sym('nn_u2')
                    nn_u3 = cs.MX.sym('nn_u3')
                    nn_u4 = cs.MX.sym('nn_u4')
                    self.nn_u = cs.vertcat(nn_u1, nn_u2, nn_u3, nn_u4)
                    nn_input = cs.vertcat(nn_input, self.nn_u)
                    input = cs.vertcat(input, self.u)
                self.trigger_var = cs.MX.sym('trigger', 1)
                self.input = nn_input * self.trigger_var + input * (1 - self.trigger_var)
                self.nn_corr = self.nn_model(self.input)
                # Transform Velocity corrections to world frame
                if "v" in self.output_features:
                    v_states = [7, 8, 9]
                    v_idx = [i for i, x in enumerate(self.nn_output_idx) if x in v_states]
                    vb_corr = self.nn_corr[v_idx]
                    vw_corr = v_dot_q(vb_corr, self.x[3:7])
                    self.nn_corr[v_idx] = vw_corr
                self.mpc_param = cs.vertcat(nn_input, self.trigger_var)

            self.B_x = np.zeros((13, len(self.nn_output_idx)))
            for i, idx in enumerate(self.nn_output_idx):
                self.B_x[idx, i] = 1
    
    def acados_setup_model(self, nominal):
        """
        Builds an Acados symbolic models using CasADi expressions.
        :param quad_name: Name of the quadrotor
        :param nominal: CasADi symbolic nominal model of the quadrotor: f(self.x, self.u) = x_dot, dimensions 13x1.
        :param mhe: Boolean variable. True if model is for MHE, and False if model is for MPC. 

        """
        # Run inference if NN's available               
        if self.use_nn:
            # Add model correction to nominal model
            dynamics_ = nominal + cs.mtimes(self.B_x, self.nn_corr)
        else:
            dynamics_ = nominal

        x_dot_ = self.x_dot
        x_ = self.x
        u_ = self.u
        param_ = self.mpc_param

        acados_model = AcadosModel()
        acados_model.f_expl_expr = dynamics_
        acados_model.f_impl_expr = x_dot_ - dynamics_
        acados_model.x = x_
        acados_model.xdot = x_dot_
        acados_model.u = u_
        acados_model.p = param_
        acados_model.name = "mpc"
    
        return acados_model, dynamics_

    def create_mpc_solver(self, model, q_cost, qt_factor, r_cost, solver_options):
        """
        Creates OCP objects to formulate the MPC optimization
        :param model: Acados model of the system
        :type model: cs.MX 
        :param q_cost: diagonal of Q matrix for LQR cost of MPC cost function. Must be a numpy array of length 12.
        :param qt_factor: integer to set the terminal cost of MPC as a factor/multiple of q_cost
        :param r_cost: diagonal of R matrix for LQR cost of MPC cost function. Must be a numpy array of length 4.
        :param solver_options: Optional set of extra options dictionary for solvers.
        """
        nx = model.x.size()[0]
        nu = model.u.size()[0]
        ny = nx + nu
        n_param = model.p.size()[0] if isinstance(model.p, cs.MX) else 0

        acados_source_path = os.environ['ACADOS_SOURCE_DIR']
        sys.path.insert(0, '../common')

        # Create OCP object to formulate the MPC optimization
        ocp_mpc = AcadosOcp()
        ocp_mpc.acados_include_path = acados_source_path + '/include'
        ocp_mpc.acados_lib_path = acados_source_path + '/lib'
        ocp_mpc.model = model
        ocp_mpc.dims.N = self.N_mpc
        ocp_mpc.solver_options.tf = self.T_mpc

        # Initialize parameters
        ocp_mpc.dims.np = n_param
        ocp_mpc.parameter_values = np.zeros(n_param)

        ocp_mpc.cost.cost_type = 'LINEAR_LS'
        ocp_mpc.cost.cost_type_e = 'LINEAR_LS'

        ocp_mpc.cost.W = np.diag(np.concatenate((q_cost, r_cost)))
        ocp_mpc.cost.W_e = np.diag(q_cost) * qt_factor

        ocp_mpc.cost.Vx = np.zeros((ny, nx))
        ocp_mpc.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp_mpc.cost.Vu = np.zeros((ny, nu))
        ocp_mpc.cost.Vu[-4:, -4:] = np.eye(nu)

        ocp_mpc.cost.Vx_e = np.eye(nx)

        # Initial reference trajectory (will be overwritten)
        x_ref = np.zeros(nx)
        ocp_mpc.cost.yref = np.concatenate((x_ref, np.array([0.0, 0.0, 0.0, 0.0])))
        ocp_mpc.cost.yref_e = x_ref

        # Initial state (will be overwritten)
        ocp_mpc.constraints.x0 = x_ref

        # Set constraints
        ocp_mpc.constraints.lbu = np.array([self.min_u] * 4)
        ocp_mpc.constraints.ubu = np.array([self.max_u] * 4)
        ocp_mpc.constraints.idxbu = np.array([0, 1, 2, 3])

        # Solver options
        ocp_mpc.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM' #'FULL_CONDENSING_DAQP'
        ocp_mpc.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp_mpc.solver_options.integrator_type = 'ERK'
        ocp_mpc.solver_options.print_level = 0
        ocp_mpc.solver_options.nlp_solver_type = 'SQP_RTI' if solver_options is None else solver_options["solver_type"]
        # ocp_mpc.solver_options.qp_solver_warm_start = 1 # Warm Start

        # Path to where code will be exported
        ocp_mpc.code_export_directory = os.path.join(self.acados_models_dir, "mpc")

        return ocp_mpc

    def quad_dynamics(self):
        """
        Symbolic dynamics of the 3D quadrotor model. The state consists on: [p_xyz, a_wxyz, v_xyz, r_xyz]^T, where p
        stands for position, a for angle (in quaternion form), v for velocity and r for body rate. The input of the
        system is: [u_1, u_2, u_3, u_4], i.e. the activation of the four thrusters.

        :return: CasADi function that computes the analytical differential state dynamics of the quadrotor model.
        Inputs: 'x' state of quadrotor (6x1) and 'u' control input (2x1). Output: differential state vector 'x_dot'
        (6x1)
        """
        x_dot = cs.vertcat(self.p_dynamics(), self.q_dynamics(), self.v_dynamics(), self.w_dynamics())
        return cs.Function('x_dot', [self.x, self.u], [x_dot], ['x', 'u'], ['x_dot'])

    def p_dynamics(self):
        return self.v

    def q_dynamics(self):
        return 1 / 2 * cs.mtimes(skew_symmetric(self.r), self.q)

    def v_dynamics(self):
        g = cs.vertcat(0.0, 0.0, 9.81)
        f_thrust = self.u * self.quad.max_thrust
        a_thrust = cs.vertcat(0.0, 0.0, f_thrust[0] + f_thrust[1] + f_thrust[2] + f_thrust[3]) / (self.quad.mass + self.load_m)

        v_dynamics = v_dot_q(a_thrust, self.q) - g

        return v_dynamics

    def w_dynamics(self):
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
        u = self.u
        x0 = self.x

        for _ in range(m_steps_per_point):
            k1 = self.quad_xdot_prop(x=self.x, u=u)['x_dot']
            k2 = self.quad_xdot_prop(x=self.x + dt / 2 * k1, u=u)['x_dot']
            k3 = self.quad_xdot_prop(x=self.x + dt / 2 * k2, u=u)['x_dot']
            k4 = self.quad_xdot_prop(x=self.x + dt * k3, u=u)['x_dot']
            x_out = self.x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

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

    def set_reference(self, x_ref, u_ref):
        """
        Sets the reference trajectory.
        :param x_target: Nx13-dimensional reference trajectory (p_xyz, angle_wxyz, v_xyz, rate_xyz). It is passed in the
        form of a 4-length list, where the first element is a Nx3 numpy array referring to the position targets, the
        second is a Nx4 array referring to the quaternion, two more Nx3 arrays for the velocity and body rate targets.
        :param u_target: Nx4-dimensional target control input vector (u1, u2, u3, u4)
        """
        # Set the reference for the given reference length 
        ref_len = x_ref.shape[0]
        for j in range(ref_len-1):
            ref = np.concatenate((x_ref[j, :], u_ref[j, :]))
            self.acados_mpc_solver.set(j, "yref", ref)
        # Set ref for remainder of prediction horizon with final element of the reference trajectory
        ref = np.concatenate((x_ref[-1, :], u_ref[-1, :]))
        for j in range(ref_len-1, self.N):
            self.acados_mpc_solver.set(j, "yref", ref)
        # the last MPC node has only a state reference but no input reference
        self.acados_mpc_solver.set(self.N, "yref", x_ref[-1, :]) 

    def set_params(self, param):
        for j in range(self.N):
            self.acados_mpc_solver.set()
    
    def solve_mpc(self, x0=None, u0=None, nn_corr=None):
        """
        Optimizes a trajectory to reach the pre-set target state, starting from the input initial state, that minimizes
        the quadratic cost function and respects the constraints of the system

        :param x0: 13-element list of the initial state. If None, 0 state will be used
        :param u0: 4-element list of the current motor thrusts. If None, previous u_opt will be used
        :param nn_corr: Offline Model Correction terms to be used in the prediction horizon. 
        :return: optimizer status
        """

        if x0 is None:
            x0 = [0, 0, 0] + [1, 0, 0, 0] + [0, 0, 0] + [0, 0, 0]
        # Set initial state. Add gp state if needed
        x0 = np.stack(x0)
        if u0 is None:
            u0 = self.u_opt[0]
        u0 = np.stack(u0)
        self.acados_mpc_solver.set(0, 'lbx', x0)
        self.acados_mpc_solver.set(0, 'ubx', x0)

        # Set parameters for NN
        if self.use_nn:
            if self.correction_mode == "offline":
                input = x0[self.nn_input_idx]
                nn_corr_0 = self.nn_model.predict(input)
                self.acados_mpc_solver.set(0, 'p', nn_corr_0)
                if nn_corr is not None:
                    for j in range(1, self.N):
                        self.acados_mpc_solver.set(j, 'p', nn_corr[j])
            elif self.correction_mode == "online":
                input = x0[self.nn_input_idx]
                if 'u' in self.input_features:
                    param = np.array((input, u0, 1))
                else:
                    param = np.array((input, 1))
                self.acados_mpc_solver.set(0, 'p', param)
                param = np.zeros_like(param)
                for j in range(1, self.N):
                    self.acados_mpc_solver.set(j, 'p', param)

        # Solve MPC
        status = self.acados_mpc_solver.solve()
        
        # Get u and projected states
        for i in range(self.N):
            self.u_opt[i, :] = self.acados_mpc_solver.get(i, "u")
            self.x_opt[i, :] = self.acados_mpc_solver.get(i, "x")
        self.x_opt[self.N, :] = self.acados_mpc_solver.get(self.N, "x")

        # Get computation time
        self.opt_dt = self.acados_mpc_solver.get_stats('time_tot') 

        return status
    
    def get_controls(self):
        return self.x_opt, self.u_opt
    
    def get_opt_dt(self):
        return self.opt_dt
