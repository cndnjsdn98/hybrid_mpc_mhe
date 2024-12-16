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
from scipy.linalg import block_diag
import l4casadi as l4c

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from src.utils.utils import v_dot_q
from src.utils.DirectoryConfig import DirectoryConfig as DirConfig
from src.quad.quad import Quadrotor

class QuadOptimizerMHE:
    def __init__(self, quad:Quadrotor, t_mhe=0.5, n_mhe=50, 
                 mhe_type="k",
                 q_mhe=None, q0_factor=None, r_mhe=None, 
                 use_nn=False, nn_params={}):
        """
        :param quad: quadrotor object.
        :type quad: Quadrotor3D
        :param t_mhe: time horizon for MHE optimization.
        :param n_mhe: number of optimization nodes in time horizon for MHE.
        :param mhe_type: define model type to be used for MHE [kinematic, or dynamic].
        :param q_mhe: diagonal of Qe matrix for model mismatch cost of MHE cost function. Must be a numpy array of length 12.
        :param q0_factor: integer to set the arrival cost of MHE as a factor/multiple of q_mhe.
        :param r_mhe: diagonal of Re matrix for measurement mismatch cost of MHE cost function. Must be a numpy array of length ___.
        :param B_x: dictionary of matrices that maps the outputs of the gp regressors to the state space.
        :param model_name: Acados model name.
        :param mhe_with_gp: GPyEnsemble instance to be utilized in MHE
        :param change_mass: Value of varying payload mass 
        # TODO: Change the chage_mass to be boolean
        """
        # MHE Params
        self.T = t_mhe
        self.N = n_mhe
        self.mhe_type = mhe_type 
        # Quad
        self.quad = quad

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

        # Init State est variables
        self.x_est = np.zeros((1, self.state_dim))
        self.x0_bar = None
        self.opt_dt = 0
        
        self.acados_models_dir = DirConfig.ACADOS_MODEL_DIR

        # Nominal model equations symbolic function (no NN)
        self.quad_xdot_nominal = self.quad_dynamics()

        # Build full model for MHE
        acados_model, dynamics = self.acados_setup_model(
            self.quad_xdot_nominal(x=self.x, u=self.u, w=self.w)['x_dot'])

        # Weighted squared error loss function q = (p_xyz, q_wxyz, v_xyz, r_xyz), r = (u1, u2, u3, u4)
        if q0_factor is None:
            q0_factor = 1
        self.x0_bar = None
        print("\n###########################################################################################")
        print("Q_estimate         = ", q_mhe)
        print("Q_arrival_cost     = ", q_mhe * q0_factor)
        print("R_estimate         = ", r_mhe)
        print("###########################################################################################\n")
        # Add one more weight to the rotation (use quaternion norm weighting in acados)
        q_mhe = np.concatenate((q_mhe[:3], np.mean(q_mhe[3:6])[np.newaxis], q_mhe[3:]))
        assert q_mhe is not None or r_mhe is not None

        # ### Setup and compile Acados OCP solvers ### #
        ocp_mhe = self.create_mhe_solver(acados_model, q_mhe, q0_factor, r_mhe)
        # NOTE: self.nu = self.state_dim

        # Compile acados OCP solver if necessary
        json_file_mhe = os.path.join(self.acados_models_dir, "mhe", acados_model.name + '.json')
        self.acados_mhe_solver = AcadosOcpSolver(ocp_mhe, json_file=json_file_mhe)

    def init_cs_vectors(self):
        # Declare model variables
        self.p = cs.MX.sym('p', 3)  # position
        self.q = cs.MX.sym('q', 4)  # angle quaternion (wxyz)
        self.v = cs.MX.sym('v', 3)  # velocity
        self.r = cs.MX.sym('r', 3)  # angle rate
        # Full state vector (13-dimensional)
        self.x = cs.vertcat(self.p, self.q, self.v, self.r)

        # Full differential state vector 
        self.p_dot = cs.MX.sym('p_dot', 3)
        self.q_dot = cs.MX.sym('q_dot', 4)
        self.v_dot_w = cs.MX.sym('v_dot_w', 3)
        self.r_dot = cs.MX.sym('r_dot', 3)
        self.x_dot = cs.vertcat(self.p_dot, self.q_dot, self.v_dot_w, self.r_dot)

        # Declare model noise variables for MHE
        self.w_p = cs.MX.sym('w_p', 3)  # position
        self.w_q = cs.MX.sym('w_q', 4)  # angle quaternion (wxyz)
        self.w_v = cs.MX.sym('w_v', 3)  # velocity
        self.w_r = cs.MX.sym('w_r', 3)  # angle rate
        self.w_a = cs.MX.sym('w_a', 3)  # linear acceleration
        self.w = cs.vertcat(self.w_p, self.w_q, self.w_v, self.w_r)

        # Model adjustments for MHE
        if self.mhe_type == "k":
            self.a = cs.MX.sym('a', 3)  # IMU acc measurement
            self.a_dot = cs.MX.sym('a_dot', 3)
            # Full state vector (16-dimensional)
            self.x = cs.vertcat(self.x, self.a)
            self.x_dot = cs.vertcat(self.x_dot, self.a_dot)
            self.state_dim = 16
            # Full state noise vector (16-dimensional)
            self.w = cs.vertcat(self.w, self.w_a)
            # Update input state vector
            self.u = cs.vertcat()
            # Full measurement state vector
            self.y = cs.vertcat(self.p, self.r, self.a)
        elif self.mhe_type == "d":
            # Full state dim remains same
            self.state_dim = 13
            # Control input vector
            u1 = cs.MX.sym('u1')
            u2 = cs.MX.sym('u2')
            u3 = cs.MX.sym('u3')
            u4 = cs.MX.sym('u4')
            self.u = cs.vertcat(u1, u2, u3, u4)
            # Full measurement state vector
            self.y = cs.vertcat(self.p, self.r)

        self.param = np.array([])

    def quad_dynamics(self):
        """
        Symbolic dynamics of the 3D quadrotor model to be used for MHE.
        The dynamics is altered slightly from traditional model to be 
        modelled for state estimation with respective measurements.
        """
        p_dyn = self.quad.p_dynamics(self.x)
        q_dyn = self.quad.q_dynamics(self.x)
        if self.mhe_type == "k":
            g = cs.vertcat(0.0, 0.0, 9.81)
            v_dyn = v_dot_q(self.a, self.q) - g
            a_dyn = cs.vertcat(0, 0, 0)
        elif self.mhe_type == "d":
            v_dyn = self.quad.v_dynamics(self.x, self.u)
            a_dyn = cs.vertcat()
        r_dyn = cs.vertcat(0, 0, 0)
        x_dot = cs.vertcat(p_dyn, q_dyn, v_dyn, r_dyn, a_dyn) + self.w
        return cs.Function('x_dot_mhe', [self.x, self.u, self.w], [x_dot], ['x', 'u', 'w'], ['x_dot'])

    def acados_setup_model(self, nominal):
        """
        Builds an Acados symbolic models using CasADi expressions.
        :param quad_name: Name of the quadrotor
        :param nominal: CasADi symbolic nominal model of the quadrotor: f(self.x, self.u) = x_dot, dimensions 13x1.
        :param mhe: Boolean variable. True if model is for MHE, and False if model is for MPC. 

        """
        dynamics_ = nominal
        x_dot_ = self.x_dot
        x_ = self.x
        u_ = self.u
        w_ = self.w
        param_ = []

        acados_model = AcadosModel()
        acados_model.f_expl_expr = dynamics_
        acados_model.f_impl_expr = x_dot_ - dynamics_
        acados_model.x = x_
        acados_model.xdot = x_dot_
        acados_model.u = w_
        acados_model.p = cs.vertcat(u_, param_)
        acados_model.name = "mhe"
    
        return acados_model, dynamics_
    
    def create_mhe_solver(self, model, q_cost, q0_factor, r_cost):
        """
        Creates OCP objects to formulate the MPC optimization
        :param model: Acados model of the system
        :type model: cs.MX 
        :param q_cost: diagonal of Q matrix for model mismatch cost of MHE cost function. Must be a numpy array of length 12.
        :param q0_factor: integer to set the arrival cost of MHE as a factor/multiple of q_mhe.
        :param r_cost: diagonal of R matrix for measurement mismatch cost of MHE cost function. 
        Must be a numpy array of length equal to number of measurements.
        """
        # Set Arrival Cost as a factor of q_cost
        q0_cost = q_cost * q0_factor

        # Number of states and Inputs of the model
        # make acceleration as error
        x = model.x
        u = model.u
        yx = self.y

        nx = x.size()[0]
        nyx = yx.size()[0]
        nu = u.size()[0]

        # Total number of elements in the cost functions
        ny_0 = nyx + nu + nx # h(x), w and arrival cost
        ny_e = 0
        ny = nyx + nu # h(x) and w
        n_param = model.p.size()[0] if isinstance(model.p, cs.MX) else 0
        
        # Set up Cost Matrices
        Q = np.diag(q_cost)
        R = np.diag(r_cost)
        Q0 = np.diag(q0_cost)

        acados_source_path = os.environ['ACADOS_SOURCE_DIR']
        sys.path.insert(0, '../common')

        # Create OCP object to formulate the MPC optimization
        ocp_mhe = AcadosOcp()
        ocp_mhe.acados_include_path = acados_source_path + '/include'
        ocp_mhe.acados_lib_path = acados_source_path + '/lib'
        ocp_mhe.model = model
        
        # Set Prediction horizon
        ocp_mhe.solver_options.tf = self.T
        ocp_mhe.dims.N = self.N
        
        # Cost of Initial stage    
        ocp_mhe.cost.cost_type_0 = 'NONLINEAR_LS'
        ocp_mhe.cost.W_0 = block_diag(R, Q, Q0)
        ocp_mhe.model.cost_y_expr_0 = cs.vertcat(yx, u, x)
        ocp_mhe.cost.yref_0 = np.zeros((ny_0,))

 
        # Cost of Intermediate stages
        ocp_mhe.cost.cost_type = 'NONLINEAR_LS'         
        ocp_mhe.cost.W = block_diag(R, Q)
        ocp_mhe.model.cost_y_expr = cs.vertcat(yx, u)
        ocp_mhe.cost.yref = np.zeros((ny,))
        

        # set y_ref terminal stage which doesn't exist so 0s
        ocp_mhe.cost.cost_type_e = "LINEAR_LS"
        ocp_mhe.cost.yref_e = np.zeros((ny_e, ))
        ocp_mhe.cost.W_e = np.zeros((0,0))
        
        # Initialize parameters
        ocp_mhe.parameter_values = np.zeros((n_param, ))
       
        # # Quaternion normalization Constraint
        # eps = 1e-6
        # ocp_mhe.constraints.nh = 1
        # ocp_mhe.constraints.lh = np.array([1 - eps])
        # ocp_mhe.constraints.uh = np.array([1 + eps])
        
        # Solver options
        ocp_mhe.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp_mhe.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp_mhe.solver_options.integrator_type = 'ERK'
        ocp_mhe.solver_options.print_level = 0
        ocp_mhe.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp_mhe.solver_options.qp_solver_warm_start = 1 # Warm Start

        # Path to where code will be exported
        ocp_mhe.code_export_directory = os.path.join(self.acados_models_dir, "mhe")
        
        return ocp_mhe

    def set_history_trajectory(self, y_history, u_history):
        """
        Sets the history of trajectory and pre-computes the cost equations for each point in the reference sequence.
        Computes the state uncertainty of the reference states with the trained GP model to set state constraints.
        :param y_history: Nx13-dimensional list of history of state measurements. It is passed in the
        form of a 4-length list, where the first element is a Nx3 numpy array referring to the position measurements, the
        second is a Nx4 array referring to the quaternion, two more Nx3 arrays for the velocity and body rate measurements.
        """
        # Initialise arrival cost
        if self.x0_bar is None:
            p = y_history[0, :3]
            q = [1, 0, 0, 0]
            v = [0, 0, 0]
            r = [0, 0, 0]
            if self.mhe_type == "k":
                a = [0, 0, 9.81]
                self.x0_bar = np.append(p, np.array(q + v + r + a))
            elif self.mhe_type == "d":
                self.x0_bar = np.append(p, np.array(q + v + r))           

        
        # Set motor thrusts for dynamic MHE
        elif self.mhe_type == "d":
            for j in range(self.N):
                u = np.array(cs.vertcat(u_history[j, :]))
                self.acados_mhe_solver.set(j, 'p', u)
        # Initialise reference for mhe
        # First reference is set separately to set the arrival cost
        # Set history of state measurements
        yref_0 = np.array(cs.vertcat(y_history[0, :], np.zeros((self.state_dim)), self.x0_bar))
        self.acados_mhe_solver.set(0, "yref", yref_0)
        for j in range(1, self.N):
            yref = np.array(cs.vertcat(y_history[j, :], np.zeros((self.state_dim))))
            self.acados_mhe_solver.set(j, "yref", yref)

    def solve_mhe(self):
        """
        Optimizes state measurements and model dynamics to estimate the system states, that minimizes
        the quadratic cost function and respects the constraints of the system
        """
        # Solve MHE
        if self.x0_bar is not None:
            status = self.acados_mhe_solver.solve()
            # Get estimated state
            self.x_est = self.acados_mhe_solver.get(self.N, "x") 
            self.x0_bar = self.acados_mhe_solver.get(1, "x")

        self.qp_iter = self.acados_mhe_solver.get_stats('qp_iter')
        self.sqp_iter = self.acados_mhe_solver.get_stats('sqp_iter')
        self.opt_dt = self.acados_mhe_solver.get_stats('time_tot')[0]

        return status

    def get_state_est(self):
        return self.x_est[:self.state_dim]

    def get_param_est(self):
        return self.x_est[self.state_dim:]
    
    def get_opt_dt(self):
        return self.opt_dt