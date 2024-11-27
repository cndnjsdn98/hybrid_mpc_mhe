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
from acados_template import AcadosOcp, AcadosOcpSolver
import rospy
from scipy.linalg import block_diag

from src.quad_opt.quad import custom_quad_param_loader
from src.quad_opt.quad_optimizer import QuadOptimizer

class QuadOptimizerMHE(QuadOptimizer):
    def __init__(self, quad, t_mhe=0.5, n_mhe=50, mhe_type="kinematic",
                 q_mhe=None, q0_factor=None, r_mhe=None, 
                 model_name="quad_3d_acados_mhe",
                 mhe_with_gp=False, y_features=[],
                 change_mass=0,
                 compile_acados=True):
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
        super().__init__(quad, t_mhe=t_mhe, n_mhe=n_mhe, mhe_type=mhe_type, 
                         mhe_with_gp=mhe_with_gp)
        self.mhe_type = mhe_type 
        self.change_mass = change_mass
        self.mhe_with_gp = mhe_with_gp

        # Weighted squared error loss function q = (p_xyz, q_wxyz, v_xyz, r_xyz), r = (u1, u2, u3, u4)
        if q_mhe is None:
            # State noise std
            # System Noise
            w_p = np.ones((1,3)) * 0.004
            w_q = np.ones((1,3)) * 0.01
            w_v = np.ones((1,3)) * 0.005            # w_v = np.ones((1,3)) * 1
            w_r = np.ones((1,3)) * 0.5
            w_d = np.ones((1,3)) * 0.00001 # 0.0000001
            w_a = np.ones((1,3)) * 0.05
            w_m = np.ones((1,1)) * 0.0001
            if mhe_type == "kinematic":
                q_mhe = 1/np.squeeze(np.hstack((w_p, w_q, w_v, w_r, w_a)))
            elif mhe_type == "dynamic":
                if not self.mhe_with_gp:
                    if change_mass != 0:
                        q_mhe = 1/np.squeeze(np.hstack((w_p, w_q, w_v, w_r, w_m)))
                    else:
                        q_mhe = 1/np.squeeze(np.hstack((w_p, w_q, w_v, w_r)))
                else:
                    if change_mass != 0:
                        q_mhe = 1/np.squeeze(np.hstack((w_p, w_q, w_v, w_r, w_d, w_m)))
                    else:
                        q_mhe = 1/np.squeeze(np.hstack((w_p, w_q, w_v, w_r, w_d)))
        if r_mhe is None:
            # Measurement noise std
            # Measurement Noise
            v_p = np.ones((1,3)) * 0.002                 # Position (vicon)
            v_r = np.ones((1,3)) * 1e-06    # Angular Velocity
            v_a = np.ones((1,3)) * 1e-05                # Acceleration
            v_d = np.ones((1,3)) * 0.0001            # Disturbance
            # Inverse covariance
            if mhe_type == "dynamic" and self.mhe_with_gp:
                r_mhe = 1/np.squeeze(np.hstack((v_p, v_r, v_d))) 
            elif mhe_type == "dynamic" and not self.mhe_with_gp:
                r_mhe = 1/np.squeeze(np.hstack((v_p, v_r)))
            else:
                r_mhe = 1/np.squeeze(np.hstack((v_p, v_r, v_a))) 

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

        self.T_mhe = t_mhe  # Time horizon for MHE
        self.N_mhe = n_mhe  # number of nodes within estimation horizon

        if self.mhe_with_gp:
            self.B_x = np.zeros((self.state_dim, len(y_features)))
            for i, idx in enumerate(y_features):
                self.B_x[idx, i] = 1

            if self.n_param > 0:
                self.B_x = np.append(self.B_x, np.zeros((self.n_param, self.B_x.shape[1])), axis=0)
            self.q_hist = np.zeros((0, 4))
            self.mhe_model_error = None

        # Nominal model equations symbolic function (no GP)
        self.quad_xdot_nominal = self.quad_dynamics(mhe=True, mhe_type=mhe_type)
        # Build full model for MHE. Will have 13 variables. self.dyn_x contains the symbolic variable that
        # should be used to evaluate the dynamics function. It corresponds to self.x if there are no GP's, or
        # self.x_with_gp otherwise      
        acados_models_mhe, nominal_with_gp_mhe = self.acados_setup_model(
            self.quad_xdot_nominal(x=self.x, u=self.u, w=self.w)['x_dot'], model_name, mhe=True)
        
        # Convert dynamics variables to functions of the state and input vectors
        for dyn_model_idx in nominal_with_gp_mhe.keys():
            dyn = nominal_with_gp_mhe[dyn_model_idx]
            self.quad_xdot_mhe[dyn_model_idx] = cs.Function('x_dot', [self.x, self.u, self.w], [dyn], ['x', 'u', 'w'], ['x_dot'])        

        # ### Setup and compile Acados OCP solvers ### #
        if compile_acados:
            for key_model in acados_models_mhe.values():
                ocp_mhe, nyx, nx, nu = self.create_mhe_solver(key_model, q_mhe, q0_factor, r_mhe)
                self.nyx = nyx
                self.nx = nx
                self.nu = nu

                # Compile acados OCP solver if necessary
                json_file_mhe = os.path.join(self.acados_models_dir, "mhe", key_model.name + '.json')
                self.acados_mhe_solver = AcadosOcpSolver(ocp_mhe, json_file=json_file_mhe)

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
        if self.n_param > 0:
            q_cost = q_cost[:-self.n_param]

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
        ocp_mhe.solver_options.tf = self.T_mhe
        ocp_mhe.dims.N = self.N_mhe
        
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

        # Quadrotor Mass Estimation limits
        if isinstance(self.k_m, cs.MX):
            ocp_mhe.constraints.lbx_0 = self.param_lbx
            ocp_mhe.constraints.ubx_0 = self.param_ubx
            ocp_mhe.constraints.idxbx_0 = np.array([self.state_dim])
        
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
        
        return ocp_mhe, nyx, nx, nu

def main():
    rospy.init_node("acados_compiler_mhe", anonymous=True) 
    ns = rospy.get_namespace()

    compile = rospy.get_param("/compile", default=True)
    
    # Load Quad Instance
    quad_name = rospy.get_param(ns + 'quad_name', default=None)
    assert quad_name != None
    quad = custom_quad_param_loader(quad_name)
    rospy.set_param(ns + quad_name + '/hover_thrust', quad.hover_thrust)
   
    if compile:
        rospy.loginfo("Compiling MHE Acados model...")
        # Load MHE parameters
        n_mhe = rospy.get_param(ns + 'n_mhe', default=50)
        t_mhe = rospy.get_param(ns + 't_mhe', default=0.5)
        mhe_type = rospy.get_param(ns + 'mhe_type', default="kinematic")
        with_gp = rospy.get_param(ns + 'with_gp', default=False)
        change_mass = rospy.get_param(ns + 'change_mass', default=0)
        y_features = rospy.get_param(ns + 'y_features', default=[7, 8, 9])

        # System Noise
        w_p = np.ones((1,3)) * rospy.get_param(ns + 'w_p', default=0.004)
        w_q = np.ones((1,3)) * rospy.get_param(ns + 'w_q', default=0.01)
        w_v = np.ones((1,3)) * rospy.get_param(ns + 'w_v', default=0.005)
        w_r = np.ones((1,3)) * rospy.get_param(ns + 'w_r', default=0.5)
        w_a = np.ones((1,3)) * rospy.get_param(ns + 'w_a', default=0.05)

        w_d = np.ones((1,3)) * rospy.get_param(ns + 'w_d', default=0.00001)
        w_m = np.ones((1,3)) * rospy.get_param(ns + 'w_m', default=0.0001)

        # Measurement Noise
        v_p = np.ones((1,3)) * rospy.get_param(ns + 'v_p', default=0.002)
        v_r = np.ones((1,3)) * rospy.get_param(ns + 'v_r', default=1e-6)
        v_a = np.ones((1,3)) * rospy.get_param(ns + 'v_a', default=1e-5)
        v_d = np.ones((1,3)) * rospy.get_param(ns + 'v_d', default=0.0001)

        # System Weights
        if mhe_type == "kinematic":
            q_mhe = 1/np.squeeze(np.hstack((w_p, w_q, w_v, w_r, w_a)))
        elif mhe_type == "dynamic" and not with_gp:
            if change_mass != 0:
                q_mhe = 1/np.squeeze(np.hstack((w_p, w_q, w_v, w_r, w_m)))
            else:
                q_mhe = 1/np.squeeze(np.hstack((w_p, w_q, w_v, w_r)))
        elif mhe_type == "dynamic" and with_gp:
            if change_mass != 0:
                q_mhe = 1/np.squeeze(np.hstack((w_p, w_q, w_v, w_r, w_d, w_m)))
            else:
                q_mhe = 1/np.squeeze(np.hstack((w_p, w_q, w_v, w_r, w_d)))
        q0_factor = 1 # arrival cost factor
        if mhe_type == "kinematic":
            r_mhe = 1/np.squeeze(np.hstack((v_p, v_r, v_a))**2) 
        elif mhe_type == "dynamic" and with_gp:
            r_mhe = 1/np.squeeze(np.hstack((v_p, v_r, v_d))**2) 
        elif mhe_type == "dynamic" and not with_gp:
            r_mhe = 1/np.squeeze(np.hstack((v_p, v_r))**2)

        # Compile Acados Model
        quad_opt = QuadOptimizerMHE(quad, t_mhe=t_mhe, n_mhe=n_mhe, mhe_type=mhe_type,
                                    q_mhe=q_mhe, q0_factor=q0_factor, r_mhe=r_mhe,
                                    model_name=quad_name, 
                                    mhe_with_gp=with_gp, y_features=y_features,
                                    change_mass=change_mass)
        rospy.loginfo("MHE Acados model Compiled Successfully...")
        
    return

def init_compile():
    quad_name = "clark"
    quad = custom_quad_param_loader(quad_name)
    quad_opt = QuadOptimizerMHE(quad, mhe_type='dynamic', mhe_with_gp=False)

if __name__ == "__main__":
    main()
    # init_compile()
