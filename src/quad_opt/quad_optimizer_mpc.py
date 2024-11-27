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

from src.quad_opt.quad import custom_quad_param_loader
from acados_template import AcadosOcp, AcadosOcpSolver
from src.quad_opt.quad_optimizer import QuadOptimizer

from src.gp.GPyModelWrapper import GPyModelWrapper
class QuadOptimizerMPC(QuadOptimizer):
    def __init__(self, quad, t_mpc=1, n_mpc=10, 
                 q_mpc=None, qt_factor=None,r_mpc=None, 
                 model_name="quad_3d_acados_mpc", solver_options=None, 
                 mpc_with_gp=False, y_features=[],
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
        super().__init__(quad, t_mpc=t_mpc, n_mpc=n_mpc,
                         mpc_with_gp=mpc_with_gp)

        self.mpc_with_gp = mpc_with_gp
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

        # If using gp for MPC Set up Necessary variables
        if self.mpc_with_gp:
            self.B_x = np.zeros((self.state_dim, len(y_features)))
            for i, idx in enumerate(y_features):
                self.B_x[idx, i] = 1
            self.lock = threading.Lock()
            # self.mpc_gp_params = cs.vertcat(self.gp_x, self.d, self.trigger_var)
            self.mpc_gp_params = cs.vertcat(self.d)

        self.x_opt_acados = np.ndarray((self.N_mpc + 1, 13))
        self.w_opt_acados = np.ndarray((self.N_mpc, 4))
            
        # Nominal model equations symbolic function (no GP)
        self.quad_xdot_nominal = self.quad_dynamics(mhe=False, prop=False)

        # Build full model for MPC. Will have 13 variables. self.dyn_x contains the symbolic variable that
        # should be used to evaluate the dynamics function. It corresponds to self.x if there are no GP's, or
        # self.x_with_gp otherwise
        acados_models, nominal_with_gp = self.acados_setup_model(
            self.quad_xdot_nominal(x=self.x, u=self.u, p=self.mpc_param)['x_dot'], model_name)
        
        # Convert dynamics variables to functions of the state and input vectors
        for dyn_model_idx in nominal_with_gp.keys():
            dyn = nominal_with_gp[dyn_model_idx]
            if self.mpc_with_gp:
                self.quad_xdot[dyn_model_idx] = cs.Function('x_dot', [self.x, self.u, cs.vertcat(self.mpc_param, self.mpc_gp_params)], [dyn], ['x', 'u', 'p'], ['x_dot'])
            else:
                self.quad_xdot[dyn_model_idx] = cs.Function('x_dot', [self.x, self.u, self.mpc_param], [dyn], ['x', 'u', 'p'], ['x_dot'])

        # ### Setup and compile Acados OCP solvers ### #
        if compile_acados:
            for key_model in acados_models.values():
                ocp_mpc = self.create_mpc_solver(key_model, q_mpc, qt_factor, r_mpc, solver_options)

                json_file_mpc = os.path.join(self.acados_models_dir, "mpc", key_model.name + '.json')
                self.acados_mpc_solver = AcadosOcpSolver(ocp_mpc, json_file=json_file_mpc)

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
   
def main():
    rospy.init_node("acados_compiler_mpc", anonymous=True)
    ns = rospy.get_namespace()
    
    # Load MHE parameters
    n_mpc = rospy.get_param(ns + 'n_mpc', default=10)
    t_mpc = rospy.get_param(ns + 't_mpc', default=1)
    quad_name = rospy.get_param(ns + 'quad_name', default=None)
    assert quad_name != None
    with_gp = rospy.get_param(ns + 'with_gp', default=False)
    change_mass = rospy.get_param(ns + 'change_mass', default=0)
    y_features = rospy.get_param(ns + 'y_features', default=[7, 8, 9])

    # MPC Costs
    q_p = np.ones((1,3)) * rospy.get_param(ns + 'q_p', default=35)
    q_q = np.ones((1,3)) * rospy.get_param(ns + 'q_q', default=25)
    q_v = np.ones((1,3)) * rospy.get_param(ns + 'q_v', default=10)
    q_r = np.ones((1,3)) * rospy.get_param(ns + 'q_r', default=10)

    qt_factor = rospy.get_param(ns + "qt_factor", default=1)

    q_mpc = np.squeeze(np.hstack((q_p, q_q, q_v, q_r)))
    r_mpc = np.array([1.0, 1.0, 1.0, 1.0]) * rospy.get_param(ns + "r_mpc", default=0.1)

    # Load Quad Instance
    quad = custom_quad_param_loader(quad_name)
    # Parameters needed for gz control commands
    rospy.set_param('quad_max_thrust', quad.max_thrust)
    rospy.set_param('quad_mass', quad.mass)

    # Compile Acados Model
    quad_opt = QuadOptimizerMPC(quad, t_mpc=t_mpc, n_mpc=n_mpc,
                                q_mpc=q_mpc, qt_factor=qt_factor, r_mpc=r_mpc, 
                                model_name=quad_name,
                                mpc_with_gp=with_gp, y_features=y_features)
    return

def init_compile():
    quad_name = "clark"
    quad = custom_quad_param_loader(quad_name)
    quad_opt = QuadOptimizerMPC(quad)

if __name__ == "__main__":
    main()
    # init_compile()
