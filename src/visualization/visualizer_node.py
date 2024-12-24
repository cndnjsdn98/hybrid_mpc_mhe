#!/usr/bin/env python

import os
import rospy
import threading
import numpy as np
from tqdm import tqdm
import casadi as cs
import pickle 
import json
import time

from std_msgs.msg import Bool
from mav_msgs.msg import Actuators
from mavros_msgs.msg import AttitudeTarget
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped, TwistStamped
from quadrotor_msgs.msg import ControlCommand
from hybrid_mpc_mhe.msg import ReferenceTrajectory, QuadSensor

from src.quad.quad import Quadrotor
from src.utils.utils import v_dot_q, quaternion_inverse, safe_mkdir_recursive, vw_to_vb, rmse, q_rmse
from src.visualization.visualization import trajectory_tracking_results, state_estimation_results
from src.utils.DirectoryConfig import DirectoryConfig as DirConfig

class VisualizerWrapper:
    def __init__(self):
        # Get Params
        self.quad_name = None
        while self.quad_name is None:
            self.quad_name = rospy.get_param("/quad_name", default=None)
            rospy.sleep(1)
            
        self.env = rospy.get_param("/environment", default="arena")
        
        self.quad = Quadrotor(self.quad_name, prop=True)

        self.t_mpc = rospy.get_param("/mpc/t_mpc", default=1)
        self.n_mpc = rospy.get_param("/mpc/n_mpc", default=10)
        self.use_groundtruth = rospy.get_param("/mpc/use_groundtruth", default=True)
        self.mpc_use_nn = rospy.get_param("/mpc/use_nn", default=False)
        self.mpc_model_name = rospy.get_param("/mpc/model_name", default=None)
        self.mpc_model_type = rospy.get_param("/mpc/model_type", default=None)
        self.mpc_correction_mode = rospy.get_param("/mpc/correction_mode", default=None)

        self.t_mhe = rospy.get_param("/mhe/t_mhe", default=0.5)
        self.n_mhe = rospy.get_param("/mhe/n_mhe", default=50)
        mhe_type = rospy.get_param("/mhe/mhe_type", default=None)
        if mhe_type == "kinematic":
            self.mhe_type = "k"
        elif mhe_type == "dynamic":
            self.mhe_type = "d"
        self.mhe_use_nn = rospy.get_param("/mhe/use_nn", default=False)
        self.mhe_model_name = rospy.get_param("/mhe/model_name", default=None)
        self.mhe_model_type = rospy.get_param("/mhe/model_type", default=None)
        self.mhe_correction_mode = rospy.get_param("/mhe/correction_mode", default=None)
        # Override since K-MHE never uses NN
        if self.mhe_type == "k":
            self.mhe_use_nn = False

        self.results_dir = DirConfig.FLIGHT_DATA_DIR

        assert self.results_dir is not None
        self.mpc_meta = {
            'quad_name': self.quad_name,
            'env': self.env,
            't_mpc': self.t_mpc,
            'n_mpc': self.n_mpc,
            'gt': self.use_groundtruth,
            'use_nn': self.mpc_use_nn,
            'model_type': self.mpc_model_type,
            'model_name': self.mpc_model_name,
            'correction_mode': self.mpc_correction_mode,
        }
        self.mhe_meta = {
            'quad_name': self.quad_name,
            'env': self.env,
            't_mhe': self.t_mhe,
            'n_mhe': self.n_mhe,
            'mhe_type': self.mhe_type,
            'use_nn': self.mhe_use_nn,
            'model_type': self.mhe_model_type,
            'model_name': self.mhe_model_name,
            'correction_mode': self.mhe_correction_mode,
        }
        # Create MPC Directory
        if self.mpc_use_nn:
            use_nn_str = "_%s_%s"%(self.mpc_model_type,
                                    self.mpc_correction_mode)
        else:
            use_nn_str = ""
        self.mpc_dataset_name = "%s_mpc_%s%s%s"%(self.env, 
                                                 "gt_" if self.use_groundtruth else "", 
                                                 self.quad_name,
                                                 use_nn_str)
        self.mpc_dir = os.path.join(self.results_dir, self.mpc_dataset_name)
        safe_mkdir_recursive(self.mpc_dir)
        # Create MHE Directory if MHE Type is given
        # else check every second
        if self.mhe_type is not None:
            if self.mhe_use_nn:
                use_nn_str = "_%s_%s"%(self.mhe_model_type,
                                        self.mhe_correction_mode)
            else:
                use_nn_str = ""
            self.mhe_dataset_name = "%s_%smhe_%s%s"%(self.env, 
                                                   self.mhe_type, 
                                                   self.quad_name,
                                                   use_nn_str)
            self.mhe_dir = os.path.join(self.results_dir, self.mhe_dataset_name)
            safe_mkdir_recursive(self.mhe_dir)

        # Check every 1 second if MPC/MHE parameters have changed
        self.timer_use_gt = rospy.Timer(rospy.Duration(1), self.check_use_gt)
        self.timer_mpc_use_nn = rospy.Timer(rospy.Duration(1), self.check_mpc_use_nn)
        self.timer_mhe_type = rospy.Timer(rospy.Duration(1), self.check_mhe_type)
        self.timer_mhe_use_nn = rospy.Timer(rospy.Duration(1), self.check_mhe_use_nn)

        self.record = False

        # Initialize vectors to store Reference Trajectory
        self.seq_len = None
        self.ref_traj_name = None
        self.ref_v = None
        self.x_ref = None
        self.t_ref = None
        self.u_ref = None
        # Initialize vectors to store Tracking and Estimation Results
        self.t_act = np.zeros((0, 1))
        self.t_imu = np.zeros((0, 1))
        self.t_sensor_measurement = np.zeros((0, 1))
        self.t_est = np.zeros((0, 1))
        self.t_acc_est = np.zeros((0, 1))
        self.x_act = np.zeros((0, 13))
        self.x_est = np.zeros((0, 13))
        self.y = np.zeros((0, 9))
        self.y_noisy = np.zeros((0, 9))
        self.accel_est = np.zeros((0, 3))
        self.motor_thrusts = np.zeros((0, 4))
        self.w_control = np.zeros((0, 3))
        self.collective_thrusts = np.zeros((0, 1))

        # System States
        self.p_act = None
        self.q_act = None
        self.v_act = None
        self.w_act = None
        self.p_meas = None
        self.w_meas = None
        self.a_meas = None

        # Subscriber topic names
        odom_topic = rospy.get_param("/odom_topic", default = "/" + self.quad_name + "/ground_truth/odometry")
        imu_topic = rospy.get_param("/imu_topic", default = "/hummingbird/ground_truth/imu") 
        state_est_topic = rospy.get_param("/state_est_topic", default = "/" + self.quad_name + "/state_est")
        acceleration_est_topic = rospy.get_param("/acceleration_est_topic", default = "/" + self.quad_name + "/acceleration_est")
        motor_thrust_topic = rospy.get_param("/motor_thrust_topic", default = "/" + self.quad_name + "/motor_thrust")
        ref_topic = rospy.get_param("/ref_topic", default = "/reference")
        control_topic = rospy.get_param("/control_topic", default = "/" + self.quad_name + "/autopilot/control_command_input")
        record_topic = rospy.get_param("/record_topic", default = "/" + self.quad_name + "/record")
        sensor_measurement_topic = rospy.get_param("/sensor_measurement_topic", default="/" + self.quad_name + "/sensor_measurement") 

        # Subscribers
        self.imu_sub = rospy.Subscriber(imu_topic, Imu, self.imu_callback, queue_size=1, tcp_nodelay=False)
        self.motor_thrust_sub = rospy.Subscriber(motor_thrust_topic, Actuators, self.motor_thrust_callback, queue_size=1, tcp_nodelay=False)
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odom_callback, queue_size=1, tcp_nodelay=False)
        self.state_est_sub = rospy.Subscriber(state_est_topic, Odometry, self.state_est_callback, queue_size=1, tcp_nodelay=False)
        self.acceleration_est_sub = rospy.Subscriber(acceleration_est_topic, Imu, self.acceleration_est_callback, queue_size=10, tcp_nodelay=False)
        self.ref_sub = rospy.Subscriber(ref_topic, ReferenceTrajectory, self.ref_callback, queue_size=1, tcp_nodelay=False)
        if self.env == "gazebo":
            self.control_sub = rospy.Subscriber(control_topic, ControlCommand, self.control_gz_callback, queue_size=1, tcp_nodelay=True)
        elif self.env == "arena":
            self.control_sub = rospy.Subscriber(control_topic, AttitudeTarget, self.control_callback, queue_size=1, tcp_nodelay=False)
        self.record_sub = rospy.Subscriber(record_topic, Bool, self.record_callback, queue_size=1, tcp_nodelay=False)
        self.sensor_measurement_sub = rospy.Subscriber(sensor_measurement_topic, QuadSensor, self.sensor_measurement_callback, queue_size=1, tcp_nodelay=True)
        rospy.loginfo("Visualizer on standby listening...")
        rospy.loginfo("MPC w/%s NN"% ("OUT" if not self.mpc_use_nn else ""))
        if self.mhe_type is not None:
            rospy.loginfo("%s-MHE w/%s"% (self.mhe_type.upper(), "OUT NN" if not self.mhe_use_nn else " %s"%self.mhe_model_type))


    def save_recording_data(self):
        # Remove Exceeding data entry if needed    
        while len(self.w_control) < self.seq_len:
            self.w_control = np.append(self.w_control, self.w_control[-1][np.newaxis], axis=0)
        self.w_control = self.w_control[:self.seq_len]
        
        while len(self.x_act) < self.seq_len * 2:
            self.x_act  = np.append(self.x_act, self.x_act[-1][np.newaxis], axis=0)
        self.t_act = self.t_act - self.t_act[0]
        while len(self.t_act) < self.seq_len * 2:
            self.t_act  = np.append(self.t_act, self.t_act[-1])
        self.x_act = self.x_act[:self.seq_len * 2]
        self.t_act = self.t_act[:self.seq_len * 2]
        
        while len(self.motor_thrusts) < self.seq_len * 2:
            self.motor_thrusts = np.append(self.motor_thrusts, self.motor_thrusts[-1][np.newaxis], axis=0)
        self.motor_thrusts = self.motor_thrusts[:self.seq_len * 2]

        if len(self.x_est) > 0:
            while len(self.x_est) < self.seq_len * 2:
                self.x_est = np.append(self.x_est, self.x_est[-1][np.newaxis], axis=0)
            self.t_est = self.t_est - self.t_est[0]
            while len(self.t_est) < self.seq_len * 2:
                self.t_est = np.append(self.t_est, self.t_est[-1])
            self.x_est = self.x_est[:self.seq_len * 2]
            self.t_est = self.t_est[:self.seq_len * 2]
        # Save MPC results
        state_in = np.zeros((self.seq_len, self.x_act.shape[1]))
        state_in_B = np.zeros((self.seq_len, self.x_act.shape[1]))
        state_out = np.zeros((self.seq_len, self.x_act.shape[1]))
        u_in = np.zeros((self.seq_len, 4))
        mpc_error = np.zeros_like(state_in)
        x_pred_traj = np.zeros_like(state_in)
        mpc_t = np.zeros((self.seq_len, 1))
        dt_traj = np.zeros((self.seq_len, 1))
        rospy.loginfo("Filling in MPC dataset and saving...")
        for i in tqdm(range(self.seq_len)): 
            # TODO: Get states in body frame
            ii = i * 2
            x0 = self.x_act[ii]
            x0_B = vw_to_vb(x0)
            xf = self.x_act[ii+1]
            xf_B = vw_to_vb(xf)

            u = self.motor_thrusts[ii]

            dt = self.t_act[ii+1] - self.t_act[ii]

            # Dynamic Model Pred
            x_pred = self.quad.forward_prop(x0, u, t_horizon=dt)
            # x_pred = x_pred[-1, np.newaxis, :]
            x_pred_B = vw_to_vb(x_pred)
            
            # MPC Model error
            x_err = xf_B - x_pred_B

            # Save to array for plots
            mpc_t[i] = self.t_act[ii]
            state_in[i] = x0 #if self.use_groundtruth else self.x_est[ii]
            state_in_B[i] = x0_B
            state_out[i] = xf #if self.use_groundtruth else self.x_est[ii+1]
            u_in[i] = u
            dt_traj[i] = dt
            x_pred_traj[i] = x_pred
            mpc_error[i] = x_err / dt if dt != 0 else 0
        self.x_act = self.x_act[:self.seq_len * 2]
        self.t_act = self.t_act[:self.seq_len * 2]
        mpc_tracking_error = rmse(self.t_ref, self.x_ref[:, :3], self.t_ref, state_in[:, :3])
        
        # Organize arrays to dictionary
        mpc_dict = {
            "t": mpc_t,
            "dt": dt_traj,
            "t_ref": self.t_ref,
            "state_in": state_in,
            "state_out": state_out,
            "x" : self.x_act if self.use_groundtruth else self.x_est,
            "x_act": self.x_act,
            "x_ref": self.x_ref,
            "error": mpc_error,
            "x_pred": x_pred_traj,
            "input_in": u_in,
            "w_control": self.w_control,
            "state_in_Body": state_in_B, 
        }
        self.mpc_meta['rmse'] = mpc_tracking_error
        v_max = np.max(np.linalg.norm(state_in[:, 7:10], axis=1))

        # Save results
        mpc_dir = os.path.join(self.mpc_dir, self.ref_traj_name)
        safe_mkdir_recursive(mpc_dir)
        with open(os.path.join(mpc_dir, "results.pkl"), "wb") as f:
            pickle.dump(mpc_dict, f)
        with open(os.path.join(mpc_dir, 'meta_data.json'), "w") as f:
            json.dump(self.mpc_meta, f, indent=4)
        try:
            trajectory_tracking_results(mpc_dir, self.t_ref, mpc_t, self.x_ref, state_in,
                                        self.u_ref, u_in, mpc_error, w_control=self.w_control, file_type='png')
        except Exception as e:
            rospy.logerr(f"An error occurred while plotting trajectory tracking results: {e}")

        # Check MHE is running and if it is continue to save MHE results
        if len(self.x_est) > 0:
            mhe = True
            # self.t_est = self.t_est - self.t_est[0]
            # while len(self.x_est) < self.seq_len * 2:
            #     self.x_est = np.append(self.x_est, self.x_est[-1][np.newaxis], axis=0)
            #     self.t_est = np.append(self.t_est, self.t_est[-1])

            if len(self.t_acc_est) > 0:
                while len(self.accel_est) < self.seq_len * 2:
                    self.accel_est = np.append(self.accel_est, self.accel_est[-1][np.newaxis], axis=0)
                self.t_acc_est = self.t_acc_est - self.t_acc_est[0]
                while len(self.t_acc_est) < self.seq_len * 2:
                    self.t_acc_est = np.append(self.t_acc_est, self.t_acc_est[-1])
                self.accel_est = self.accel_est[:self.seq_len * 2]
                self.t_acc_est = self.t_acc_est[:self.seq_len * 2]

            while len(self.y) < self.seq_len * 2:
                self.y = np.append(self.y, self.y[-1][np.newaxis], axis=0)
            self.t_imu = self.t_imu - self.t_imu[0]
            while len(self.t_imu) < self.seq_len * 2:
                self.t_imu = np.append(self.t_imu, self.t_imu[-1])
            self.y = self.y[:self.seq_len * 2]
            self.t_imu = self.t_imu[:self.seq_len * 2]
            while len(self.y_noisy) < self.seq_len * 2:
                self.y_noisy = np.append(self.y_noisy, self.y_noisy[-1][np.newaxis], axis=0)
            self.t_sensor_measurement = self.t_sensor_measurement - self.t_sensor_measurement[0]
            while len(self.t_sensor_measurement) < self.seq_len * 2:
                self.t_sensor_measurement = np.append(self.t_sensor_measurement, 
                                                      self.t_sensor_measurement[-1])
            self.t_sensor_measurement = self.t_sensor_measurement[:self.seq_len * 2]
            self.y_noisy = self.y_noisy[:self.seq_len * 2]
                
            mhe_error = np.zeros_like(self.x_est)
            a_meas_traj = np.zeros((len(self.x_est), 3))
            a_est_b_traj = np.zeros((len(self.x_est), 3))
            a_thrust_traj = np.zeros((len(self.x_est), 3))
            rospy.loginfo("Filling in MHE dataset and saving...")
            g = cs.vertcat(0, 0, -9.81)
            for i in tqdm(range(self.seq_len*2)):
                u = self.motor_thrusts[i]
                q_act = self.x_act[i][3:7]
                q_act_inv = quaternion_inverse(q_act)
                q = self.x_est[i][3:7]
                q_inv = quaternion_inverse(q)

                # Measured Acceleration
                a_meas = self.y[i][6:9]
                a_meas = np.stack(a_meas + v_dot_q(g, q_inv).T)
                a_meas_traj[i] = np.squeeze(self.y[i][6:9])

                # Model Accel Estimation
                a_thrust = cs.vertcat(0, 0, (u[0] + u[1] + u[2] + u[3]) * self.quad.max_thrust) / self.quad.mass
                a_thrust_traj[i] = np.squeeze(a_thrust.T)

                a_est_b = v_dot_q(v_dot_q(a_thrust, q) + g, q_inv)
                a_est_b = np.squeeze(a_est_b.T)
                a_est_b_traj[i] = a_est_b

                # MHE Model Error
                a_error = np.concatenate((np.zeros((1, 7)), a_meas - a_est_b, np.zeros((1, 3))), axis=None)
                mhe_error[i] = a_error
            mhe_p_error = rmse(self.t_act, self.x_act[:, :3], self.t_est, self.x_est[:, :3])
            mhe_q_error = q_rmse(self.t_act, self.x_act[:, 3:7], self.t_est, self.x_est[:, 3:7])
            mhe_v_error = rmse(self.t_act, self.x_act[:, 7:10], self.t_est, self.x_est[:, 7:10])
            # Organize arrays to dictionary
            mhe_dict = {
                "t": self.t_imu,
                "x_est": self.x_est,
                "x_act": self.x_act,
                "sensor_meas": self.y,
                "motor_thrusts": self.motor_thrusts,
                "error": mhe_error,
                "a_est_b": a_est_b_traj,
                "accel_est": self.accel_est,
            } 
            self.mhe_meta['p_rmse'] = mhe_p_error
            self.mhe_meta['q_rmse'] = mhe_q_error
            self.mhe_meta['v_rmse'] = mhe_v_error
            # Save results
            mhe_dir = os.path.join(self.mhe_dir, self.ref_traj_name)
            safe_mkdir_recursive(mhe_dir)
            with open(os.path.join(mhe_dir, "results.pkl"), "wb") as f:
                pickle.dump(mhe_dict, f)
            with open(os.path.join(mhe_dir, 'meta_data.json'), "w") as f:
                json.dump(self.mhe_meta, f, indent=4)
            
            self.t_act = self.t_act[:self.seq_len * 2]
            self.x_act = self.x_act[:self.seq_len * 2]
            self.t_est = self.t_est[:self.seq_len * 2]
            self.x_est = self.x_est[:self.seq_len * 2]
            self.t_imu = self.t_imu[:self.seq_len * 2]
            self.y = self.y[:self.seq_len * 2]
            self.t_sensor_measurement = self.t_sensor_measurement[:self.seq_len * 2]
            self.y_noisy = self.y_noisy[:self.seq_len * 2]
            if self.mhe_type == "k":
                self.t_acc_est = self.t_acc_est[:self.seq_len * 2]
                self.accel_est = self.accel_est[:self.seq_len * 2]
            try:
                if self.mhe_type == "k":
                    state_estimation_results(mhe_dir, self.t_act, self.x_act, self.t_est, self.x_est, self.t_imu, self.y,
                                             self.t_sensor_measurement, self.y_noisy, mhe_error, t_acc_est=self.t_acc_est, 
                                             accel_est=self.accel_est, file_type='png')
                else:
                    state_estimation_results(mhe_dir, self.t_act, self.x_act, self.t_est, self.x_est, self.t_imu, self.y,
                                             self.t_sensor_measurement, self.y_noisy, mhe_error, a_thrust=a_thrust_traj, 
                                             a_meas=a_meas_traj, file_type='png')
            except Exception as e:
                rospy.logerr(f"An error occurred while plotting state estimation results: {e}")    
        else:
            mhe = False
        # --- Reset all vectors ---
        # Vectors to store Reference Trajectory
        self.seq_len = None
        self.ref_traj_name = None
        self.ref_v = None
        self.x_ref = None
        self.t_ref = None
        self.u_ref = None
        # Vectors to store Tracking and Estimation Results
        self.t_act = np.zeros((0, 1))
        self.t_imu = np.zeros((0, 1))
        self.t_sensor_measurement = np.zeros((0, 1))
        self.t_est = np.zeros((0, 1))
        self.t_acc_est = np.zeros((0, 1))
        self.x_act = np.zeros((0, 13))
        self.x_est = np.zeros((0, 13))
        self.y = np.zeros((0, 9))
        self.y_noisy = np.zeros((0, 9))
        self.accel_est = np.zeros((0, 3))
        self.motor_thrusts = np.zeros((0, 4))
        self.w_control = np.zeros((0, 3))
        self.collective_thrusts = np.zeros((0, 1))

        # System States
        self.p_act = None
        self.q_act = None
        self.v_act = None
        self.w_act = None
        self.p_meas = None
        self.w_meas = None
        self.a_meas = None
        rospy.loginfo("Recording Complete.")
        rospy.loginfo("MPC: tracking RMSE: %.5f m. Max Vel: %.3f m/s" % (mpc_tracking_error, v_max))
        if mhe:
            rospy.loginfo("MHE: p Estimation RMSE: %.5f m" % (mhe_p_error))
            rospy.loginfo("MHE: q Estimation RMSE: %.5f deg" % (np.rad2deg(mhe_q_error)))
            rospy.loginfo("MHE: v Estimation RMSE: %.5f m/s" % (mhe_v_error))
    
    def imu_callback(self, msg):
        if not self.record:
            return
        
        self.w_meas = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        self.a_meas = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        if self.p_meas is not None:
            y = self.p_meas + self.w_meas + self.a_meas
            self.y = np.append(self.y, np.array(y)[np.newaxis, :], axis=0)
            self.t_imu = np.append(self.t_imu, msg.header.stamp.to_time())

    def motor_thrust_callback(self, msg):
        if not self.record:
            return
        
        motor_thrusts = tuple(msg.angular_velocities)
        self.motor_thrusts = np.append(self.motor_thrusts, [motor_thrusts], axis=0)
        self.motor_thrusts = np.append(self.motor_thrusts, [motor_thrusts], axis=0)
    
    def odom_callback(self, msg):
        # print("odom time: %.3f | act time: %.3f"%(msg.header.stamp.to_time(), time.time()))
        if not self.record:
            return
        
        p = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        q = [msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z]
        v = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
        w = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
        self.p_meas = p

        if self.env == "gazebo":
            v = v_dot_q(np.array(v), np.array(q)).tolist()

        x = p + q + v + w
        
        self.x_act = np.append(self.x_act, np.array(x)[np.newaxis, :], axis=0)
        self.t_act = np.append(self.t_act, msg.header.stamp.to_time())

    def state_est_callback(self, msg):
        # print("state est time: %.3f | act time: %.3f"%(msg.header.stamp.to_time(), time.time()))
        if not self.record:
            return
        
        p = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        q = [msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z]
        v_w = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
        w = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]

        x = p + q + v_w + w

        self.x_est = np.append(self.x_est, np.array(x)[np.newaxis, :], axis=0)
        self.t_est = np.append(self.t_est, msg.header.stamp.to_time())

    def acceleration_est_callback(self, msg):
        if not self.record:
            return
        
        a_est = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]

        self.accel_est = np.append(self.accel_est, np.array(a_est)[np.newaxis, :], axis=0)
        self.t_acc_est = np.append(self.t_acc_est, msg.header.stamp.to_time())

    def ref_callback(self, msg):
        if self.x_ref is not None:
            return
        
        self.seq_len = msg.seq_len
        if self.seq_len == 0:
            return
        
        self.ref_traj_name = msg.traj_name
        self.ref_v = msg.v_input

        # Save reference trajectory, relative times and inputs
        self.x_ref = np.array(msg.trajectory).reshape(self.seq_len, -1)
        self.t_ref = np.array(msg.dt)
        self.u_ref = np.array(msg.inputs).reshape(self.seq_len, -1)

    def control_callback(self, msg):
        if not self.record:
            return
        
        w_control = [msg.body_rate.x, msg.body_rate.y, msg.body_rate.z]
        # collective_thrust = msg.thrust
        
        self.w_control = np.append(self.w_control, np.array(w_control)[np.newaxis, :], axis=0)
        # self.w_control = np.append(self.w_control, np.array(w_control)[np.newaxis, :], axis=0)
        # self.collective_thrusts = np.append(self.collective_thrusts, collective_thrust)

    def control_gz_callback(self, msg):
        if not self.record:
            return
        w_control = [msg.bodyrates.x, msg.bodyrates.y, msg.bodyrates.z]
        # collective_thrust = msg.thrusts

        self.w_control = np.append(self.w_control, np.array(w_control)[np.newaxis, :], axis=0)
        # self.w_control = np.append(self.w_control, np.array(w_control)[np.newaxis, :], axis=0)
        # self.collective_thrusts = np.append(self.collective_thrusts, collective_thrust)

    def record_callback(self, msg):  
        if not self.record and msg.data == True:
            # Recording has begun
            rospy.loginfo("Recording...")     
            self.record = True

        if self.record and msg.data == False:
            # Recording ended
            self.record = False
            # Run thread for saving the recorded data
            _save_record_thread = threading.Thread(target=self.save_recording_data(), args=(), daemon=True)
            _save_record_thread.start()
    
    def sensor_measurement_callback(self, msg):
        if not self.record:
            return
        
        y_noisy = [msg.position.x, msg.position.y, msg.position.z,
                   msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
                   msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        self.y_noisy = np.append(self.y_noisy, np.array(y_noisy)[np.newaxis, :], axis=0)
        self.t_sensor_measurement = np.append(self.t_sensor_measurement, msg.header.stamp.to_time())

    def check_use_gt(self, event):
        use_groundtruth = rospy.get_param("/mpc/use_groundtruth", default=None)
        if use_groundtruth is not None and self.use_groundtruth != use_groundtruth:
            self.use_groundtruth = use_groundtruth
            rospy.loginfo("%sUsing Groundtruth!"%("" if use_groundtruth else "NOT "))
            if self.mpc_use_nn:
                use_nn_str = "_%s_%s"%(self.mpc_model_type,
                                       self.mpc_correction_mode)
            else:
                use_nn_str = ""
            self.mpc_dataset_name = "%s_mpc_%s%s%s"%(self.env, 
                                                     "gt_" if self.use_groundtruth else "", 
                                                     self.quad_name,
                                                     use_nn_str)
            self.mpc_dir = os.path.join(self.results_dir, self.mpc_dataset_name)
            safe_mkdir_recursive(self.mpc_dir)
            self.mpc_meta['gt'] = use_groundtruth
            # Create Directory
            safe_mkdir_recursive(self.mpc_dir)

    def check_mpc_use_nn(self, event):
        mpc_use_nn = rospy.get_param("/mpc/use_nn", default=None)
        if mpc_use_nn is not None and self.mpc_use_nn != mpc_use_nn:
            self.mpc_use_nn = mpc_use_nn
            self.mpc_model_name = rospy.get_param("/mpc/model_name", default=None)
            self.mpc_model_type = rospy.get_param("/mpc/model_type", default=None)
            self.mpc_correction_mode = rospy.get_param("/mpc/correction_mode", default=None)

            rospy.loginfo("MPC w/%s"% ("OUT NN" if not mpc_use_nn else " %s"%self.mpc_model_type))
            if self.mpc_use_nn:
                use_nn_str = "_%s_%s"%(self.mpc_model_type,
                                       self.mpc_correction_mode)
            else:
                use_nn_str = ""
            self.mpc_dataset_name = "%s_mpc_%s%s%s"%(self.env, 
                                                     "gt_" if self.use_groundtruth else "", 
                                                     self.quad_name,
                                                     use_nn_str)
            self.mpc_dir = os.path.join(self.results_dir, self.mpc_dataset_name)
            self.mpc_meta['use_nn'] = self.mpc_use_nn
            self.mpc_meta['model_type'] = self.mpc_model_type
            self.mpc_meta['model_name'] = self.mpc_model_name
            self.mpc_meta['correction_mode'] = self.mpc_correction_mode

            # Create Directory
            safe_mkdir_recursive(self.mpc_dir)

    def check_mhe_type(self, event):
        mhe_type = rospy.get_param("/mhe/mhe_type", default=None)
        if mhe_type.lower() == "kinematic":
            mhe_type = "k"
        elif mhe_type.lower() == "dynamic":
            mhe_type = "d"
        if mhe_type is not None and self.mhe_type != mhe_type:
            self.mhe_type = mhe_type
            rospy.loginfo("%s-MHE w/%s"% (self.mhe_type.upper(), "OUT NN" if not self.mhe_use_nn else " %s"%self.mhe_model_type))
            self.mhe_meta['mhe_type'] = self.mhe_type
            if self.mhe_use_nn:
                use_nn_str = "_%s_%s"%(self.mhe_model_type,
                                        self.mhe_correction_mode)
            else:
                use_nn_str = ""
            self.mhe_dataset_name = "%s_%smhe_%s%s"%(self.env, 
                                                   self.mhe_type, 
                                                   self.quad_name,
                                                   use_nn_str)
            self.mhe_dir = os.path.join(self.results_dir, self.mhe_dataset_name)
            # Create Directory
            safe_mkdir_recursive(self.mhe_dir)

    def check_mhe_use_nn(self, event):
        if self.mhe_type == "k":
            return
        
        mhe_use_nn = rospy.get_param("/mhe/use_nn", default=None)
        if mhe_use_nn is not None and self.mhe_use_nn != mhe_use_nn:
            self.mhe_use_nn = mhe_use_nn
            self.mhe_model_name = rospy.get_param("/mhe/model_name", default=None)
            self.mhe_model_type = rospy.get_param("/mhe/model_type", default=None)
            self.mhe_correction_mode = rospy.get_param("/mhe/correction_mode", default=None)
            rospy.loginfo("%s-MHE w/%s"% (self.mhe_type.upper(), "OUT NN" if not self.mhe_use_nn else " %s"%self.mhe_model_type))

            self.mhe_meta['use_nn'] = self.mhe_use_nn
            self.mhe_meta['model_type'] = self.mhe_model_type
            self.mhe_meta['model_name'] = self.mhe_model_name
            self.mhe_meta['correction_mode'] = self.mhe_correction_mode
            if self.mhe_use_nn:
                use_nn_str = "_%s_%s"%(self.mhe_model_type,
                                        self.mhe_correction_mode)
            else:
                use_nn_str = ""
            self.mhe_dataset_name = "%s_%smhe_%s%s"%(self.env, 
                                                   self.mhe_type, 
                                                   self.quad_name,
                                                   use_nn_str)
            self.mhe_dir = os.path.join(self.results_dir, self.mhe_dataset_name)

            # Create Directory
            safe_mkdir_recursive(self.mhe_dir)



def main():
    rospy.init_node("visualizer")
    visualizer = VisualizerWrapper()

    rospy.spin()

if __name__ == "__main__":
    main()
