#!/usr/bin/env python

import rospy
import threading
import numpy as np
import time

from src.utils.utils import v_dot_q, state_features_to_idx, sensor_features_to_idx
from src.quad.quad import Quadrotor
from src.mhe.quad_optimizer_mhe import QuadOptimizerMHE
from src.neural_ode.NeuralODE import load_neural_ode
from src.gp.GPyModelWrapper import GPyModelWrapper

from hybrid_mpc_mhe.msg import QuadSensor, ModelCorrection
from std_msgs.msg import Bool, Header
from mav_msgs.msg import Actuators
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3

def load_model(model_name, model_type):
    if model_type == "node":
        model = load_neural_ode(model_name)
    if model_type == "gp":
        model = GPyModelWrapper(model_name, load=True)
    return model

def odometry_parse(odom_msg):
    p = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z])
    q = np.array([odom_msg.pose.pose.orientation.w, odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y,
         odom_msg.pose.pose.orientation.z])
    v = np.array([odom_msg.twist.twist.linear.x, odom_msg.twist.twist.linear.y, odom_msg.twist.twist.linear.z])
    w = np.array([odom_msg.twist.twist.angular.x, odom_msg.twist.twist.angular.y, odom_msg.twist.twist.angular.z])

    return p, q, v, w

def imu_parse(imu_msg):
    w = np.array([imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z])
    a = np.array([imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z])

    return w, a

class MHENode:
    def __init__(self):
        rospy.init_node('MHENode')
        self.init_params()
        self.init_mhe()
        self.init_publishers()
        self.init_subscribers()

        rate = rospy.Rate(1)

        if not self.y_available:
            rospy.loginfo("MHE: Waiting for Sensor Measurements...")
            while (not self.y_available and not rospy.is_shutdown()):
                rate.sleep()
        
        if (self.mhe_type == "d" and not self.u_available):
            self.u = np.ones(4) * self.quad.get_hover_thrust()
            rospy.loginfo("MHE: Waiting for Command Inputs...")
            while (not self.u_available and not rospy.is_shutdown()):
                rate.sleep()

        rospy.loginfo("%s-MHE: %s Loaded in %s"%(self.mhe_type.upper(), self.quad_name, self.env))
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()
    
    def init_params(self):
        # System environment
        self.env = rospy.get_param("/environment", default="gazebo")
        self.quad_name = rospy.get_param("/quad_name", default=None)
        assert self.quad_name != None

        # Initialize State Measurement Variables
        self.p = None
        self.r = None
        self.a = None
        self.y = None
        self.y_hist = np.zeros((0, 9))
        self.y_hist_cp = None
        self.y_available = False
        self.y_history_filled = False
        # Initialize Input Command Variables
        self.u = None
        self.u_hist = np.zeros((0, 4))
        self.u_hist_cp = None
        self.u_available = False
        self.u_history_filled = False

        # Initialize State Estimation Variables
        self.x_est = None
        self.a_est = None

        # Initialize MHE Variables
        self.mhe_idx = 0
        self.opt_dt = 0
        self.mhe_seq_num = 0

        # Initialize MHE Thread
        self.mhe_thread = threading.Thread()
        self.mhe_thread.start()
        self.lock = threading.Lock()
        self.last_u_count = 0
        
        # Remember the sequence number of the last odometry message received.
        self.last_imu_seq_number = 0

        # Boolean indicating whether system is recording
        self.record = False

        # Simulate Sensor noise
        self.sensor_noise = rospy.get_param("/sensor_noise", default=False)
        self.p_noise_std = rospy.get_param("/p_noise_std", default=0)
        self.r_noise_std = rospy.get_param("/r_noise_std", default=0)
        self.a_noise_std = rospy.get_param("/a_noise_std", default=0)

    def init_publishers(self):
        # Publisher topic names
        state_est_topic = rospy.get_param("/state_est_topic", default="/" + self.quad_name + "/state_est") 
        acceleration_est_topic = rospy.get_param("/acceleration_est_topic", default="/" + self.quad_name + "/acceleration_est") 
        sensor_measurement_topic = rospy.get_param("/sensor_measurement_topic", default="/" + self.quad_name + "/sensor_measurement") 

        # Publishers
        self.state_est_pub = rospy.Publisher(state_est_topic, Odometry, queue_size=10, tcp_nodelay=True)
        self.acceleration_est_pub = rospy.Publisher(acceleration_est_topic, Imu, queue_size=10, tcp_nodelay=True)
        self.sensor_measurement_pub = rospy.Publisher(sensor_measurement_topic, QuadSensor, queue_size=10, tcp_nodelay=True)

        if self.use_nn and self.correction_mode == "offline":
            model_correction_topic = rospy.get_param("~model_correction_topic", default="/" + self.quad_name + "/mhe/model_correction")
            self.model_correction_pub = rospy.Publisher(model_correction_topic, ModelCorrection, queue_size=10, tcp_nodelay=True)

    def init_subscribers(self):
        # Subscriber topic names
        motor_thrust_topic = rospy.get_param("/motor_thrust_topic", default="/" + self.quad_name + "/motor_thrust")
        record_topic = rospy.get_param("/record_topic", default="/" + self.quad_name + "/record")
        pose_topic = rospy.get_param("/pose_topic", default="/" + self.quad_name + "/ground_truth/pose")
        imu_topic = rospy.get_param("/imu_topic", default="/" + self.quad_name + "/ground_truth/imu")

        # Subscribers
        self.motor_thrust_sub = rospy.Subscriber(motor_thrust_topic, Actuators, self.motor_thrust_callback, queue_size=10, tcp_nodelay=True)
        self.record_sub = rospy.Subscriber(record_topic, Bool, self.record_callback, queue_size=10, tcp_nodelay=True)
        self.pose_sub = rospy.Subscriber(pose_topic, Pose, self.pose_callback, queue_size=10, tcp_nodelay=True)
        self.imu_sub = rospy.Subscriber(imu_topic, Imu, self.imu_callback, queue_size=10, tcp_nodelay=True)

    def init_mhe(self):
        # MHE Parameters
        mhe_type = rospy.get_param("~mhe_type", default="kinematic")
        if mhe_type == "kinematic":
            self.mhe_type = "k"
        elif mhe_type == "dynamic":
            self.mhe_type = "d"
        else:
            rospy.logerr("MHE: Invalid MHE type selected")  
            return 0 
        self.use_nn = rospy.get_param("~use_nn", default=False)
        self.n_mhe = rospy.get_param("~n_mhe", default=50)
        self.t_mhe = rospy.get_param("~t_mhe", default=0.5)
        
        # Load NN models
        if (self.use_nn):
            self.model_name = rospy.get_param("~nn/model_name", default=None)
            self.model_type = rospy.get_param("~nn/model_type", default=None).lower()
            self.input_features = rospy.get_param("~nn/input_features", default=None)
            self.nn_input_idx = sensor_features_to_idx(self.input_features)
            self.output_features = rospy.get_param("~nn/output_features", default=None)
            self.nn_output_idx = state_features_to_idx(self.output_features)
            self.correction_mode = rospy.get_param("~nn/correction_mode", default="online")
            self.nn_model = load_model(self.model_name, self.model_type)
            self.nn_params = {
                'model_name': self.model_name,
                'model_type': self.model_type,
                'input_features': self.input_features,
                'nn_input_idx': self.nn_input_idx,
                'output_features': self.output_features,
                'nn_output_idx': self.nn_output_idx,
                "correction_mode": self.correction_mode,
                'nn_model': self.nn_model,
            }
            self.nn_corr = None
        else:
            self.nn_params = {}

        # MHE costs
        # System Noise
        w_p = np.ones((1,3)) * rospy.get_param("~cost/w_p", default=0.004)
        w_q = np.ones((1,3)) * rospy.get_param("~cost/w_q", default=0.01)
        w_v = np.ones((1,3)) * rospy.get_param("~cost/w_v", default=0.005)
        w_r = np.ones((1,3)) * rospy.get_param("~cost/w_r", default=0.5)
        w_a = np.ones((1,3)) * rospy.get_param("~cost/w_a", default=0.05)
        w_d = np.ones((1,3)) * rospy.get_param("~cost/w_d", default=0.00001)
        # Measurement Noise
        v_p = np.ones((1,3)) * rospy.get_param("~cost/v_p", default=0.002)
        v_r = np.ones((1,3)) * rospy.get_param("~cost/v_r", default=1e-6)
        v_a = np.ones((1,3)) * rospy.get_param("~cost/v_a", default=1e-5)
        v_d = np.ones((1,3)) * rospy.get_param("~cost/v_d", default=0.0001)
        # Arrival cost factor
        q0_factor = rospy.get_param("~cost/q0_factor", default=1)

        if self.mhe_type == "k":
            q_mhe = 1/np.squeeze(np.hstack((w_p, w_q, w_v, w_r, w_a))**2)
            r_mhe = 1/np.squeeze(np.hstack((v_p, v_r, v_a))**2)
        elif self.mhe_type == "d":
            if not self.use_nn:
                q_mhe = 1/np.squeeze(np.hstack((w_p, w_q, w_v, w_r))**2)
                r_mhe = 1/np.squeeze(np.hstack((v_p, v_r))**2)
            if self.use_nn and self.correction_mode == "offline":
                q_mhe = 1/np.squeeze(np.hstack((w_p, w_q, w_v, w_r, w_d))**2)
                r_mhe = 1/np.squeeze(np.hstack((v_p, v_r, v_d))**2)

        # Load Quadrotor instance
        self.quad = Quadrotor(self.quad_name)

        # Compile Acados Model
        self.mhe = QuadOptimizerMHE(self.quad, t_mhe=self.t_mhe, n_mhe=self.n_mhe,
                                    mhe_type=self.mhe_type,
                                    q_mhe=q_mhe, q0_factor=q0_factor, r_mhe=r_mhe,
                                    use_nn=self.use_nn, nn_params=self.nn_params)

    def motor_thrust_callback(self, msg):
        self.u = np.array(msg.angular_velocities)
        # self.u_hist = np.append(self.u_hist, self.u[np.newaxis, :], axis=0)
        # self.u_hist = np.append(self.u_hist, self.u[np.newaxis, :], axis=0)
        self.u_available = True

    def record_callback(self, msg):
        if (self.record and msg.data == False):
            self.opt_dt /= self.mhe_idx
            self.opt_dt *= 1000
            rospy.loginfo("MHE: Estimation complete. Mean MHE opt. time: %.3f ms", self.opt_dt)
            self.opt_dt = 0
            # self.save_recording_data()
        elif (not self.record and msg.data == True):
            self.opt_dt = 0
            self.mhe_idx = 0
   
        self.record = msg.data

    def pose_callback(self, msg):
        self.p = np.array([msg.position.x,
                           msg.position.y,
                           msg.position.z])
    
    
    def imu_callback(self, msg):
        self.r, self.a = imu_parse(msg)
        if self.p is None:
            self.last_imu_seq_number = msg.header.seq
            return
        
        if self.sensor_noise:
            p_noise = self.p_noise_std * np.random.standard_normal(3)
            r_noise = self.r_noise_std * np.random.standard_normal(3)
            a_noise = self.a_noise_std * np.random.standard_normal(3)
            self.p = self.p + p_noise
            self.r = self.r + r_noise
            self.a = self.a + a_noise
        
        quad_sensor_msg = QuadSensor()
        quad_sensor_msg.header = Header()
        quad_sensor_msg.header.stamp = rospy.Time.now()
        quad_sensor_msg.header.seq = msg.header.seq
        quad_sensor_msg.position.x = self.p[0]
        quad_sensor_msg.position.y = self.p[1]
        quad_sensor_msg.position.z = self.p[2]
        quad_sensor_msg.angular_velocity.x = self.r[0]
        quad_sensor_msg.angular_velocity.y = self.r[1]
        quad_sensor_msg.angular_velocity.z = self.r[2]
        quad_sensor_msg.linear_acceleration.x = self.a[0]
        quad_sensor_msg.linear_acceleration.y = self.a[1]
        quad_sensor_msg.linear_acceleration.z = self.a[2]
        self.sensor_measurement_pub.publish(quad_sensor_msg)

        # Concatenate sensor measurements
        if self.mhe_type == "k":
            self.y = np.hstack((self.p, self.r, self.a))
        elif self.mhe_type == "d" and not self.use_nn:
            self.y = np.hstack((self.p, self.r))
        elif self.mhe_type == "d" and self.use_nn:
            if (self.correction_mode == "offline"):
                # Compute model error using NN model
                y = np.hstack((self.p, self.r, self.a))
                self.nn_corr = self.nn_model(y[self.nn_input_idx])
                self.y = np.hstack((self.p, self.r, self.nn_corr))
            elif self.correction_mode == "online":
                # TODO: it may not always be this 
                self.y = np.hstack((self.p, self.r, self.a))
        self.y_available = True
        
        if self.mhe_type == "d" and self.u is None:
            self.last_imu_seq_number = msg.header.seq
            return

        # Check for any skipped messages.
        skipped_messages = 0
        if self.last_imu_seq_number > 0 and self.mhe_seq_num > 10:
        # Count how many messages were skipped
            skipped_messages = int(msg.header.seq - self.last_imu_seq_number - 1)
            if skipped_messages > 0:
                warn_msg = "MHE Recording time skipped messages: %d" % skipped_messages
                rospy.logwarn(warn_msg)
        self.last_imu_seq_number = msg.header.seq

        # Fill empty vectors with current sensor measurements and motor thrust
        if not self.y_history_filled:
            self.y_hist = np.tile(self.y, (self.n_mhe, 1))
            self.y_history_filled = True
        
        # Add current measurement to array and also add the number of missed measurements to be up to sync
        for _ in range(1 + skipped_messages):
            self.y_hist = np.append(self.y_hist, self.y[np.newaxis, :], axis=0)

        # Correct Measurement and Input history list lengths
        if self.y_hist.shape[0] >= self.n_mhe:
            self.y_hist = self.y_hist[-(self.n_mhe):, :]
        self.y_hist_cp = self.y_hist.copy()
        

        if self.mhe_type == "d":
            extra_u_len = 0
            if self.last_u_count == 2:
                extra_u_len = 0
            elif self.last_u_count == 0:
                extra_u_len = 2
            elif self.last_u_count == 1:
                extra_u_len = 1
            self.last_u_count += 1
            if not self.u_history_filled and self.u is not None:
                self.u_hist = np.tile(self.u, (self.n_mhe, 1))
                self.u_history_filled = True
            for _ in range(1 + skipped_messages):
                self.u_hist = np.append(self.u_hist, self.u[np.newaxis, :], axis=0)
            # if self.u_hist.shape[0] > self.n_mhe+extra_u_len:
            #     self.u_hist = self.u_hist[-(self.n_mhe+extra_u_len):, :]
            if self.u_hist.shape[0] >= self.n_mhe:
                self.u_hist = self.u_hist[-(self.n_mhe):, :]
            self.u_hist_cp = self.u_hist.copy()

        # Run MHE
        def _mhe_thread_func():
            self.run_mhe()

        self.mhe_thread.join()
        self.mhe_thread = threading.Thread(target=_mhe_thread_func(), args=(), daemon=True)
        self.mhe_thread.start()

    def run_mhe(self):             
        self.mhe.set_history_trajectory(self.y_hist_cp, self.u_hist_cp)
        if (self.mhe.solve_mhe() == 0):
            self.x_est = self.mhe.get_state_est()
            self.opt_dt += self.mhe.get_opt_dt()
            self.mhe_idx += 1

        # Publish acceleration estimation
        if self.mhe_type == "k":
            self.accel_est = self.x_est[-3:]
            accel_est_msg = Imu()
            accel_est_msg.header = Header()
            accel_est_msg.header.stamp = rospy.Time.now()
            accel_est_msg.header.seq = self.mhe_seq_num
            accel_est_msg.header.frame_id = self.quad_name + "/base_link"
            accel_est_msg.linear_acceleration.x = self.accel_est[0]
            accel_est_msg.linear_acceleration.y = self.accel_est[1]
            accel_est_msg.linear_acceleration.z = self.accel_est[2]
            self.acceleration_est_pub.publish(accel_est_msg)

        # Publish State estimates
        state_est_msg = Odometry()
        state_est_msg.header = Header()
        state_est_msg.header.stamp = rospy.Time.now()
        state_est_msg.header.seq = self.mhe_seq_num
        state_est_msg.header.frame_id = "world"
        state_est_msg.child_frame_id = self.quad_name + "/base_link"
        state_est_msg.pose.pose.position = Point(*self.x_est[:3])
        w, x, y, z = self.x_est[3:7]
        state_est_msg.pose.pose.orientation = Quaternion(x, y, z, w)
        state_est_msg.twist.twist.linear = Vector3(*self.x_est[7:10])
        state_est_msg.twist.twist.angular = Vector3(*self.x_est[10:13])
        self.state_est_pub.publish(state_est_msg)

        # Publish Model corrections if offline correction
        if self.use_nn and self.correction_mode == "offline":
            correction_msg = ModelCorrection()
            idx = 13
            if 'q' in self.output_features:
                w, x, y, z = self.x_est[idx:idx+4]
                correction_msg.orientation = Quaternion(x, y, z, w)
                idx += 4
            if 'v' in self.output_features:
                correction_msg.velocity = Vector3(*self.x_est[idx:idx+3])
                idx += 3
            if 'w' in self.output_features:
                correction_msg.angular_velocity = Vector3(*self.x_est[idx:idx+3])
            self.model_correction_pub.publish(correction_msg)

        self.mhe_seq_num += 1

def main():
    MHENode()

if __name__ == "__main__":
    main()