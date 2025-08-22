#!/usr/bin/env python

import rospy
import rosnode
import threading
import numpy as np
from scipy.signal import find_peaks

from src.utils.utils import v_dot_q, quaternion_state_mse, state_features_to_idx, world_to_body_velocity_mapping
from src.quad.quad import Quadrotor
from src.mpc.quad_optimizer_mpc import QuadOptimizerMPC
from src.neural_ode.NeuralODE import load_neural_ode
from src.gp.GPyModelWrapper import GPyModelWrapper

from hybrid_mpc_mhe.msg import ReferenceTrajectory
from mav_msgs.msg import Actuators
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Empty, Header, Float32
from quadrotor_msgs.msg import ControlCommand
from mavros_msgs.msg import AttitudeTarget
from mavros_msgs.srv import CommandBool, SetMode

def load_model(model_name, model_type):
    if model_type == "node":
        model = load_neural_ode(model_name)
    if model_type == "gp":
        model = GPyModelWrapper(model_name, load=True)
    return model

def odometry_parse(odom_msg):
    p = [odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z]
    q = [odom_msg.pose.pose.orientation.w, odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y,
         odom_msg.pose.pose.orientation.z]
    v = [odom_msg.twist.twist.linear.x, odom_msg.twist.twist.linear.y, odom_msg.twist.twist.linear.z]
    w = [odom_msg.twist.twist.angular.x, odom_msg.twist.twist.angular.y, odom_msg.twist.twist.angular.z]

    return p, q, v, w

class MPCNode:
    def __init__(self):
        rospy.init_node('MPCNode')
        self.init_params()
        self.init_publishers()
        self.init_rosservice()
        self.init_mpc()
        self.init_subscribers()

        rate = rospy.Rate(1)
        
        if not self.x_available:
            rospy.loginfo("MPC: Waiting for System States...")
            while(not self.x_available and not rospy.is_shutdown()):
                rate.sleep()    

        rospy.loginfo("MPC: %s Loaded in %s"%(self.quad_name, self.env))
        while not rospy.is_shutdown():
            # Publish if MPC is busy with a current trajectory
            msg = Bool()
            msg.data = not (self.x_ref is None and self.x_available)
            self.status_pub.publish(msg)
            rate.sleep()
        return
    
    def init_params(self):
        # System environment
        self.env = rospy.get_param("/environment", default="gazebo")
        self.quad_name = rospy.get_param("/quad_name", default=None)

        assert self.quad_name != None
        # Load Quad Instance
        self.quad = Quadrotor(self.quad_name)

        self.use_groundtruth = rospy.get_param("~use_groundtruth", default=True)
        
        # Sensor noise
        self.sensor_noise = rospy.get_param("/sensor_noise", default=False)
        if self.sensor_noise and not self.use_groundtruth:
            self.sensor_noise_level = rospy.get_param("/noise_level", default=3)
        else:
            self.sensor_noise_level = 0

        # Initial flight parameters
        self.init_thr = rospy.get_param("~init_thr", default=0.5)
        self.init_v = rospy.get_param("~init_v", default=0.3)

        # Landing Parameters
        self.land_thr = rospy.get_param("~land_thr", default=0.05)
        self.land_z = rospy.get_param("~land_z", default=0.05)
        self.land_dz = rospy.get_param("~land_dz", default=0.1)
        self.ref_gen_node_name = rospy.get_param('/ref_pub_node_name', default="/ref_gen/ref_gen") 
        self.no_more_ref = True

        # Simulate payload mass
        self.payload_mass_gt = rospy.get_param("/payload_mass", default=None)
        if self.payload_mass_gt is None or self.payload_mass_gt == 0:
            self.payload = False
            self.payload_mass_est = None
        else:
            self.payload = True
            self.payload_mass_est = 0
        self.payload_mass_pickedup = False
        self.base_mass = self.quad.get_base_mass()

        # Initialize State Variables
        self.x = None
        self.u = None
        self.x_available = False
        self.last_x_seq_number = 0
        self.land_override = False
        self.ground_level = True

        # Initialize Reference Trajectory variables
        self.ref_received = False
        self.ref_traj_name = None
        self.ref_len = None
        self.x_ref = None
        self.u_ref = None
        self.t_ref = None
        self.last_x_ref = None
        self.last_u_ref = None
        self.x_initial_reached = False

        # Initialize Provisional Reference Points
        self.x_ref_prov = None
        self.u_ref_prov = None

        # Initialize MPC Variables
        self.mpc_idx = 0
        self.opt_dt = 0

        # Initialize MPC Thread
        self.mpc_thread = threading.Thread()
        self.mpc_thread.start()

    def init_publishers(self):
        """
        Initialize ROS Publishers
        """
        # Publisher topic names
        control_topic = rospy.get_param("/control_topic", default="/" + self.quad_name + "/autopilot/control_command_input")
        motor_thrust_topic = rospy.get_param("/motor_thrust_topic", default="/" + self.quad_name + "/motor_thrust")
        record_topic = rospy.get_param("/record_topic", default="/" + self.quad_name + "/record")
        status_topic = rospy.get_param("/mpc_status_topic", default="/mpc/busy")
        # control_gz_topic = rospy.get_param("~control_gz_topic", default="/" + self.quad_name + "/autopilot/control_command_input")
        quad_mass_change_topic = rospy.get_param("/quad_mass_change_topic", default="/" + self.quad_name + "/mass_change")

        # Publishers
        if self.env == "gazebo":
            self.control_pub = rospy.Publisher(control_topic, ControlCommand, queue_size=1, tcp_nodelay=True)
        elif self.env == "arena":
            self.control_pub = rospy.Publisher(control_topic, AttitudeTarget, queue_size=1, tcp_nodelay=True)
        self.motor_thrust_pub = rospy.Publisher(motor_thrust_topic, Actuators, queue_size=1, tcp_nodelay=True)
        self.record_pub = rospy.Publisher(record_topic, Bool, queue_size=1, tcp_nodelay=True)
        self.status_pub = rospy.Publisher(status_topic, Bool, queue_size=1, tcp_nodelay=True)
        if self.payload:
            self.quad_mass_change_pub = rospy.Publisher(quad_mass_change_topic, Float32, queue_size=1, tcp_nodelay=True) # Only works for Gazebo
            # Ensure Quad mass is correct mass
            self.quad_mass_change_pub.publish(Float32(self.base_mass))

    def init_rosservice(self):
        set_mode_srvc = rospy.get_param("/set_mode_srvc", default="/mavros/set_mode")
        arming_srvc = rospy.get_param("/arming_srvc", default="/mavros/cmd/arming")
        self.set_mode_client = rospy.ServiceProxy(set_mode_srvc, SetMode)
        self.arming_client = rospy.ServiceProxy(arming_srvc, CommandBool)

    def init_subscribers(self):
        '''
        '''
        # Subscriber topic names
        ref_topic = rospy.get_param("/ref_topic", default="/reference")
        state_est_topic = rospy.get_param("/state_est_topic", default= "/" + self.quad_name + "/state_est")
        odom_topic = rospy.get_param("/odom_topic", default="/" + self.quad_name + "/ground_truth/odometry")
        land_topic = rospy.get_param("/land", default="/" + self.quad_name + "/land")
        payload_mass_est_topic = rospy.get_param("/payload_mass_est_topic", default="/" + self.quad_name + "/payload_mass_est")

        # Subscribers        
        self.ref_sub = rospy.Subscriber(ref_topic, ReferenceTrajectory, self.reference_callback)
        if self.use_groundtruth:
            self.state_est_sub = rospy.Subscriber(odom_topic, Odometry, self.state_est_callback, queue_size=1, tcp_nodelay=True)
        else:
            self.state_est_sub = rospy.Subscriber(state_est_topic, Odometry, self.state_est_callback, queue_size=1, tcp_nodelay=True)
        self.land_sub = rospy.Subscriber(land_topic, Empty, self.land_callback)
        if not self.use_groundtruth:
            self.payload_mass_est_sub = rospy.Subscriber(payload_mass_est_topic, Float32, self.payload_mass_est_callback, queue_size=10, tcp_nodelay=True)

    def init_mpc(self):
        # Binary variable to run MPC only once every other odometry callback
        self.optimize_next = False

        # MPC Parameters
        self.control_freq_factor = rospy.get_param("~control_freq_factor", default=5)
        self.use_nn = rospy.get_param("~use_nn", default=False)
        self.n_mpc = rospy.get_param("~n_mpc", default=10)
        self.t_mpc = rospy.get_param("~t_mpc", default=1)

        # MPC Costs
        q_p = np.ones((1,3)) * rospy.get_param("~cost/noise_level_" + str(self.sensor_noise_level) + "/q_p", default=35)
        q_q = np.ones((1,3)) * rospy.get_param("~cost/noise_level_" + str(self.sensor_noise_level) + "/q_q", default=25)
        q_v = np.ones((1,3)) * rospy.get_param("~cost/noise_level_" + str(self.sensor_noise_level) + "/q_v", default=10)
        q_r = np.ones((1,3)) * rospy.get_param("~cost/noise_level_" + str(self.sensor_noise_level) + "/q_r", default=10)

        qt_factor = rospy.get_param("~cost/noise_level_" + str(self.sensor_noise_level) + "/qt_factor", default=0.1)

        q_mpc = np.squeeze(np.hstack((q_p, q_q, q_v, q_r)))
        r_mpc = np.array([1.0, 1.0, 1.0, 1.0]) * rospy.get_param("~cost/noise_level_" + str(self.sensor_noise_level) + "/r_mpc", default=1)
        rospy.loginfo("MPC Weights Q: %s"%str(q_mpc))
        rospy.loginfo("MPC Weights R: %s"%str(r_mpc))
        # Load NN models
        if (self.use_nn):
            self.model_name = rospy.get_param("~nn/model_name", default=None)
            self.model_type = rospy.get_param("~nn/model_type", default=None).lower()
            self.input_features = rospy.get_param("~nn/input_features", default=None)
            self.nn_input_idx = state_features_to_idx(self.input_features)
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
        else:
            self.nn_params = {}
            self.nn_corr_i = None
            
        # Compile Acados Model
        self.quad_opt = QuadOptimizerMPC(self.quad, t_mpc=self.t_mpc, n_mpc=self.n_mpc,
                                         q_mpc=q_mpc, qt_factor=qt_factor, r_mpc=r_mpc, 
                                         use_nn=self.use_nn, nn_params=self.nn_params,
                                         payload=self.payload)
    
    def land_callback(self, msg):
        """
        Callback function for land 
        """
        if msg.data:
            # Lower drone to a safe height
            self.land_override = True
            rospy.loginfo("Landing...")
            # Stop recording
            record_msg = Bool()
            record_msg.data = False
            self.record_pub.publish(record_msg)

    def reference_callback(self, msg):
        """
        Callback function for reference trajectory
        """
        if (not self.ref_received):
            self.ref_len = msg.seq_len
            # TODO: This functionality not tested
            # if self.ref_len == 0:
            #     # Hover-in-place mode
            #     self.x_ref = self.x[:7]
            #     self.u_ref = None
            #     self.t_ref = None

            #     off_msg = Empty()
            #     self.off_pub.publish(off_msg)
            #     # self.controller_off = True

            #     self.landing = False
            #     # rospy.loginfo("No more references will be received")
            #     return

            # Save reference name
            self.ref_traj_name = msg.traj_name

            # Save reference trajectory, relative times and inputs
            self.x_ref = np.array(msg.trajectory).reshape(self.ref_len, -1)
            self.t_ref = np.array(msg.dt)
            self.u_ref = np.array(msg.inputs).reshape(self.ref_len, -1)

            # If using GP, compute GP corrections
            if self.use_nn and self.correction_mode == "offline":
                if "v" in self.input_features:
                    x_ref = world_to_body_velocity_mapping(self.x_ref)
                else:
                    x_ref = x_ref
                input = x_ref[:, self.nn_input_idx]
                if "u" in self.input_features:
                    input = np.append(input, self.u_ref, axis=1)
                self.nn_corr = self.nn_model.predict(input, skip_variance=True)

            # Select Payload mass pickup and dropoff points
            if self.payload:
                if np.all(self.x_ref[:, 2] == self.x_ref[0, 2]):
                    # Find peaks in the x-trajectory to pickup and drop payload mass
                    self.drop_mass_idx,_ = find_peaks(self.x_ref[:, 0])
                    self.pickup_mass_idx,_ = find_peaks(-self.x_ref[:, 0])
                    if self.pickup_mass_idx[-1] > self.drop_mass_idx[-1]:
                        self.pickup_mass_idx = self.pickup_mass_idx[:-1]
                else:
                    # Find peaks in the z-trajectory to pickup and drop payload mass
                    self.drop_mass_idx,_ = find_peaks(self.x_ref[:, 2])
                    self.pickup_mass_idx,_ = find_peaks(-self.x_ref[:, 2])
                    if self.pickup_mass_idx[-1] > self.drop_mass_idx[-1]:
                        self.pickup_mass_idx = self.pickup_mass_idx[:-1]
            if len(self.t_ref) > 0:
                rospy.loginfo("New trajectory received. Time duration: %.2f s" % self.t_ref[-1])
                self.ref_received = True
            else:
                rospy.logwarn("Trajectory vector is empty.")
        else:
            rospy.logwarn("Ignoring new trajectory received. Still in execution of another trajectory.")

    def set_reference_trajectory(self):
        """
        Set reference trajectory
        """
        if not self.x_available:
            return

        # Check if landing mode
        if self.land_override:
            if self.payload:
                self.quad_mass_change_pub.publish(Float32(self.base_mass))
            x_ref = self.last_x_ref if self.last_x_ref is not None else np.array(self.x)[np.newaxis, :]
            dz = np.sign(self.land_z - self.x[2]) * self.land_dz
            x_ref[0, 2] = min(self.land_z, self.x[2] + dz) if dz > 0 else max(self.land_z, self.x[2] + dz)
            u_ref = self.last_u_ref if self.last_u_ref is not None else np.array([[0, 0, 0, 0]])

            # TODO: Disarming drone
            # Reached landing heigh
            if (abs(self.x[2] - self.land_z) < self.land_thr):
                if (not self.ground_level):
                    rospy.loginfo("Vehicle at Ground Level")
                    self.ground_level = True
                    if self.ref_gen_node_name in rosnode.get_node_names():
                        print(rosnode.get_node_names())
                    else:
                        self.no_more_ref = True
                self.ref_received = False
                self.ref_traj_name = None
                self.ref_len = None
                self.x_ref = None
                self.u_ref = None
                self.t_ref = None
                self.last_x_ref = None
                self.x_initial_reached = False
                self.mpc_idx = 0
                self.land_override = False

            # If using offline GP set parameters - No corrections made
            if self.use_nn and self.correction_mode == "offline":
                self.nn_corr_i = np.zeros((self.n_mpc, len(self.nn_output_idx)))
            return self.quad_opt.set_reference(x_ref, u_ref)

        # If reference trajectory not received, pick current position as ref
        if (not self.ref_received):
            if self.payload:
                self.quad_mass_change_pub.publish(Float32(self.base_mass))
            if self.x_ref_prov is None:
                self.x_ref_prov = np.array(self.x)[np.newaxis, :]
                self.x_ref_prov[0, 7:] = 0 # Set velocity states to zero
                if self.x_ref_prov[0, 2] < 1:
                    self.x_ref_prov[0, 2] = 1
                self.u_ref_prov = np.array([[0, 0, 0, 0]])  
                rospy.loginfo("Selecting current position as provisional setpoint. %s"%str(self.x_ref_prov[0, :3]))
            x_ref = self.x_ref_prov
            u_ref = self.u_ref_prov
            # If using offline GP set parameters - No corrections made
            if self.use_nn and self.correction_mode == "offline":
                self.nn_corr_i = np.zeros((self.n_mpc, len(self.nn_output_idx)))
            return self.quad_opt.set_reference(x_ref, u_ref)
        
        # If reference exists then exit out of provisional hovering mode
        if (self.x_ref_prov is not None):
            self.x_ref_prov = None
            self.u_ref_prov = None
            self.ground_level = False
            self.no_more_ref = False
            rospy.loginfo("Abandoning provisional setpoint.")
            rospy.loginfo("Initial position target: %s"%str(self.x_ref[0, :3]))
        
        # Check if starting position of trajectory has been reached
        if (not self.x_initial_reached):
            if self.payload:
                self.quad_mass_change_pub.publish(Float32(self.base_mass))
            mask = [1] * 6 + [0] * 6
            if (quaternion_state_mse(np.array(self.x), self.x_ref[0, :], mask) < self.init_thr and not self.x_initial_reached): 
                # Initial Point reached
                self.x_initial_reached = True
                self.opt_dt = 0
                rospy.loginfo("Reached initial position of trajectory.")
                # Begin recording
                msg = Bool()
                msg.data = True
                self.record_pub.publish(msg)
                # Set reference to initial reference trajectory point
                x_ref = np.array([self.x_ref[0, :]])
                u_ref = self.u_ref[np.newaxis, 0, :]
            else:
                # Initial point not reached yet
                # Fly towards initial position of trajectory
                # x_ref = self.x_ref[np.newaxis, 0, :]
                x_ref = np.array([self.x_ref[0, :]])
                x_ref[0, 3] = 1
                x_ref[0, 4:] = 0
                u_ref = np.array([[0, 0, 0, 0]])
                dx = self.init_v * np.sign(self.x_ref[0, 0] - self.x[0])
                dy = self.init_v * np.sign(self.x_ref[0, 1] - self.x[1])
                dz = self.init_v * np.sign(self.x_ref[0, 2] - self.x[2])
                x_ref[0, 0] = min(self.x_ref[0, 0], self.x[0] + dx) if dx > 0 else max(self.x_ref[0, 0], self.x[0] + dx)
                x_ref[0, 1] = min(self.x_ref[0, 1], self.x[1] + dy) if dy > 0 else max(self.x_ref[0, 1], self.x[1] + dy)
                x_ref[0, 2] = min(self.x_ref[0, 2], self.x[2] + dz) if dz > 0 else max(self.x_ref[0, 2], self.x[2] + dz)
            
            # If with offline GP set params - No corrections made
            if self.use_nn and self.correction_mode == "offline":
                self.nn_corr_i = np.zeros((self.n_mpc, len(self.nn_output_idx)))
            return self.quad_opt.set_reference(x_ref, u_ref)
        
        # Executing Trajectory Tracking
        if (self.mpc_idx < self.ref_len):
            # Payload mass
            if self.payload:
                if self.mpc_idx in self.pickup_mass_idx:
                    self.payload_mass_pickedup = True
                    if self.use_groundtruth:
                        self.payload_mass_est = self.payload_mass_gt
                elif self.mpc_idx in self.drop_mass_idx:
                    self.payload_mass_pickedup = False
                    if self.use_groundtruth:
                        self.payload_mass_est = 0
                if self.payload_mass_pickedup:
                    self.quad_mass_change_pub.publish(Float32(self.base_mass + self.payload_mass_gt))
                else:
                    self.quad_mass_change_pub.publish(Float32(self.base_mass))
            # Trajectory tracking
            ref_traj = self.x_ref[self.mpc_idx:self.mpc_idx + self.n_mpc * self.control_freq_factor, :]
            ref_u = self.u_ref[self.mpc_idx:self.mpc_idx + self.n_mpc * self.control_freq_factor, :]

            # Indices for down-sampling the reference to number of MPC nodes
            downsample_ref_ind = np.arange(0, min(self.control_freq_factor * self.n_mpc, ref_traj.shape[0]),
                                           self.control_freq_factor, dtype=int)

            # Sparser references (same dt as node separation)
            x_ref = ref_traj[downsample_ref_ind, :]
            u_ref = ref_u[downsample_ref_ind, :]

            # If with offline GP retrieve GP corrections and set params
            if self.use_nn and self.correction_mode == "offline":
                ref_corr = self.nn_corr[self.mpc_idx:self.mpc_idx + self.n_mpc * self.control_freq_factor, :]
                self.nn_corr_i = ref_corr[downsample_ref_ind, :]

            self.mpc_idx += 1
            return self.quad_opt.set_reference(x_ref, u_ref)
        # End of reference reached
        elif (self.mpc_idx == self.ref_len):
            if self.payload:
                self.quad_mass_change_pub.publish(Float32(self.base_mass))
            # Compute optimization dt
            self.opt_dt /= self.mpc_idx
            self.opt_dt *= 1000
            rospy.loginfo("Tracking complete. Mean MPC opt. time: %.3f ms"%self.opt_dt)
            self.mpc_idx += 1
            # Lower drone to ground
            self.land_override = True
            rospy.loginfo("Landing...")

            # Stop recording
            self.x_initial_reached = False
            msg = Bool()
            msg.data = False
            self.record_pub.publish(msg)

            # Set reference to final position of trajectory
            x_ref = self.x_ref[np.newaxis, -1, :]
            x_ref[7:] = 0 # Set velocity states to zero
            u_ref = np.array([[0, 0, 0, 0]])
            
            # if offline GP set parameters
            if self.use_nn and self.correction_mode == "offline":
                self.nn_corr_i = np.zeros((self.n_mpc, len(self.nn_output_idx)))
            return self.quad_opt.set_reference(x_ref, u_ref)
        
    def state_est_callback(self, msg):
        """
        Callback function for State Estimation
        """        
        p, q, v, w = odometry_parse(msg)
        if self.env == "gazebo" and self.use_groundtruth:
            v_w = v_dot_q(np.array(v), np.array(q)).tolist()
        else:
            v_w = v
        self.x = p + q + v_w + w

        self.x_available = True
        
        def _mpc_thread_func():
            self.run_mpc()

        # We only optimize once every two odometry messages
        if not self.optimize_next:
            self.mpc_thread.join()

            # If currently on trajectory tracking, pay close attention to any skipped messages.
            if self.x_initial_reached:
                # Count how many messages were skipped (ideally 0)
                skipped_messages = int(msg.header.seq - self.last_x_seq_number - 1)
                if skipped_messages > 0:
                    rospy.logwarn("MPC Recording time skipped messages: %d" % skipped_messages)

                # Adjust current index in trajectory
                self.mpc_idx += divmod(skipped_messages, 2)[0]
                # If odd number of skipped messages, do optimization
                if skipped_messages > 0 and skipped_messages % 2 == 1:
                    # Run MPC now
                    self.mpc_thread = threading.Thread(target=_mpc_thread_func(), args=(), daemon=True)
                    self.mpc_thread.start()
                    self.last_x_seq_number = msg.header.seq
                    self.optimize_next = False
                    return

            self.optimize_next = True
            return

        if msg.header.seq > self.last_x_seq_number + 2 and self.x_initial_reached:
            # If one message was skipped at this point, then the reference is already late. Compensate by
            # optimizing twice in a row and hope to do it fast...
            self.mpc_thread = threading.Thread(target=_mpc_thread_func(), args=(), daemon=True)
            self.mpc_thread.start()
            self.optimize_next = True
            rospy.logwarn("Odometry skipped at Optimization step. Last: %d, current: %d", msg.header.seq, self.last_x_seq_number);
            self.last_x_seq_number = msg.header.seq
            return
        
        # Everything is Operating as it should
        self.mpc_thread = threading.Thread(target=_mpc_thread_func(), args=(), daemon=True)
        self.mpc_thread.start()

        self.last_x_seq_number = msg.header.seq
        self.optimize_next = False

    def run_mpc(self):
        if (not self.x_available):
            rospy.logwarn("States not available.")
            return
        
        # Set Reference Trajectory
        self.set_reference_trajectory()

        # optimize MPC
        try:
            if (self.quad_opt.solve_mpc(self.x, nn_corr=self.nn_corr_i, 
                                        payload_m=self.payload_mass_est) == 0):
                x_opt, u_opt = self.quad_opt.get_controls()
                self.opt_dt += self.quad_opt.get_opt_dt()
            else:
                rospy.logwarn("MPC Optimization was not sucessful.")
                return
        except RuntimeError as e:
            rospy.logwarn(f"MPC optimization failed with error: {str(e)}")
            # rospy.logwarn("Tried to run an MPC optimization but MPC is not ready yet.")
            return

        # Publish controls
        control_method = 'w'
        if self.env == "gazebo":
            control_cmd_msg = ControlCommand()
            control_cmd_msg.header = Header()
            control_cmd_msg.header.stamp = rospy.Time.now()
            control_cmd_msg.control_mode = 2
            control_cmd_msg.armed = True
            control_cmd_msg.bodyrates.x = x_opt[1, -3]
            control_cmd_msg.bodyrates.y = x_opt[1, -2]
            control_cmd_msg.bodyrates.z = x_opt[1, -1]
            collective_thrust = np.sum(u_opt[0]) * self.quad.get_max_thrust() / self.quad.get_mass()
            if self.ground_level and self.no_more_ref:
                collective_thrust *= 0.01
            control_cmd_msg.collective_thrust = collective_thrust
        elif self.env == "arena":
            control_cmd_msg = AttitudeTarget()
            control_cmd_msg.header = Header()
            control_cmd_msg.header.stamp = rospy.Time.now()
            if control_method == 'q':
                control_cmd_msg.orientation.w = x_opt[1, 3]
                control_cmd_msg.orientation.x = x_opt[1, 4]
                control_cmd_msg.orientation.y = x_opt[1, 5]
                control_cmd_msg.orientation.z = x_opt[1, 6]
                control_cmd_msg.type_mask = 7
            elif control_method == 'w':
                control_cmd_msg.body_rate.x = x_opt[1, -3]
                control_cmd_msg.body_rate.y = x_opt[1, -2]
                control_cmd_msg.body_rate.z = x_opt[1, -1]
                control_cmd_msg.type_mask = 128
            thrust = np.sum(u_opt[0])/4
            if self.ground_level and self.no_more_ref:
                thrust *= 0.5
            control_cmd_msg.thrust =  thrust
        self.control_pub.publish(control_cmd_msg)
        # Publish motor thrusts
        motor_thrust_msg = Actuators()
        motor_thrust_msg.header = Header()
        motor_thrust_msg.angular_velocities = u_opt[0]
        self.motor_thrust_pub.publish(motor_thrust_msg)
    
    def payload_mass_est_callback(self, msg):
        self.payload_mass_est = msg.data

def main():
    # Load parameters?

    MPCNode()

if __name__ == "__main__":
    main()
