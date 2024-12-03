import rospy
import threading
import numpy as np

from src.utils.DirectoryConfig import DirectoryConfig as DirConfig
from src.utils.utils import v_dot_q, quaternion_state_mse, features_to_idx
from src.quad_opt.quad import custom_quad_param_loader
from src.quad_opt.quad_optimizer_mpc import QuadOptimizerMPC
from src.model_fitting.NeuralODE import load_model

from hybrid_mpc_mhe.msg import ReferenceTrajectory
from mav_msgs.msg import Actuators
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Empty, Header
from quadrotor_msgs.msg import ControlCommand
from mavros_msgs.msg import AttitudeTarget
from mavros_msgs.srv import CommandBool, SetMode

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
        while not rospy.is_shutdown():
            # Publish if MPC is busy with a current trajectory
            msg = Bool()
            msg.data = not (self.x_ref is None and self.x_available)
            self.status_pub.publish(msg)
            rate.sleep()
        return
    
    def init_params(self):
        ns = rospy.get_namespace()
        # System environment
        self.environment = rospy.get_param("~environment", default="gazebo")
        self.quad_name = rospy.get_param("~quad_name", default=None)
        assert self.quad_name != None
        self.use_groundtruth = rospy.get_param("~use_groundtruth", default=True)

        # Initial flight parameters
        self.init_thr = rospy.get_param(ns + "init_thr", default=0.5)
        self.init_v = rospy.get_param(ns + "init_v", default=0.3)

        # Landing Parameters
        self.land_thr = rospy.get_param(ns + "land_thr", default=0.05)
        self.land_z = rospy.get_param(ns + "land_z", default=0.05)
        self.land_dz = rospy.get_param(ns + "land_dz", default=0.1)

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
        ns = rospy.get_namespace()
        # Publisher topic names
        control_topic = rospy.get_param("~control_topic", default="/" + self.quad_name + "/autopilot/control_command_input")
        motor_thrust_topic = rospy.get_param("~motor_thrust_topic", default="/" + self.quad_name + "/motor_thrust")
        record_topic = rospy.get_param("~motor_thrust", default="/" + self.quad_name + "/motor_thrust")
        status_topic = rospy.get_param("~mpc_status_topic", default="/mpc/busy")
        # control_gz_topic = rospy.get_param("~control_gz_topic", default="/" + self.quad_name + "/autopilot/control_command_input")

        # Publishers
        if self.env == "gazebo":
            self.control_pub = rospy.Publisher(control_topic, ControlCommand, queue_size=1, tcp_nodelay=True)
        elif self.env == "arena":
            self.control_pub = rospy.Publisher(control_topic, AttitudeTarget, queue_size=1, tcp_nodelay=True)
        self.motor_thrust_pub = rospy.Publisher(motor_thrust_topic, Actuators, queue_size=1, tcp_nodelay=True)
        self.record_pub = rospy.Publisher(record_topic, Bool, queue_size=1, tcp_nodelay=True)
        self.status_pub = rospy.Publisher(status_topic, Bool, queue_size=1, tcp_nodelay=True)
        
    def init_rosservice(self):
        set_mode_srvc = rospy.get_param("~set_mode_srvc", default="/mavros/set_mode")
        arming_srvc = rospy.get_param("~arming_srvc", default="/mavros/cmd/arming")
        self.set_mode_client = rospy.ServiceProxy(set_mode_srvc, SetMode)
        self.arming_client = rospy.ServiceProxy(arming_srvc, CommandBool)

    def init_subscribers(self):
        '''
        '''
        ns = rospy.get_namespace()

        # Subscriber topic names
        ref_topic = rospy.get_param("~ref_topic", default="/reference")
        state_est_topic = rospy.get_param("~state_est_topic", default= "/" + self.quad_name + "/state_est")
        odom_topic = rospy.get_param("~odom_topic", default="/" + self.quad_name + "/ground_truth/odometry")
        land_topic = rospy.get_param("~land", default="/" + self.quad_name + "/land")

        # Subscribers        
        self.ref_sub = rospy.Subscriber(ref_topic, ReferenceTrajectory, self.reference_callback)
        if self.use_groundtruth:
            self.state_est_sub = rospy.Subscriber(state_est_topic, Odometry, self.state_est_callback, queue_size=1, tcp_nodelay=True)
        else:
            self.state_est_sub = rospy.Subscriber(odom_topic, Odometry, self.state_est_callback, queue_size=1, tcp_nodelay=True)
        self.land_sub = rospy.Subscriber(land_topic, Empty, self.land_callback)

    def init_mpc(self):
        ns = rospy.get_namespace()
        # MPC Parameters
        self.control_freq_factor = rospy.get_param(ns + "control_freq_factor", default=5)
        self.use_nn = rospy.get_params(ns + "use_nn", default=False)
        self.n_mpc = rospy.get_param(ns + 'n_mpc', default=10)
        self.t_mpc = rospy.get_param(ns + 't_mpc', default=1)

        # MPC Costs
        q_p = np.ones((1,3)) * rospy.get_param(ns + 'q_p', default=35)
        q_q = np.ones((1,3)) * rospy.get_param(ns + 'q_q', default=25)
        q_v = np.ones((1,3)) * rospy.get_param(ns + 'q_v', default=10)
        q_r = np.ones((1,3)) * rospy.get_param(ns + 'q_r', default=10)

        qt_factor = rospy.get_param(ns + "qt_factor", default=1)

        q_mpc = np.squeeze(np.hstack((q_p, q_q, q_v, q_r)))
        r_mpc = np.array([1.0, 1.0, 1.0, 1.0]) * rospy.get_param(ns + "r_mpc", default=0.1)

        # Load Quad Instance
        self.quad = custom_quad_param_loader(self.quad_name)

        # Load NN models
        if (self.use_nn):
            self.model_name = rospy.get_param(ns + "model_name", default=None)
            self.model_type = rospy.get_param(ns + "model_type", default=None)
            self.input_features = rospy.get_param(ns + "input_features", default=None)
            self.nn_input_idx = features_to_idx(self.input_features)
            self.output_features = rospy.get_param(ns + "output_features", default=None)
            self.nn_output_idx = features_to_idx(self.output_features)
            self.correction_mode = rospy.get_param(ns + "correction_mode", default="online")
            self.nn_model = load_model(self.model_name)
        else:
            self.model_name = None
            self.model_type = None
            self.input_features = None
            self.output_features = None
            self.nn_model = None
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
        # Compile Acados Model
        self.quad_opt = QuadOptimizerMPC(self.quad, t_mpc=self.t_mpc, n_mpc=self.n_mpc,
                                         q_mpc=q_mpc, qt_factor=qt_factor, r_mpc=r_mpc, 
                                         quad_name=self.quad_name,
                                         use_nn=self.use_nn, nn_params=self.nn_params)
    
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

            if self.t_ref:
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
            x_ref = self.last_x_ref if self.last_x_ref is not None else self.x.copy()
            dz = np.sign(self.land_z - self.x[2]) * self.land_dz
            x_ref[2] = min(self.land_z, self.x[2] + dz) if dz > 0 else max(self.land_z, self.x[2] + dz)
            u_ref = self.last_u_ref if self.last_u_ref is not None else np.array([0, 0, 0, 0])

            # TODO: Disarming drone
            # Reached landing heigh
            if (abs(self.x[2] - self.land_z) < self.land_thr):
                if (not self.ground_level):
                    rospy.loginfo("Vehicle at Ground Level")
                    self.ground_level = True
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

            # TODO: If using offline GP set parameters

            return self.quad_opt.set_reference(x_ref, u_ref)

        # If reference trajectory not received, pick current position as ref
        if (not self.ref_received):
            if self.x_ref_prov is None:
                self.x_ref_prov = self.x
                self.x_ref_prov[7:] = 0 # Set velocity states to zero
                self.u_ref_prov = np.array([0, 0, 0, 0])  
                rospy.loginfo("Selecting current position as provisional setpoint.")
            x_ref = self.x_ref_prov
            u_ref = self.u_ref_prov
            # TODO: If using offline GP set parameters

            return self.quad_opt.set_reference(x_ref, u_ref)
        
        # If reference exists then exit out of provisional hovering mode
        if (self.x_ref_prov is not None):
            self.x_ref_prov = None
            self.u_ref_prov = None
            self.ground_level = False
            rospy.loginfo("Abandoning provisional setpoint.")
        
        # Check if starting position of trajectory has been reached
        if (not self.x_initial_reached):
            mask = [1] * 9 + [0] * 3
            if (quaternion_state_mse(np.array(self.x), self.x_ref[0, :], mask) < self.init_thr): 
                # Initial Point reached
                self.x_initial_reached = True
                self.opt_dt = 0
                rospy.loginfo("Reached initial position of trajectory.")
                # Begin recording
                msg = Bool()
                msg.data = True
                self.record_pub.publish(msg)
                # Set reference to initial reference trajectory point
                x_ref = self.x_ref[0, :]
                u_ref = self.u_ref[0, :]
            else:
                # Initial point not reached yet
                # Fly towards initial position of trajectory
                x_ref = self.x_ref[0, :]
                u_ref = np.array([0, 0, 0, 0])
                dx = self.init_v * np.sign(self.x_ref[0, 0] - self.x[0])
                dy = self.init_v * np.sign(self.x_ref[0, 1] - self.x[1])
                dz = self.init_v * np.sign(self.x_ref[0, 2] - self.x[2])
                x_ref[0] = min(self.x_ref[0, 0], self.x[0] + dx) if dx > 0 else max(self.x_ref[0, 0], self.x[0] + dx)
                x_ref[1] = min(self.x_ref[0, 1], self.x[1] + dy) if dy > 0 else max(self.x_ref[0, 1], self.x[1] + dy)
                x_ref[2] = min(self.x_ref[0, 2], self.x[2] + dz) if dz > 0 else max(self.x_ref[0, 2], self.x[2] + dz)
            
            # TODO: If with offline GP set params

            return self.quad_opt.set_reference(x_ref, u_ref)
        
        # Executing Trajectory Tracking
        if (self.mpc_idx < self.ref_len):
            # Trajectory tracking
            ref_traj = self.x_ref[self.mpc_idx:self.mpc_idx + self.n_mpc * self.control_freq_factor, :]
            ref_u = self.u_ref[self.mpc_idx:self.mpc_idx + self.n_mpc * self.control_freq_factor, :]

            # Indices for down-sampling the reference to number of MPC nodes
            downsample_ref_ind = np.arange(0, min(self.control_freq_factor * self.n_mpc, ref_traj.shape[0]),
                                           self.control_freq_factor, dtype=int)

            # Sparser references (same dt as node separation)
            x_ref = ref_traj[downsample_ref_ind, :]
            u_ref = ref_u[downsample_ref_ind, :]

            # TODO: If with offline GP retrieve GP corrections and set params

            self.mpc_idx += 1
            return self.quad_opt.set_reference(x_ref, u_ref)
        # End of reference reached
        elif (self.mpc_idx == self.ref_len):
            # Compute optimization dt
            self.opt_dt /= self.mpc_idx
            rospy.loginfo("Tracking complete. Mean MPC opt. time: %.3f ms"%self.opt_dt *1000)
            # Lower drone to ground
            self.land_override = True
            rospy.loginfo("Landing...")

            # Stop recording
            self.x_initial_reached = False
            msg = Bool()
            msg.data = False
            self.record_pub.publish(msg)

            # Set reference to final position of trajectory
            x_ref = self.x_ref[-1, :]
            self.x_ref[7:] = 0 # Set velocity states to zero
            self.u_ref = np.array([0, 0, 0, 0])
            # TODO: if offline GP set parameters
            return self.quad_opt.set_reference(x_ref, u_ref)
        
    def state_est_callback(self, msg):
        """
        Callback function for State Estimation
        """        
        p, q, v, w = odometry_parse(msg)
        if self.environment == "gazebo" and self.use_groundtruth:
            v_w = v_dot_q(np.array(v), np.array(q)).tolist()
        else:
            v_w = v
        self.x = p + q + v_w + w

        # TODO: Change to x_available
        self.x_available = True
        
        def _mpc_thread_func():
            self.run_mpc(msg)

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
            if (self.quad_opt.solve_mpc(self.x)):
                x_opt, u_opt = self.quad_opt.get_controls()
                self.opt_dt += self.quad_opt.get_opt_dt()
            else:
                rospy.logwarn("MPC Optimization was not sucessful.")
        except RuntimeError as e:
            rospy.logwarn(f"MPC optimization failed with error: {str(e)}")
            # rospy.logwarn("Tried to run an MPC optimization but MPC is not ready yet.")

        # Publish controls
        control_method = 'w'
        if self.environment == "gazebo":
            control_cmd_msg = ControlCommand()
            control_cmd_msg.header = Header()
            control_cmd_msg.header.stamp = rospy.Time.now()
            control_cmd_msg.control_mode = 2
            control_cmd_msg.armed = True
            control_cmd_msg.bodyrates.x = x_opt[1, -3]
            control_cmd_msg.bodyrates.y = x_opt[1, -2]
            control_cmd_msg.bodyrates.z = x_opt[1, -1]
            collective_thrust = np.sum(u_opt[:4]) * self.quad.max_thrust / self.quad.mass
            if self.ground_level:
                collective_thrust *= 0.01
            control_cmd_msg.collective_thrust = collective_thrust
        elif self.environment == "arena":
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
            thrust = np.sum(u_opt[:4])/4
            if self.ground_level:
                thrust *= 0.5
            control_cmd_msg.thrust =  thrust
        self.control_pub.publish(control_cmd_msg)
        # Publish motor thrusts
        motor_thrust_msg = Actuators()
        motor_thrust_msg.header = Header()
        motor_thrust_msg.angular_velocities = u_opt[:4]
        self.motor_thrust_pub.publish(motor_thrust_msg)

def main():
    rospy.init_node("mpc")

    # Load parameters?

    MPCNode()

if __name__ == "__main__":
    main()
