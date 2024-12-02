import rospy
import threading
import numpy as np

from src.utils.DirectoryConfig import DirectoryConfig as DirConfig
from src.utils.utils import v_dot_q
from src.quad_opt.quad import custom_quad_param_loader
from src.quad_opt.quad_optimizer_mpc import QuadOptimizerMPC
from src.model_fitting.NeuralODE import load_model

from node_mpc_mhe.msg import ReferenceTrajectory
from mav_msgs.msg import Actuators
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Empty
from quadrotor_msgs.msg import ControlCommand
from mavros_msgs.msg import AttitudeTarget

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
            msg.data = not (self.x_ref is None and self.state_est_available)
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
        self.p = None
        self.q = None
        self.v = None
        self.w = None
        self.last_state_est_seq_number = 0
        
        # Initialize MPC Variables
        self.mpc_idx

        # Initialize MPC Thread
        self.mpc_thread = threading.Thread()
        self.mpc_thread.start()

    def init_publishers(self):
        '''
        '''
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
            self.x_features = rospy.get_param(ns + "x_features", default=None)
            self.y_features = rospy.get_param(ns + "y_features", default=None)
            self.nn_model = load_model(self.model_name)
        # Compile Acados Model
        self.quad_opt = QuadOptimizerMPC(self.quad, t_mpc=self.t_mpc, n_mpc=self.n_mpc,
                                    q_mpc=q_mpc, qt_factor=qt_factor, r_mpc=r_mpc, 
                                    model_name=self.quad_name,
                                    use_nn=self.use_nn, nn_model=self.nn_model,
                                    x_features=self.x_features, y_features=self.y_features)
    
    def state_est_callback(self, msg):
        """
        Callback function for State Estimation
        """        
        p, q, v, w = odometry_parse(msg)
        if self.environment == "gazebo" and self.use_groundtruth:
            v_w = v_dot_q(np.array(v), np.array(q)).tolist()
        else:
            v_w = v
        x_est = p + q + v_w + w

        self.x = x_est
        # If the above try passed then MPC is ready
        self.state_est_available = True
        
        def _mpc_thread_func():
            self.run_mpc(msg)

        # We only optimize once every two odometry messages
        if not self.optimize_next:
            self.mpc_thread.join()

            # If currently on trajectory tracking, pay close attention to any skipped messages.
            if self.x_initial_reached:
                # Count how many messages were skipped (ideally 0)
                skipped_messages = int(msg.header.seq - self.last_state_est_seq_number - 1)
                if skipped_messages > 0:
                    warn_msg = "MPC Recording time skipped messages: %d" % skipped_messages
                    rospy.logwarn(warn_msg)

                # Adjust current index in trajectory
                self.mpc_idx += divmod(skipped_messages, 2)[0]
                # If odd number of skipped messages, do optimization
                if skipped_messages > 0 and skipped_messages % 2 == 1:

                    if self.recording_options["recording"]:
                        self.check_out_initial_state(msg,)

                    # Run MPC now
                    self.mpc_thread = threading.Thread(target=_mpc_thread_func(), args=(), daemon=True)
                    self.mpc_thread.start()
                    self.last_state_est_seq_number = msg.header.seq
                    self.optimize_next = False
                    return

            self.optimize_next = True
            if self.recording_options["recording"] and self.x_initial_reached:
                self.check_out_initial_state(msg)
            return

        # Run MPC
        if msg.header.seq > self.last_state_est_seq_number + 2 and self.x_initial_reached:
            # If one message was skipped at this point, then the reference is already late. Compensate by
            # optimizing twice in a row and hope to do it fast...
            if self.recording_options["recording"] and self.x_initial_reached:
                self.check_out_initial_state(msg)
            self.mpc_thread = threading.Thread(target=_mpc_thread_func(), args=(), daemon=True)
            self.mpc_thread.start()
            self.optimize_next = True
            rospy.logwarn("Odometry skipped at Optimization step. Last: %d, current: %d", msg.header.seq, self.last_state_est_seq_number);

            self.last_state_est_seq_number = msg.header.seq
            return
        
        # Everything is Operating as it should
        self.mpc_thread = threading.Thread(target=_mpc_thread_func(), args=(), daemon=True)
        self.mpc_thread.start()

        self.last_state_est_seq_number = msg.header.seq
        self.optimize_next = False