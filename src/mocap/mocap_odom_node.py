#!/usr/bin/env python
""" ROS node for the data-augmented MPC, to use in the Gazebo simulator and real world experiments.

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
import numpy as np
from pykalman import KalmanFilter
import rospy
import threading
from geometry_msgs.msg import PoseStamped, Point, Quaternion, TwistStamped, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from collections import deque
from mavros_msgs.srv import CommandHome

def pose_parse(pose_msg):
    p = [pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z]
    q = [pose_msg.pose.orientation.w, pose_msg.pose.orientation.x, pose_msg.pose.orientation.y,
        pose_msg.pose.orientation.z]
    
    return p, q

def twist_parse(twist_msg):
    v = [twist_msg.twist.linear.x, twist_msg.twist.linear.y, twist_msg.twist.linear.z]
    w = [twist_msg.twist.angular.x*100, twist_msg.twist.angular.y*100, twist_msg.twist.angular.z*100]

    return v, w

def imu_parse(imu_msg):
    w = [imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z]
    a = [imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z]

    return w, a

def control_cmd_parse(cmd_msg):
    """
    Parser function to extract control commands from control command message.
    """
    u = cmd_msg.thrusts

    return u

def quaternion_inverse(q):
    return [q[0], -q[1], -q[2], -q[3]]

def quaternion_mult(q, r):
    return [r[0]*q[0]-r[1]*q[1]-r[2]*q[2]-r[3]*q[3],
            r[0]*q[1]+r[1]*q[0]-r[2]*q[3]+r[3]*q[2],
            r[0]*q[2]+r[1]*q[3]+r[2]*q[0]-r[3]*q[1],
            r[0]*q[3]-r[1]*q[2]+r[2]*q[1]+r[3]*q[0]]

def point_rotation_by_quaternion(point, q):
    r = [0]+point
    return quaternion_mult(quaternion_mult(q,r),quaternion_inverse(q))[1:]

class MocapOdomWrapper:
    def __init__(self, quad_name, odom_rate=100, simulate_noise=False, use_ekf=False):
        self.quad_name = quad_name

        # TODO: set up publishers and subscribers for arena evironment
        pose_topic = "/mocap/" + quad_name +"/pose"
        twist_topic = "/mocap/" + quad_name + "/twist"
        imu_topic = "/mavros/imu/data_raw"

        odom_topic = quad_name + "/ground_truth/odometry"


        # Publishers
        self.odom_pub = rospy.Publisher(odom_topic, Odometry, queue_size=10, tcp_nodelay=True)

        self.odom_pub_thread = threading.Thread()
        self.odom_pub_thread.start()

        # Subscribers
        self.pose_sub = rospy.Subscriber(pose_topic, PoseStamped, self.pose_callback, queue_size=10, tcp_nodelay=True)
        self.twist_sub = rospy.Subscriber(twist_topic, TwistStamped, self.twist_callback, queue_size=10, tcp_nodelay=True)
        self.imu_sub = rospy.Subscriber(imu_topic, Imu, self.imu_callback, queue_size=10, tcp_nodelay=True)
        
        # Rosservice
        self.set_home_srv = rospy.ServiceProxy('/mavros/cmd/set_home', CommandHome)

        # Set Home position
        response = self.set_home_srv(current_gps=False, latitude=0.0, longitude=0.0, altitude=0.6)
        # Check if the service call was successful
        if response.success:
            rospy.loginfo("Home position set to origin successfully.")
        else:
            rospy.logwarn("Failed to set home position.")

        self.seq_num = 0
        self.p, self.q, self.v, self.w = None, None, None, None
        
        mocap_hz = 100

        # Define the state-space model
        A = np.eye(13)  # state transition matrix
        for i in range(3):
            A[i, 7+i] = 1/mocap_hz
        H = np.eye(13)  # observation matrix

        # Define the noise covariances 
        Q = np.eye(13)*0.1  # process noise covariance
        R = np.diag([0.01, 0.01, 0.01, 
                     0.01, 0.01, 0.01, 0.01, 
                     0.05, 0.05, 0.05, 
                     0.05, 0.05, 0.05])  # measurement noise covariance

        # Initial state
        state_means = np.zeros(13)
        state_covs = np.eye(13)

        # Create the Kalman filter
        kf = KalmanFilter(transition_matrices=A, 
                          observation_matrices=H, 
                          transition_covariance=Q, 
                          observation_covariance=R,
                          initial_state_mean=state_means, 
                          initial_state_covariance=state_covs)
        # KF smooths signal over N number of sequence of measurements
        self.N = 1
        self.measurements = []
        rate = rospy.Rate(odom_rate)

        # Wait for measurements
        while self.p is None or self.v is None or self.w is None:
            rospy.sleep(1)

        rospy.loginfo("State Measurements Received")
        while not rospy.is_shutdown():
            # Rotate velocity measurement to body frame
            v_b = point_rotation_by_quaternion(self.v, quaternion_inverse(self.q))
            # Add measured states to list
            x = self.p + self.q + v_b + self.w
            self.seq_num += 1

            # Smooth signal
            if use_ekf:
                state_means, state_covs = kf.filter_update(state_means,
                                                        state_covs,
                                                        x)
            else:
                state_means = x

            quad_state_msg = Odometry()
            quad_state_msg.header.stamp = rospy.Time.now()
            quad_state_msg.header.seq = self.seq_num
            quad_state_msg.header.frame_id = 'odom'
            quad_state_msg.child_frame_id ="base_link"
            quad_state_msg.pose.pose.position = Point(*state_means[:3])
            w, x, y, z = state_means[3:7]
            quad_state_msg.pose.pose.orientation = Quaternion(x, y, z, w)
            quad_state_msg.twist.twist.linear = Vector3(*state_means[7:10])
            quad_state_msg.twist.twist.angular = Vector3(*state_means[10:])
            # quad_state_msg.acceleration.linear = Vector3(*[0, 0, 0])
            # quad_state_msg.motors = []

            def _odom_pub_thread_func():
                self.odom_pub.publish(quad_state_msg)

            self.odom_pub_thread = threading.Thread(target=_odom_pub_thread_func(), args=(), daemon=True)
            self.odom_pub_thread.start()

            # While rospy is not shutdownSpin 
            rate.sleep()

    def pose_callback(self, msg):
        """
        Callback function for the Pose Subscriber.
        :param msg: message from subscriber.
        :type msg: geometry_msgs/Pose
        """

        self.p, self.q = pose_parse(msg)

    def twist_callback(self, msg):
        """
        Callback function for the Twist Subscriber.
        :param msg: message from subscriber.
        :type msg: geometry_msgs/Twist
        """     
        # self.v, self.w = twist_parse(msg)
        self.v, _ = twist_parse(msg)
    
    def imu_callback(self, msg):
        """
        Call back function for the IMU Subscriber
        :param msg: message from subscriber.
        :type msg: sensor_msgs/Imu
        """
        self.w, _ = imu_parse(msg)
        # _, _ = imu_parse(msg)

    def command_callback(self, msg):
        self.motor_speeds = tuple(thrust/self.quad.max_thrust for thrust in control_cmd_parse(msg))


def main():
    rospy.init_node("mocap_odom")

    default_quad = "clara"

    quad_name = rospy.get_param('~quad_name', default=None)
    quad_name = quad_name if quad_name is not None else default_quad

    # Odometer Publishing Rate
    odom_rate = rospy.get_param('~odom_rate', default=100)

    # Simulate sensor noise switch boolean
    simulate_noise = rospy.get_param('~simulate_noise', default=False)
    
    # use ekf
    use_ekf = rospy.get_param('~use_ekf', default=False)
    
    MocapOdomWrapper(quad_name, odom_rate, simulate_noise=simulate_noise, use_ekf=use_ekf)


if __name__ == "__main__":
    main()
