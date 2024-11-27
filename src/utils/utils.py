""" Miscellaneous utility functions.

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
import errno
import shutil
import numpy as np
import casadi as cs
import xml.etree.ElementTree as XMLtree
import pyquaternion
from scipy.interpolate.interpolate import interp1d

def safe_mkdir_recursive(directory, overwrite=False):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(directory):
                pass
            else:
                raise
    else:
        if overwrite:
            try:
                shutil.rmtree(directory)
            except:
                print('Error while removing directory: {0}'.format(directory))

def vw_to_vb(x):
    """
    Applies inverse rotation on the system states to return the 
    velocity states to be in Body Frame.
    :param x: 13-length array of quadrotor states 
              where velocity is in World frame. [p, q, v_w, w] 
    :return xb: 13-length array of quadrotor states 
               where velocity is in Body frame
    """
    xb = np.copy(x)
    q = x[3:7]
    vw = x[7:10]
    vb = v_dot_q(vw, quaternion_inverse(q))
    xb[7:10] = vb
    return xb

def world_to_body_velocity_mapping(state_sequence):
    """

    :param state_sequence: N x 13 state array, where N is the number of states in the sequence.
    :return: An N x 13 sequence of states, but where the velocities (assumed to be in positions 7, 8, 9) have been
    rotated from world to body frame. The rotation is made using the quaternion in positions 3, 4, 5, 6.
    """

    p, q, v_w, w = separate_variables(state_sequence)
    v_b = []
    for i in range(len(q)):
        v_b.append(v_dot_q(v_w[i], quaternion_inverse(q[i])))
    v_b = np.stack(v_b)
    return np.concatenate((p, q, v_b, w), 1)


def v_dot_q(v, q):
    rot_mat = q_to_rot_mat(q)
    if isinstance(q, np.ndarray):
        return rot_mat.dot(v)

    return cs.mtimes(rot_mat, v)

def q_dot_q(q, r):
    """
    Applies the rotation of quaternion r to quaternion q. In order words, rotates quaternion q by r. Quaternion format:
    wxyz.

    :param q: 4-length numpy array or CasADi MX. Initial rotation
    :param r: 4-length numpy array or CasADi MX. Applied rotation
    :return: The quaternion q rotated by r, with the same format as in the input.
    """

    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    rw, rx, ry, rz = r[0], r[1], r[2], r[3]

    t0 = rw * qw - rx * qx - ry * qy - rz * qz
    t1 = rw * qx + rx * qw - ry * qz + rz * qy
    t2 = rw * qy + rx * qz + ry * qw - rz * qx
    t3 = rw * qz - rx * qy + ry * qx + rz * qw

    if isinstance(q, np.ndarray):
        return np.array([t0, t1, t2, t3])
    else:
        return cs.vertcat(t0, t1, t2, t3)

def q_to_rot_mat(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    if isinstance(q, np.ndarray):
        rot_mat = np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]])

    else:
        rot_mat = cs.vertcat(
            cs.horzcat(1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
            cs.horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)),
            cs.horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)))

    return rot_mat

def skew_symmetric(v):
    """
    Computes the skew-symmetric matrix of a 3D vector (PAMPC version)

    :param v: 3D numpy vector or CasADi MX
    :return: the corresponding skew-symmetric matrix of v with the same data type as v
    """

    if isinstance(v, np.ndarray):
        return np.array([[0, -v[0], -v[1], -v[2]],
                         [v[0], 0, v[2], -v[1]],
                         [v[1], -v[2], 0, v[0]],
                         [v[2], v[1], -v[0], 0]])

    return cs.vertcat(
        cs.horzcat(0, -v[0], -v[1], -v[2]),
        cs.horzcat(v[0], 0, v[2], -v[1]),
        cs.horzcat(v[1], -v[2], 0, v[0]),
        cs.horzcat(v[2], v[1], -v[0], 0))

def quaternion_inverse(q):
    w, x, y, z = q[0], q[1], q[2], q[3]

    if isinstance(q, np.ndarray):
        return np.array([w, -x, -y, -z])
    else:
        return cs.vertcat(w, -x, -y, -z)

def quaternion_to_euler(q):
    q = pyquaternion.Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
    yaw, pitch, roll = q.yaw_pitch_roll
    return [roll, pitch, yaw]

def rotation_matrix_to_quat(rot):
    """
    Calculate a quaternion from a 3x3 rotation matrix.

    :param rot: 3x3 numpy array, representing a valid rotation matrix
    :return: a quaternion corresponding to the 3D rotation described by the input matrix. Quaternion format: wxyz
    """

    q = pyquaternion.Quaternion(matrix=rot)
    return np.array([q.w, q.x, q.y, q.z])


def undo_quaternion_flip(q_past, q_current):
    """
    Detects if q_current generated a quaternion jump and corrects it. Requires knowledge of the previous quaternion
    in the series, q_past
    :param q_past: 4-dimensional vector representing a quaternion in wxyz form.
    :param q_current: 4-dimensional vector representing a quaternion in wxyz form. Will be corrected if it generates
    a flip wrt q_past.
    :return: q_current with the flip removed if necessary
    """

    if np.sqrt(np.sum((q_past - q_current) ** 2)) > np.sqrt(np.sum((q_past + q_current) ** 2)):
        return -q_current
    return q_current


def separate_variables(traj):
    """
    Reshapes a trajectory into expected format.

    :param traj: N x 13 array representing the reference trajectory
    :return: A list with the components: Nx3 position trajectory array, Nx4 quaternion trajectory array, Nx3 velocity
    trajectory array, Nx3 body rate trajectory array
    """

    p_traj = traj[:, :3]
    q_traj = traj[:, 3:7]
    v_traj = traj[:, 7:10]
    r_traj = traj[:, 10:]
    
    return [p_traj, q_traj, v_traj, r_traj]


def unwrap(p):
    # for i in range(len(p)-1):
    #     if (p[i]-p[i+1]) > (2*np.pi-0.5):
    #         p[i+1:] = p[i+1:] + (2*np.pi)
    #     elif (p[i+1]-p[i]) < -(2*np.pi-0.5):
    #         p[i+1:] = p[i+1:] - (2*np.pi)
    # Check for NaN or infinite values
    if np.any(np.isnan(p)) or np.any(np.isinf(p)):
        raise ValueError("Input contains NaN or infinite values.")
    
    dp = np.diff(p)
    dps = np.mod(dp+np.pi, 2*np.pi)-np.pi
    dps_loc = np.where((dps==-np.pi) & (dp>0))
    dps[dps_loc] = np.pi
    dp_corr = dps-dp
    dp_corr_loc = np.where(abs(dp)<np.pi)
    dp_corr[dp_corr_loc] = 0
    p[1:] = p[1:] + np.cumsum(dp_corr)
    return p

def parse_xacro_file(xacro):
    """
    Reads a .xacro file describing a robot for Gazebo and returns a dictionary with its properties.
    :param xacro: full path of .xacro file to read
    :return: a dictionary of robot attributes
    """

    tree = XMLtree.parse(xacro)

    attrib_dict = {}

    for node in tree.getroot().getchildren():
        # Get attributes
        attributes = node.attrib

        if 'value' in attributes.keys():
            attrib_dict[attributes['name']] = attributes['value']

        if node.getchildren():
            try:
                attrib_dict[attributes['name']] = [child.attrib for child in node.getchildren()]
            except:
                continue

    return attrib_dict

def rmse(t_1, x_1, t_2, x_2, n_interp_samples=4000):
    if np.all(t_1 == t_2):
        return np.mean(np.sqrt(np.sum((x_1 - x_2) ** 2, axis=1)))

    assert x_1.shape[1] == x_2.shape[1]

    if t_1[0] != 0:
        t_1 -= t_1[0]

    if t_2[0] != 0:
        t_2 -= t_2[0]

    # Find duplicates
    t_1, idx_1 = np.unique(t_1, return_index=True)
    x_1 = x_1[idx_1, :]
    t_2, idx_2 = np.unique(t_2, return_index=True)
    x_2 = x_2[idx_2, :]

    t_min = max(t_1[0], t_2[0])
    t_max = min(t_1[-1], t_2[-1])

    t_interp = np.linspace(t_min, t_max, n_interp_samples)
    err = np.zeros((n_interp_samples, x_1.shape[1]))
    for dim in range(x_1.shape[1]):
        x1_interp = interp1d(np.squeeze(t_1), np.squeeze(x_1[:, dim]), kind='cubic', bounds_error=False, fill_value="extrapolate")
        x2_interp = interp1d(np.squeeze(t_2), np.squeeze(x_2[:, dim]), kind='cubic', bounds_error=False, fill_value="extrapolate")

        x1_sample = x1_interp(t_interp)
        x2_sample = x2_interp(t_interp)

        err[:, dim] = x1_sample - x2_sample

    return np.mean(np.sqrt(np.sum(err ** 2, axis=1)))
