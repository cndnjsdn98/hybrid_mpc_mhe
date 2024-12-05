import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared

def draw_poly(traj, u_traj, t, target_points=None, target_t=None):
    """
    Plots the generated trajectory of length n with the used keypoints.
    :param traj: Full generated reference trajectory. Numpy array of shape nx13
    :param u_traj: Generated reference inputs. Numpy array of shape nx4
    :param t: Timestamps of the references. Numpy array of length n
    :param target_points: m position keypoints used for trajectory generation. Numpy array of shape 3 x m.
    :param target_t: Timestamps of the reference position keypoints. If not passed, then they are extracted from the
    t vector, assuming constant time separation.
    """

    ders = 2
    dims = 3

    y_labels = [r'pos $[m]$', r'vel $[m/s]$', r'acc $[m/s^2]$', r'jer $[m/s^3]$']
    dim_legends = ['x', 'y', 'z']

    if target_t is None and target_points is not None:
        target_t = np.linspace(0, t[-1], target_points.shape[1])

    p_traj = traj[:, :3]
    a_traj = traj[:, 3:7]
    v_traj = traj[:, 7:10]
    r_traj = traj[:, 10:]

    plt_traj = [p_traj, v_traj]

    fig = plt.figure()
    for d_ord in range(ders):

        plt.subplot(ders + 2, 2, d_ord * 2 + 1)

        for dim in range(dims):

            plt.plot(t, plt_traj[d_ord][:, dim], label=dim_legends[dim])

            if d_ord == 0 and target_points is not None:
                plt.plot(target_t, target_points[dim, :], 'bo')

        plt.gca().set_xticklabels([])
        plt.legend()
        plt.grid()
        plt.ylabel(y_labels[d_ord])

    dim_legends = [['w', 'x', 'y', 'z'], ['x', 'y', 'z']]
    y_labels = [r'att $[quat]$', r'rate $[rad/s]$']
    plt_traj = [a_traj, r_traj]
    for d_ord in range(ders):

        plt.subplot(ders + 2, 2, d_ord * 2 + 1 + ders * 2)
        for dim in range(plt_traj[d_ord].shape[1]):
            plt.plot(t, plt_traj[d_ord][:, dim], label=dim_legends[d_ord][dim])

        plt.legend()
        plt.grid()
        plt.ylabel(y_labels[d_ord])
        if d_ord == ders - 1:
            plt.xlabel(r'time $[s]$')
        else:
            plt.gca().set_xticklabels([])

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    plt.plot(p_traj[:, 0], p_traj[:, 1], p_traj[:, 2])
    if target_points is not None:
        plt.plot(target_points[0, :], target_points[1, :], target_points[2, :], 'bo')
    plt.title('Target position trajectory')
    ax.set_xlabel(r'$p_x [m]$')
    ax.set_ylabel(r'$p_y [m]$')
    ax.set_zlabel(r'$p_z [m]$')

    plt.subplot(ders + 1, 2, (ders + 1) * 2)
    for i in range(u_traj.shape[1]):
        plt.plot(t, u_traj[:, i], label=r'$u_{}$'.format(i))
    plt.grid()
    plt.legend()
    plt.gca().yaxis.set_label_position("right")
    plt.gca().yaxis.tick_right()
    plt.xlabel(r'time $[s]$')
    plt.ylabel(r'single thrusts $[N]$')
    plt.title('Control inputs')

    plt.suptitle('Generated polynomial trajectory')

    plt.show()


def get_full_traj(poly_coeffs, target_dt, int_dt):

    dims = poly_coeffs.shape[-1]
    full_traj = np.zeros((4, dims, 0))
    t_total = np.zeros((0,))

    if isinstance(target_dt, float):
        # Adjust target_dt to make it divisible by int_dt
        target_dt = round(target_dt / int_dt) * int_dt

        # Assign target time for each keypoint using homogeneous spacing
        t_vec = np.arange(0, target_dt * (poly_coeffs.shape[0] + 1) - 1e-5, target_dt)

    else:
        # The time between each pair of points is assigned independently
        # First, also adjust each value of the target_dt vector to make it divisible by int_dt
        for i, dt in enumerate(target_dt):
            target_dt[i] = round(dt / int_dt) * int_dt

        t_vec = np.append(np.zeros(1), np.cumsum(target_dt[:-1]))

    for seg in range(len(t_vec) - 1):

        # Select time sampling (linear or quadratic) mode
        tau_dt = np.arange(t_vec[seg], t_vec[seg + 1] + 1e-5, int_dt)

        # Re-normalize time sampling vector between -1 and 1
        t1 = (tau_dt - t_vec[seg]) / (t_vec[seg + 1] - t_vec[seg]) * 2 - 1

        # Compression ratio
        compress = 2 / np.diff(t_vec)[seg]

        # Integrate current segment of trajectory
        traj = np.zeros((4, dims, len(t1)))

        for der_order in range(4):
            for i in range(dims):
                traj[der_order, i, :] = np.polyval(np.polyder(poly_coeffs[seg, :, i], der_order), t1) * (compress ** der_order)

        if seg < len(t_vec) - 2:
            # Remove last sample (will be the initial point of next segment)
            traj = traj[:, :, :-1]
            t_seg = tau_dt[:-1]
        else:
            t_seg = tau_dt

        full_traj = np.concatenate((full_traj, traj), axis=-1)
        t_total = np.concatenate((t_total, t_seg))

    # Separate into p_xyz and yaw trajectories
    yaw_traj = full_traj[:, -1, :]
    full_traj = full_traj[:, :-1, :]

    return full_traj, yaw_traj, t_total


def fit_multi_segment_polynomial_trajectory(p_targets, yaw_targets):

    p_targets = np.concatenate((p_targets, yaw_targets[np.newaxis, :]), 0)
    m = multiple_waypoints(p_targets.shape[1] - 1)

    dims = p_targets.shape[0]
    n_segments = p_targets.shape[1]

    poly_coefficients = np.zeros((n_segments - 1, 8, dims))
    for dim in range(dims):
        b = rhs_generation(p_targets[dim, :])
        poly_coefficients[:, :, dim] = np.fliplr(np.linalg.solve(m, b).reshape(n_segments - 1, 8))

    return poly_coefficients


def matrix_generation(ts):
    b = np.array([[1, ts,  ts**2, ts**3,    ts**4,    ts**5,     ts**6,     ts**7],
                  [0, 1, 2*ts,  3*ts**2,  4*ts**3,  5*ts**4,   6*ts**5,   7*ts**6],
                  [0, 0, 2,     6*ts,    12*ts**2, 20*ts**3,  30*ts**4,  42*ts**5],
                  [0, 0, 0,     6,       24*ts,    60*ts**2, 120*ts**3, 210*ts**4],
                  [0, 0, 0,     0,       24,      120*ts,    360*ts**2, 840*ts**3],
                  [0, 0, 0,     0,       0,       120,       720*ts,   2520*ts**2],
                  [0, 0, 0,     0,       0,       0,         720,      5040*ts],
                  [0, 0, 0,     0,       0,       0,         0,        5040]])

    return b


def multiple_waypoints(n_segments):

    m = np.zeros((8 * n_segments, 8 * n_segments))

    for i in range(n_segments):

        if i == 0:

            # initial condition of the first curve
            b = matrix_generation(-1.0)
            m[8 * i:8 * i + 4, 8 * i:8 * i + 8] = b[:4, :]

            # intermediary condition of the first curve
            b = matrix_generation(1.0)
            m[8 * i + 4:8 * i + 7 + 4, 8 * i:8 * i + 8] = b[:-1, :]

            # starting condition of the second curve position and derivatives
            b = matrix_generation(-1.0)
            m[8 * i + 4 + 1:8 * i + 4 + 7, 8 * (i + 1):8 * (i + 1) + 8] = -b[1:-1, :]
            m[8 * i + 4 + 7:8 * i + 4 + 8, 8 * (i + 1):8 * (i + 1) + 8] = b[0, :]

        elif i != n_segments - 1:

            # starting condition of the ith curve position and derivatives
            b = matrix_generation(1.0)
            m[8 * i + 4:8 * i + 7 + 4, 8 * i:8 * i + 8] = b[:-1, :]

            # end condition of the ith curve position and derivatives
            b = matrix_generation(-1.0)
            m[8 * i + 4 + 1:8 * i + 4 + 7, 8 * (i + 1):8 * (i + 1) + 8] = -b[1:-1, :]
            m[8 * i + 4 + 7:8 * i + 4 + 8, 8 * (i + 1):8 * (i + 1) + 8] = b[0, :]

        if i == n_segments - 1:
            # end condition of the final curve position and derivatives (4 boundary conditions)
            b = matrix_generation(1.0)
            m[8 * i + 4:8 * i + 4 + 4, 8 * i:8 * i + 8] = b[:4, :]

    return m


def fit_single_segment(p_start, p_end, v_start=None, v_end=None, a_start=None, a_end=None, j_start=None, j_end=None):

    if v_start is None:
        v_start = np.array([0, 0])
    if v_end is None:
        v_end = np.array([0, 0])
    if a_start is None:
        a_start = np.array([0, 0])
    if a_end is None:
        a_end = np.array([0, 0])
    if j_start is None:
        j_start = np.array([0, 0])
    if j_end is None:
        j_end = np.array([0, 0])

    poly_coefficients = np.zeros((8, len(p_start)))

    tf = 1
    ti = -1
    A = np.array(([
        [1 * tf ** 7,   1 * tf ** 6,   1 * tf ** 5,   1 * tf ** 4,   1 * tf ** 3,  1 * tf ** 2,  1 * tf ** 1,  1],
        [7 * tf ** 6,   6 * tf ** 5,   5 * tf ** 4,   4 * tf ** 3,   3 * tf ** 2,  2 * tf ** 1,  1,            0],
        [42 * tf ** 5,  30 * tf ** 4,  20 * tf ** 3,  12 * tf ** 2,  6 * tf ** 1,  2,            0,            0],
        [210 * tf ** 4, 120 * tf ** 3, 60 * tf ** 2,  24 * tf ** 1,  6,            0,            0,            0],
        [1 * ti ** 7,   1 * ti ** 6,   1 * ti ** 5,   1 * ti ** 4,   1 * ti ** 3,  1 * ti ** 2,  1 * ti ** 1,  1],
        [7 * ti ** 6,   6 * ti ** 5,   5 * ti ** 4,   4 * ti ** 3,   3 * ti ** 2,  2 * ti ** 1,  1,            0],
        [42 * ti ** 5,  30 * ti ** 4,  20 * ti ** 3,  12 * ti ** 2,  6 * ti ** 1,  2,            0,            0],
        [210 * ti ** 4, 120 * ti ** 3, 60 * ti ** 2,  24 * ti ** 1,  6,            0,            0,            0]]))

    A = np.tile(A[:, :, np.newaxis], (1, 1, len(p_start)))

    b = np.concatenate((p_end, v_end, a_end, j_end, p_start, v_start, a_start, j_start)).reshape(8, -1)

    for i in range(len(p_start)):
        poly_coefficients[:, i] = np.linalg.inv(A[:, :, i]).dot(np.array(b[:, i]))

    return np.expand_dims(poly_coefficients, 0)


def rhs_generation(x):
    n = x.shape[0] - 1

    big_x = np.zeros((8 * n))
    big_x[:4] = np.array([x[0], 0, 0, 0]).T
    big_x[-4:] = np.array([x[-1], 0, 0, 0]).T

    for i in range(1, n):
        big_x[8 * (i - 1) + 4:8 * (i - 1) + 8 + 4] = np.array([x[i], 0, 0, 0, 0, 0, 0, x[i]]).T

    return big_x

def random_periodical_trajectory(plot=False, random_state=None, map_limits=None):

    if random_state is None:
        random_state = np.random.randint(0, 9999)

    kernel_z = ExpSineSquared(length_scale=5.5, periodicity=60)
    kernel_y = ExpSineSquared(length_scale=4.5, periodicity=30) + ExpSineSquared(length_scale=4.0, periodicity=15)
    kernel_x = ExpSineSquared(length_scale=4.5, periodicity=30) + ExpSineSquared(length_scale=4.5, periodicity=60)

    gp_x = GaussianProcessRegressor(kernel=kernel_x)
    gp_y = GaussianProcessRegressor(kernel=kernel_y)
    gp_z = GaussianProcessRegressor(kernel=kernel_z)

    # High resolution sampling for track boundaries
    inputs_x = np.linspace(0, 60, 100)
    inputs_y = np.linspace(0, 30, 100)
    inputs_z = np.linspace(0, 60, 100)

    x_sample_hr = gp_x.sample_y(inputs_x[:, np.newaxis], 1, random_state=random_state)
    y_sample_hr = gp_y.sample_y(inputs_y[:, np.newaxis], 1, random_state=random_state)
    z_sample_hr = gp_z.sample_y(inputs_z[:, np.newaxis], 1, random_state=random_state)

    max_x_coords = np.max(x_sample_hr, 0)
    max_y_coords = np.max(y_sample_hr, 0)
    max_z_coords = np.max(z_sample_hr, 0)

    min_x_coords = np.min(x_sample_hr, 0)
    min_y_coords = np.min(y_sample_hr, 0)
    min_z_coords = np.min(z_sample_hr, 0)

    x_sample_hr, y_sample_hr, z_sample_hr = center_and_scale(
        x_sample_hr, y_sample_hr, z_sample_hr,
        max_x_coords, min_x_coords, max_y_coords, min_y_coords, max_z_coords, min_z_coords)

    # Additional constraint on map limits
    if map_limits is not None:
        x_sample_hr, y_sample_hr, z_sample_hr = apply_map_limits(x_sample_hr, y_sample_hr, z_sample_hr, map_limits)

    # Low resolution for control points
    lr_ind = np.linspace(0, len(x_sample_hr) - 1, 10, dtype=int)
    lr_ind[-1] = 0
    x_sample_lr = x_sample_hr[lr_ind, :]
    y_sample_lr = y_sample_hr[lr_ind, :]
    z_sample_lr = z_sample_hr[lr_ind, :]

    x_sample_diff = x_sample_hr[lr_ind + 1, :] - x_sample_lr
    y_sample_diff = y_sample_hr[lr_ind + 1, :] - y_sample_lr
    z_sample_diff = z_sample_hr[lr_ind + 1, :] - z_sample_lr

    # Get angles at keypoints
    a_x = np.arctan2(z_sample_diff, y_sample_diff) * 0
    a_y = np.arctan2(x_sample_diff, z_sample_diff) * 0
    a_z = (np.arctan2(y_sample_diff, x_sample_diff) - np.pi/4) * 0

    if plot:
        # Plot checking
        # Build rotation matrices
        rx = [np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]) for a in a_x[:, 0]]
        ry = [np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]) for a in a_y[:, 0]]
        rz = [np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]) for a in a_z[:, 0]]

        main_axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        quiver_axes = np.zeros((len(lr_ind), 3, 3))

        for i in range(len(lr_ind)):
            r_mat = rz[i].dot(ry[i].dot(rx[i]))
            rot_body = r_mat.dot(main_axes)
            quiver_axes[i, :, :] = rot_body

        shortest_axis = min(np.max(x_sample_hr) - np.min(x_sample_hr),
                            np.max(y_sample_hr) - np.min(y_sample_hr),
                            np.max(z_sample_hr) - np.min(z_sample_hr))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_sample_lr, y_sample_lr, z_sample_lr)
        ax.plot(x_sample_hr[:, 0], y_sample_hr[:, 0], z_sample_hr[:, 0], '-', alpha=0.5)
        ax.quiver(x_sample_lr[:, 0], y_sample_lr[:, 0], z_sample_lr[:, 0],
                  x_sample_diff[:, 0], y_sample_diff[:, 0], z_sample_diff[:, 0], color='g',
                  length=shortest_axis / 6, normalize=True, label='traj. norm')
        ax.quiver(x_sample_lr, y_sample_lr, z_sample_lr,
                  quiver_axes[:, 0, :], quiver_axes[:, 1, :], quiver_axes[:, 2, :], color='b',
                  length=shortest_axis / 6, normalize=True, label='quad. att.')
        ax.tick_params(labelsize=16)
        ax.legend(fontsize=16)
        ax.set_xlabel('x [m]', size=16, labelpad=10)
        ax.set_ylabel('y [m]', size=16, labelpad=10)
        ax.set_zlabel('z [m]', size=16, labelpad=10)
        ax.set_xlim([np.min(x_sample_hr), np.max(x_sample_hr)])
        ax.set_ylim([np.min(y_sample_hr), np.max(y_sample_hr)])
        ax.set_zlim([np.min(z_sample_hr), np.max(z_sample_hr)])
        ax.set_title('Source keypoints', size=18)
        plt.show()

    curve = np.concatenate((x_sample_lr, y_sample_lr, z_sample_lr), 1)
    attitude = np.concatenate((a_x, a_y, a_z), 1)

    return curve, attitude


def apply_map_limits(x, y, z, limits):

    # Find out which axis is the most constrained
    x_max_range = limits["x"][1] - limits["x"][0]
    y_max_range = limits["y"][1] - limits["y"][0]
    z_max_range = limits["z"][1] - limits["z"][0]

    x_actual_range = np.max(x) - np.min(x)
    y_actual_range = np.max(y) - np.min(y)
    z_actual_range = np.max(z) - np.min(z)

    # One or more of the ranges violates the constraints.
    if x_actual_range > x_max_range or y_actual_range > y_max_range or z_actual_range > z_max_range:
        shrink_ratio = max(x_actual_range / x_max_range, y_actual_range / y_max_range, z_actual_range / z_max_range)
        x = (x - np.mean(x)) / shrink_ratio
        y = (y - np.mean(y)) / shrink_ratio
        z = (z - np.mean(z)) / shrink_ratio

        x += (limits["x"][0] - np.min(x))
        y += (limits["y"][0] - np.min(y))
        z += (limits["z"][0] - np.min(z))

    return x, y, z


def center_and_scale(x, y, z, x_max, x_min, y_max, y_min, z_max, z_min):

    x -= (x_min + (x_max - x_min) / 2)
    y -= (y_min + (y_max - y_min) / 2)
    z -= (z_min + (z_max - z_min) / 2)

    scaling = np.mean([x_max - x_min, y_max - y_min, z_max - z_min])
    x = x * 6 / scaling
    y = y * 6 / scaling
    z = z * 6 / scaling

    return x, y, z
