import casadi as cs
import numpy as np
import matplotlib.pyplot as plt
from src.utils.utils import separate_variables, quaternion_inverse, \
                            quaternion_to_euler, unwrap, q_dot_q, v_dot_q, \
                            world_to_body_velocity_mapping

def trajectory_tracking_results(img_save_dir, t_ref, t_executed, x_ref, x_executed, u_ref, u_executed, mpc_error,
                                w_control=None, legend_labels=None,
                                quat_error=True, file_type='png'):
    if legend_labels is None:
        legend_labels = ['reference', 'executed']

    with_ref = True if x_ref is not None else False
    plt.switch_backend('Agg')
    fig, ax = plt.subplots(3, 4, sharex='all', figsize=(24, 16))

    SMALL_SIZE = 14
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 20

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    n_tref = len(t_ref)
    n_texec = len(t_executed)
    labels = ['x', 'y', 'z']
    for i in range(3):
        if with_ref:
            ax[i, 0].plot(t_ref, x_ref[:n_tref, i], label=legend_labels[0])
        ax[i, 0].plot(t_executed, x_executed[:n_texec, i], label=legend_labels[1])
        ax[i, 0].legend()
        ax[i, 0].set_ylabel(labels[i])
    ax[0, 0].set_title(r'$p\:[m]$')
    ax[2, 0].set_xlabel(r'$t [s]$')

    q_euler = np.stack([quaternion_to_euler(x_executed[j, 3:7]) for j in range(x_executed.shape[0])])
    for i in range(3):
        ax[i, 1].plot(t_ref, q_euler[:n_tref, i], label=legend_labels[1])
    if with_ref:
        ref_euler = np.stack([quaternion_to_euler(x_ref[j, 3:7]) for j in range(x_ref.shape[0])])
        q_err = []
        for i in range(t_ref.shape[0]):
            q_err.append(q_dot_q(x_executed[i, 3:7], quaternion_inverse(x_ref[i, 3:7])))
        q_err = np.stack(q_err)

        for i in range(3):
            ax[i, 1].plot(t_ref, ref_euler[:n_tref, i], label=legend_labels[0])
            if quat_error:
                ax[i, 1].plot(t_ref, q_err[:n_tref, i + 1], label='quat error')
        ax[i, 1].legend()
    ax[0, 1].set_title(r'$\theta\:[rad]$')
    ax[2, 1].set_xlabel(r'$t [s]$')

    for i in range(3):
        if with_ref:
            ax[i, 2].plot(t_ref, x_ref[:n_tref, i + 7], label=legend_labels[0])
        ax[i, 2].plot(t_executed, x_executed[:n_texec, i + 7], label=legend_labels[1])
        ax[i, 2].legend()
    ax[0, 2].set_title(r'$v\:[m/s]$')
    ax[2, 2].set_xlabel(r'$t [s]$')

    for i in range(3):
        ax[i, 3].plot(t_executed, x_executed[:n_texec, i + 10], label=legend_labels[1])
        if with_ref:
            ax[i, 3].plot(t_ref, x_ref[:n_tref, i + 10], label=legend_labels[0])
        if w_control is not None:
            ax[i, 3].plot(t_ref, w_control[:n_tref, i], label='control')
        ax[i, 3].legend()
    ax[0, 3].set_title(r'$\omega\:[rad/s]$')
    ax[2, 3].set_xlabel(r'$t [s]$')

    fig.savefig(img_save_dir + '/tracking_error.'+file_type, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', format=file_type,
                transparent=False, bbox_inches=None, pad_inches=0.05,
                metadata=None)
    plt.close(fig)
    
    fig, ax = plt.subplots(3, 1, sharex="all", sharey="all", figsize=(22,15))
    for i in range(3):
        ax[i].plot(t_executed, x_executed[:n_texec, i + 10], label=legend_labels[1])
        if with_ref:
            ax[i].plot(t_ref, x_ref[:n_tref, i + 10], label=legend_labels[0])
        if w_control is not None:
            ax[i].plot(t_ref, w_control[:n_tref, i], label='control ref')
        ax[i].legend()
    ax[0].set_title(r'$\omega\:[rad/s]$')
    ax[2].set_xlabel(r'$t [s]$')
    fig.savefig(img_save_dir + '/angular_velocity_control_result.'+file_type, dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', format=file_type,
                    transparent=False, bbox_inches=None, pad_inches=0.05,
                    metadata=None)
    plt.close(fig)

    if u_ref is not None and u_executed is not None:
        fig, ax = plt.subplots(1, 4, sharex="all", sharey="all", figsize=(15,10))
        for i in range(4):
            ax[i].plot(t_ref, u_ref[:n_tref, i], label='ref', linewidth=0.75)
            ax[i].plot(t_executed, u_executed[:n_texec, i], label='executed', linewidth=0.75)
            ax[i].set_xlabel(r'$t [s]$')
            tit = 'Control %d' % (i + 1)
            ax[i].set_title(tit)
            ax[i].legend()
            # ax[i].set_ylim(bottom=0, top=1)
        fig.savefig(img_save_dir + '/inputs.'+file_type, dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', format=file_type,
                    transparent=False, bbox_inches=None, pad_inches=0.05,
                    metadata=None)
        plt.close(fig)

    fig, ax = plt.subplots(3, 1, sharex='all', figsize=(13, 14))
    for i in range(3):
        if with_ref:
            ax[i].plot(t_ref, x_ref[:n_tref, i], label=legend_labels[0])
        ax[i].plot(t_executed, x_executed[:n_texec, i], label=legend_labels[1])
        tit = 'P_' + labels[i]
        ax[i].set_ylabel(tit)
        ax[i].legend()
        ax[i].grid()
    ax[0].set_title(r'$p\:[m]$')
    ax[2].set_xlabel(r'$t [s]$')
    plt.tight_layout()
    fig.savefig(img_save_dir + '/position_tracking_error.'+file_type, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', format=file_type,
                transparent=False, bbox_inches=None, metadata=None)
    plt.close(fig)
    
    fig, ax = plt.subplots(3, 1, sharex='all', figsize=(13, 14))
    for i in range(3):
        if with_ref:
            ax[i].plot(t_ref, ref_euler[:n_tref, i], label=legend_labels[0])
        ax[i].plot(t_ref, q_euler[:n_tref, i], label=legend_labels[1])
        tit = 'q_' + labels[i]
        ax[i].set_ylabel(tit)
        ax[i].legend()
        ax[i].grid()
    ax[0].set_title(r'$\theta\:[rad]$')
    ax[2].set_xlabel(r'$t [s]$')
    plt.tight_layout()
    fig.savefig(img_save_dir + '/angular_position_tracking_error.'+file_type, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', format=file_type,
                transparent=False, bbox_inches=None, metadata=None)
    plt.close(fig)

    fig, ax = plt.subplots(3, 1, sharex='all', figsize=(13, 14))
    for i in range(3):
        if with_ref:
            ax[i].plot(t_ref, x_ref[:n_tref, i+7], label=legend_labels[0])
        ax[i].plot(t_executed, x_executed[:n_texec, i+7], label=legend_labels[1])
        tit = 'V_' + labels[i]
        ax[i].set_ylabel(tit)
        ax[i].legend()
        ax[i].grid()
    ax[0].set_title(r'$v\:[m]$')
    ax[2].set_xlabel(r'$t [s]$')
    plt.tight_layout()
    fig.savefig(img_save_dir + '/velocity_tracking_error.'+file_type, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', format=file_type,
                transparent=False, bbox_inches=None, metadata=None)
    plt.close(fig)

    fig, ax = plt.subplots(3, 1, sharex='all', figsize=(13, 14))
    for i in range(3):
        if with_ref:
            ax[i].plot(t_ref, x_ref[:n_tref, i+10], label=legend_labels[0])
        ax[i].plot(t_executed, x_executed[:n_texec, i+10], label=legend_labels[1])
        if w_control is not None:
            ax[i].plot(t_ref, w_control[:n_tref, i], label='control')
        tit = 'dq_' + labels[i]
        ax[i].set_ylabel(tit)
        ax[i].legend()
        ax[i].grid()
    ax[0].set_title(r'$\omega\:[rad/s]$')
    ax[2].set_xlabel(r'$t [s]$')
    plt.tight_layout()
    fig.savefig(img_save_dir + '/angular_velocity_tracking_error.'+file_type, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', format=file_type,
                transparent=False, bbox_inches=None, metadata=None)
    plt.close(fig)

    fig = plt.figure(figsize=(15,7))
    ax = plt.axes()
    plt.plot(x_ref[:, 0], x_ref[:, 1], label=legend_labels[0], linewidth=0.8)
    plt.plot(x_executed[:, 0], x_executed[:, 1], label=legend_labels[1], linewidth=0.8)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)
    plt.grid()
    ax.set_xlabel(r'$x [m]$')
    ax.set_ylabel(r'$y [m]$')
    # plt.ylim((-3.1, 2.6))
    # plt.xlim((-10.7, 0.5))
    plt.tight_layout()
    fig.savefig(img_save_dir + '/px_py_tracking_error.'+file_type, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', format=file_type,
                transparent=False, bbox_inches=None, metadata=None)
    plt.close(fig)

    fig = plt.figure(figsize=(13,10))
    ax = plt.axes(projection='3d')
    plt.plot(x_ref[:, 0], x_ref[:, 1], x_ref[:, 2], label=legend_labels[0], linewidth=0.7)
    plt.plot(x_executed[:, 0], x_executed[:, 1], x_executed[:, 2,], label=legend_labels[1], linewidth=0.7)
    plt.legend()
    plt.grid()
    ax.set_xlabel(r'$x [m]$')
    ax.set_ylabel(r'$y [m]$')
    ax.set_zlabel(r'$z [m]$')
    fig.savefig(img_save_dir + '/px_py_pz_tracking_error.'+file_type, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', format=file_type,
                transparent=False, bbox_inches=None, metadata=None)
    plt.close(fig)

    x_executed_B = world_to_body_velocity_mapping(x_executed)
    fig, ax = plt.subplots(3, 1, figsize=(13, 14))
    for i in range(3):
        ax[i].plot(x_executed_B[:, i+7], mpc_error[:, i+7], 'o', label=legend_labels[1])
        tit = 'dv_' + labels[i]
        ax[i].set_ylabel(tit)
        ax[i].legend()
        ax[i].grid()
    ax[0].set_title(r'$Model Error$')
    ax[2].set_xlabel(r'$v [m/s]$')
    plt.tight_layout()
    fig.savefig(img_save_dir + '/model_error.'+file_type, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', format=file_type,
                transparent=False, bbox_inches=None, metadata=None)
    plt.close(fig)


def state_estimation_results(img_save_dir, t_act, x_act, t_est, x_est, t_meas, y_measured, 
                             t_meas_noisy, y_measured_noisy, mhe_error, t_acc_est=None, accel_est=None, 
                             model_corr=None, model_corr_features=[], file_type='png',
                             show_error=False, show_dvdt=False, a_thrust=None,
                             a_est_b=None, a_meas_b=None):
    plt.switch_backend('Agg')
    SMALL_SIZE = 14
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 20

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    labels = ['x', 'y', 'z']
    n_test = len(t_est)
    n_tact = len(t_act)
    n_tmeas = len(t_meas)
    fig, ax = plt.subplots(3, 1, sharex='all', figsize=(13, 14))
    for i in range(3):
        ax[i].plot(t_est, x_est[:n_test, i], label="estimated", zorder=3)
        ax[i].plot(t_act, x_act[:n_tact, i], label="actual", zorder=2)
        ax[i].plot(t_meas_noisy, y_measured_noisy[:n_tmeas, i], label="measurement", zorder=1, alpha=0.5)
        tit = 'P_' + labels[i]
        ax[i].set_ylabel(tit)
        ax[i].legend()
        ax[i].grid()
    ax[0].set_title(r'$p\:[m]$')
    ax[2].set_xlabel(r'$t [s]$')
    plt.tight_layout()
    fig.savefig(img_save_dir + '/position_state_estimation.'+file_type, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', format=file_type,
                transparent=False, bbox_inches=None, metadata=None)
    plt.close(fig)

    q_euler_est = np.stack([quaternion_to_euler(x_est[j, 3:7]) for j in range(x_est.shape[0])])
    q_euler_est[:, 0] = np.rad2deg(unwrap(q_euler_est[:, 0]))
    q_euler_est[:, 1] = np.rad2deg(unwrap(q_euler_est[:, 1]))
    q_euler_est[:, 2] = np.rad2deg(unwrap(q_euler_est[:, 2]))
    q_euler_act = np.stack([quaternion_to_euler(x_act[j, 3:7]) for j in range(x_act.shape[0])])
    q_euler_act[:, 0] = np.rad2deg(unwrap(q_euler_act[:, 0]))
    q_euler_act[:, 1] = np.rad2deg(unwrap(q_euler_act[:, 1]))
    q_euler_act[:, 2] = np.rad2deg(unwrap(q_euler_act[:, 2]))
    fig, ax = plt.subplots(3, 1, sharex='all', figsize=(13, 14))
    for i in range(3):
        ax[i].plot(t_est, q_euler_est[:n_test, i], label="estimated", zorder=3)
        ax[i].plot(t_act, q_euler_act[:n_tact, i], label="actual", zorder=2)
        tit = 'theta_' + labels[i] + ' estimation'
        ax[i].set_ylabel(tit)
        ax[i].legend()
        ax[i].grid()
    ax[0].set_title(r'$\theta\:[deg]$')
    ax[2].set_xlabel(r'$t [s]$')
    plt.tight_layout()
    fig.savefig(img_save_dir + '/angular_position_state_estimation.'+file_type, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', format=file_type,
                transparent=False, bbox_inches=None, metadata=None)
    plt.close(fig)

    quart_labels = ['w', 'x', 'y', 'z']
    fig, ax = plt.subplots(4, 1, sharex='all', figsize=(13, 14))
    for i in range(4):
        ax[i].plot(t_est, x_est[:n_test, i+3], label="estimated", zorder=3)
        ax[i].plot(t_act, x_act[:n_tact, i+3], label="actual", zorder=2)
        tit = 'q' + quart_labels[i] + ' estimation'
        ax[i].set_ylabel(tit)
        ax[i].legend()
        ax[i].grid()
    # ax[0].set_ylim([0.85, 1.01])
    ax[0].set_title(r'quaternion')
    ax[3].set_xlabel(r'$t [s]$')
    plt.tight_layout()
    fig.savefig(img_save_dir + '/quaternion_state_estimation.'+file_type, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', format=file_type,
                transparent=False, bbox_inches=None, metadata=None)
    plt.close(fig)
    
    fig, ax = plt.subplots(3, 1, sharex='all', figsize=(13, 14))
    for i in range(3):
        ax[i].plot(t_est, x_est[:n_test, i+7], label="estimated", zorder=3)
        ax[i].plot(t_act, x_act[:n_tact, i+7], label="actual", zorder=2)
        tit = 'v_' + labels[i] + ' estimation error'
        ax[i].set_ylabel(tit)
        ax[i].legend()
        ax[i].grid()
    ax[0].set_title(r'$p\:[m]$')
    ax[2].set_xlabel(r'$t [s]$')
    plt.tight_layout()
    fig.savefig(img_save_dir + '/velocity_state_estimation.'+file_type, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', format=file_type,
                transparent=False, bbox_inches=None, metadata=None)
    plt.close(fig)

    fig, ax = plt.subplots(3, 1, sharex='all', figsize=(13, 14))
    xb_est = world_to_body_velocity_mapping(x_est)
    xb_act = world_to_body_velocity_mapping(x_act)
    for i in range(3):
        ax[i].plot(t_est, xb_est[:n_test, i+7], label="estimated", zorder=3)
        ax[i].plot(t_act, xb_act[:n_tact, i+7], label="actual", zorder=2)
        tit = 'v_' + labels[i] + ' estimation error'
        ax[i].set_ylabel(tit)
        ax[i].legend()
        ax[i].grid()
    ax[0].set_title(r'$p\:[m]$')
    ax[2].set_xlabel(r'$t [s]$')
    plt.tight_layout()
    fig.savefig(img_save_dir + '/velocity_Body_state_estimation.'+file_type, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', format=file_type,
                transparent=False, bbox_inches=None, metadata=None)
    plt.close(fig)

    fig, ax = plt.subplots(3, 1, sharex='all', figsize=(13, 14))
    for i in range(3):
        ax[i].plot(t_est, x_est[:n_test, i+10], label="estimated", zorder=3)
        ax[i].plot(t_act, x_act[:n_tact, i+10], label="actual", zorder=2)
        # ax[i].plot(t_meas, y_measured[:, i+3], label="measurement", zorder=2)
        ax[i].plot(t_meas_noisy, y_measured_noisy[:n_tmeas, i+3], label="measurement", zorder=1, alpha=0.5)
        tit = 'w_' + labels[i] + ' estimation error'
        ax[i].set_ylabel(tit)
        ax[i].legend()
        ax[i].grid()
    ax[0].set_title(r'$p\:[m]$')
    ax[2].set_xlabel(r'$t [s]$')
    plt.tight_layout()
    fig.savefig(img_save_dir + '/angular_velocity_state_estimation.'+file_type, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', format=file_type,
                transparent=False, bbox_inches=None, metadata=None)
    plt.close(fig)

    fig = plt.figure(figsize=(15,7))
    ax = plt.axes()
    plt.plot(x_est[:, 0], x_est[:, 1], label="estimation", linewidth=0.8)
    plt.plot(x_act[:, 0], x_act[:, 1], label="actual", linewidth=0.8)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)
    plt.grid()
    ax.set_xlabel(r'$x [m]$')
    ax.set_ylabel(r'$y [m]$')
    plt.tight_layout()
    fig.savefig(img_save_dir + '/state_estimation_px_py.'+file_type, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', format=file_type,
                transparent=False, bbox_inches=None, metadata=None)
    plt.close(fig)

    fig = plt.figure(figsize=(13,10))
    ax = plt.axes(projection='3d')
    plt.plot(x_est[:, 0], x_est[:, 1], x_est[:, 2], label="estimation", linewidth=0.7)
    plt.plot(x_act[:, 0], x_act[:, 1], x_act[:, 2,], label="actual", linewidth=0.7)
    plt.legend()
    plt.grid()
    ax.set_xlabel(r'$x [m]$')
    ax.set_ylabel(r'$y [m]$')
    ax.set_zlabel(r'$z [m]$')
    fig.savefig(img_save_dir + '/state_estimation_px_py_pz.'+file_type, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', format=file_type,
                transparent=False, bbox_inches=None, metadata=None)
    plt.close(fig)

    if show_error:
        fig, ax = plt.subplots(3, 1, sharex='all', figsize=(13, 14))
        for i in range(3):
            ax[i].plot(t_est, x_est[:n_test, i] - x_act[:n_test, i])
            tit = 'p_' + labels[i] + ' estimation error'
            ax[i].set_ylabel(tit)
            ax[i].grid()
        ax[0].set_title(r'$p\:[m]$')
        ax[2].set_xlabel(r'$t [s]$')
        plt.tight_layout()
        fig.savefig(img_save_dir + '/position_state_estimation_error.'+file_type, dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', format=file_type,
                    transparent=False, bbox_inches=None, metadata=None)
        plt.close(fig)

        fig, ax = plt.subplots(3, 1, sharex='all', figsize=(13, 14))
        for i in range(3):
            ax[i].plot(t_est, q_euler_est[:n_test, i] - q_euler_act[:n_test, i])
            tit = 'q_' + labels[i] + ' estimation error'
            ax[i].set_ylabel(tit)
            ax[i].grid()
        ax[0].set_title(r'$p\:[m]$')
        ax[2].set_xlabel(r'$t [s]$')
        plt.tight_layout()
        fig.savefig(img_save_dir + '/angular_position_state_estimation_error.'+file_type, dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', format=file_type,
                    transparent=False, bbox_inches=None, metadata=None)
        plt.close(fig)

        fig, ax = plt.subplots(3, 1, sharex='all', figsize=(13, 14))
        for i in range(3):
            ax[i].plot(t_est, x_est[:n_test, i+7] - x_act[:n_test, i+7])
            tit = 'v_' + labels[i] + ' estimation error'
            ax[i].set_ylabel(tit)
            ax[i].grid()
        ax[0].set_title(r'$p\:[m]$')
        ax[2].set_xlabel(r'$t [s]$')
        plt.tight_layout()
        fig.savefig(img_save_dir + '/velocity_state_estimation_error.'+file_type, dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', format=file_type,
                    transparent=False, bbox_inches=None, metadata=None)
        plt.close(fig)

        fig, ax = plt.subplots(3, 1, sharex='all', figsize=(13, 14))
        for i in range(3):
            ax[i].plot(t_est, x_est[:n_test, i+10] - x_act[:n_test, i+10])
            tit = 'w_' + labels[i] + ' estimation error'
            ax[i].set_ylabel(tit)
            ax[i].grid()
        ax[0].set_title(r'$p\:[m]$')
        ax[2].set_xlabel(r'$t [s]$')
        plt.tight_layout()
        fig.savefig(img_save_dir + '/angular_velocity_state_estimation_error.'+file_type, dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', format=file_type,
                    transparent=False, bbox_inches=None, metadata=None)
        plt.close(fig)

    if accel_est is not None:
        n_taccest = len(t_acc_est)
        fig, ax = plt.subplots(3, 1, sharex='all', figsize=(13, 14))
        for i in range(3):
            p1, = ax[i].plot(t_meas, y_measured[:n_tmeas, i+6], label='actual', color='C1', zorder=2)
            p2, = ax[i].plot(t_acc_est, accel_est[:n_taccest, i], label='mhe est', color='C0', zorder=3)
            p3, = ax[i].plot(t_meas_noisy, y_measured_noisy[:n_tmeas, i+6], label="measurement", color='C2', zorder=1, alpha=0.5)

            tit = 'a_' + labels[i] + ' measurement'
            # lns = [p2, p1]
            ax[i].legend()
            ax[i].set_ylabel(tit)
            ax[i].grid()
        ax[0].set_title(r'$a\:[m/s^2]$')
        ax[2].set_xlabel(r'$t [s]$')
        plt.tight_layout()
        fig.savefig(img_save_dir + '/acceleration_estimation.'+file_type, dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', format=file_type,
                    transparent=False, bbox_inches=None, metadata=None)
        plt.close(fig)

    if a_thrust is not None:
        fig, ax = plt.subplots(3, 1, sharex='all', figsize=(13, 14))
        for i in range(3):
            p1, = ax[i].plot(t_meas, y_measured[:n_tmeas, i+6], label='gt', color='C1', zorder=2)
            p3, = ax[i].plot(t_meas, y_measured_noisy[:n_tmeas, i+6], label='measured', color='C2', zorder=1, alpha=0.5)
            p2, = ax[i].plot(t_meas, a_thrust[:n_tmeas, i], label='thrust command', color='C0', zorder=3)
            tit = 'a_' + labels[i] + ' measurement'
            ax[i].legend()
            ax[i].set_ylabel(tit)
            ax[i].grid()
        ax[0].set_title(r'$a\:[m/s^2]$')
        ax[2].set_xlabel(r'$t [s]$')
        plt.tight_layout()
        fig.savefig(img_save_dir + '/thrust_estimation.'+file_type, dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', format=file_type,
                    transparent=False, bbox_inches=None, metadata=None)
        plt.close(fig)

    if a_est_b is not None and a_meas_b is not None:
        fig, ax = plt.subplots(3, 1, sharex='all', figsize=(13, 14))
        for i in range(3):
            _, = ax[i].plot(t_meas, a_est_b[:n_tmeas, i], label='model', color='C0', zorder=3)
            _, = ax[i].plot(t_meas, a_meas_b[:n_tmeas, i], label='measured', color='C1', zorder=1, alpha=0.75)
            tit = 'a_' + labels[i] + ' measurement'
            ax[i].legend()
            ax[i].set_ylabel(tit)
            ax[i].grid()
        ax[0].set_title(r'$a\:[m/s^2]$')
        ax[2].set_xlabel(r'$t [s]$')
        plt.tight_layout()
        fig.savefig(img_save_dir + '/compare_model_imu_no_gravity.'+file_type, dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', format=file_type,
                    transparent=False, bbox_inches=None, metadata=None)
        plt.close(fig)

    fig, ax = plt.subplots(3, 1, figsize=(13, 14))
    for i in range(3):
        ax[i].scatter(y_measured_noisy[:, i+6], mhe_error[:, i+7])
        tit = 'v_' + labels[i] + ' estimation error'
        ax[i].set_ylabel(tit)
        ax[i].grid()
    ax[0].set_title(r'model error')
    ax[2].set_xlabel(r'$acceleration [m/s^2]$')
    plt.tight_layout()
    fig.savefig(img_save_dir + '/mhe_error_imu_meas_regression.'+file_type, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', format=file_type,
                transparent=False, bbox_inches=None, metadata=None)
    plt.close(fig)   

    fig, ax = plt.subplots(3, 1, figsize=(13, 14))
    for i in range(3):
        ax[i].scatter(t_est, mhe_error[:, i+7])
        tit = 'v_' + labels[i] + ' estimation error'
        ax[i].set_ylabel(tit)
        ax[i].grid()
    ax[0].set_title(r'$p\:[m]$')
    ax[2].set_xlabel(r'$t [s]$')
    plt.tight_layout()
    fig.savefig(img_save_dir + '/mhe_error.'+file_type, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', format=file_type,
                transparent=False, bbox_inches=None, metadata=None)
    plt.close(fig)   

    if model_corr is not None:
        feature_idx = 0
        if 'q' in model_corr_features:
            fig, ax = plt.subplots(4, 1, figsize=(13, 14))
            for i in range(4):
                ax[i].plot(t_est, model_corr[:n_test, i+feature_idx], label="est")
                ax[i].plot(t_act, mhe_error[:n_tact, i+3], label="act")
                tit = 'q_' + quart_labels[i] + ' estimated model error'
                ax[i].set_ylabel(tit)
                ax[i].grid()
                ax[i].legend()
            ax[0].set_title(r'$Estimated Model Error$')
            ax[2].set_xlabel(r'$t [s]$')
            plt.tight_layout()
            fig.savefig(img_save_dir + '/q_estimated_model_error.'+file_type, dpi=None, facecolor='w', edgecolor='w',
                        orientation='portrait', format=file_type,
                        transparent=False, bbox_inches=None, metadata=None)
            plt.close(fig)   
            feature_idx += 4
        if 'v' in model_corr_features:
            fig, ax = plt.subplots(3, 1, figsize=(13, 14))
            for i in range(3):
                ax[i].plot(t_est, model_corr[:n_test, i+feature_idx], label="est", zorder=2)
                ax[i].plot(t_act, mhe_error[:n_tact, i+7], label="act", alpha=0.6, zorder=1)
                tit = 'v_' + labels[i] + ' estimated model error'
                ax[i].set_ylabel(tit)
                ax[i].grid()
                ax[i].legend()
            ax[0].set_title(r'$Estimated Model Error$')
            ax[2].set_xlabel(r'$t [s]$')
            plt.tight_layout()
            fig.savefig(img_save_dir + '/v_estimated_model_error.'+file_type, dpi=None, facecolor='w', edgecolor='w',
                        orientation='portrait', format=file_type,
                        transparent=False, bbox_inches=None, metadata=None)
            plt.close(fig)   
            n_corr = min(n_tmeas, n_test)
            if n_tmeas < n_test:
                t_corr = t_meas
            else: 
                t_corr = t_est
            corr_a_thrust = a_thrust[:n_corr] + model_corr[:n_corr, feature_idx:feature_idx+3]
            if a_thrust is not None:
                fig, ax = plt.subplots(3, 1, sharex='all', figsize=(13, 14))
                for i in range(3):
                    p1, = ax[i].plot(t_meas, y_measured[:n_tmeas, i+6], label='gt', color='C1', zorder=2)
                    p3, = ax[i].plot(t_meas, y_measured_noisy[:n_tmeas, i+6], label='measured', color='C2', zorder=1, alpha=0.5)
                    p2, = ax[i].plot(t_corr, corr_a_thrust[:n_tmeas, i], label='thrust command', color='C0', zorder=3)
                    tit = 'a_' + labels[i] + ' measurement'
                    ax[i].legend()
                    ax[i].set_ylabel(tit)
                    ax[i].grid()
                ax[0].set_title(r'$a\:[m/s^2]$')
                ax[2].set_xlabel(r'$t [s]$')
                plt.tight_layout()
                fig.savefig(img_save_dir + '/corrected_thrust_estimation.'+file_type, dpi=None, facecolor='w', edgecolor='w',
                            orientation='portrait', format=file_type,
                            transparent=False, bbox_inches=None, metadata=None)
                plt.close(fig)

            feature_idx += 3
        if 'w' in model_corr_features:
            fig, ax = plt.subplots(3, 1, figsize=(13, 14))
            for i in range(3):
                ax[i].plot(t_est, model_corr[:n_test, i+feature_idx], label="est")
                ax[i].plot(t_act, mhe_error[:n_tact, i+10], label="act")
                tit = 'w_' + labels[i] + ' estimated model error'
                ax[i].set_ylabel(tit)
                ax[i].grid()
                ax[i].legend()
            ax[0].set_title(r'$Estimated Model Error$')
            ax[2].set_xlabel(r'$t [s]$')
            plt.tight_layout()
            fig.savefig(img_save_dir + '/w_estimated_model_error.'+file_type, dpi=None, facecolor='w', edgecolor='w',
                        orientation='portrait', format=file_type,
                        transparent=False, bbox_inches=None, metadata=None)
            plt.close(fig)   

    if show_dvdt:
        p, q, v_b_executed, w = separate_variables(x_act)
        v_w_executed =[]
        for i in range(len(t_act)):
            v_w_executed.append(v_dot_q(v_b_executed[i], (q[i])))
        v_w_executed = np.stack(v_w_executed)
        a_w_executed = []
        for i in range(len(t_act)-1):
            a_w_executed.append((v_w_executed[i+1, :] - v_w_executed[i, :])/(0.02))
        a_w_executed.append((v_w_executed[i+1, :] - v_w_executed[i, :])/(0.02))
        a_w_executed = np.stack(a_w_executed)

        a_b_executed = []
        for i in range(len(t_act)):
            a_b_executed.append(v_dot_q(a_w_executed[i, :], quaternion_inverse(q[i])))
        a_b_executed = np.stack(a_b_executed)

        a_w_meas = []
        for i in range(len(t_meas)):
            a_w_meas.append(v_dot_q(y_measured[i, 6:], (q[i])) - cs.vertcat(0, 0, 9.8))
        a_w_meas = np.stack(a_w_meas)

        a_b_meas = []
        for i in range(len(t_meas)):
            a_b_meas.append(v_dot_q(a_w_meas[i, :], quaternion_inverse(q[i])))
        a_b_meas = np.stack(a_b_meas)
        
        fig, ax = plt.subplots(3, 1, sharex='all', figsize=(13, 14))
        for i in range(3):
            p1, = ax[i].plot(t_meas, a_b_meas[:n_tmeas, i], label='measurement', color='C0', zorder=1)
            tit = 'a_' + labels[i] + ' measurement'
            ax[i].set_ylabel(tit)
            p2, = ax[i].plot(t_act, a_b_executed[:n_tact, i], label='dv/dt', color='C1', zorder=2)
            tit = 'a_' + labels[i] + ' executed'
            lns = [p1, p2]
            ax[i].legend(handles=lns)
            ax[i].grid()
        ax[0].set_title(r'$a\:[m/s^2]$')
        ax[2].set_xlabel(r'$t [s]$')
        plt.tight_layout()
        fig.savefig(img_save_dir + '/linear_acc_body_measurement.'+file_type, dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', format=file_type,
                    transparent=False, bbox_inches=None, metadata=None)
        plt.close(fig)

        fig = plt.figure(figsize=(40, 40))
        fig, ax = plt.subplots(3, 1, sharex='all', figsize=(13, 14))
        for i in range(3):
            p1, = ax[i].plot(t_meas, a_w_meas[:n_tmeas, i], label='acceleration', color='C0', zorder=100000)
            tit = 'a_' + labels[i] + ' measurement'
            ax[i].set_ylabel(tit)
            p2, = ax[i].plot(t_act, a_w_executed[:n_tact, i], label='dv/dt', color='C1')
            tit = 'a_' + labels[i] + ' executed'
            lns = [p1, p2]
            ax[i].legend(handles=lns)
            ax[i].grid()
        ax[0].set_title(r'$a\:[m/s^2]$')
        ax[2].set_xlabel(r'$t [s]$')
        plt.tight_layout()
        fig.savefig(img_save_dir + '/linear_acc_world_measurement.'+file_type, dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', format=file_type,
                    transparent=False, bbox_inches=None, metadata=None)
        plt.close(fig)
