import numpy as np
import pickle
import os
import json
import matplotlib.pyplot as plt

from src.utils.utils import world_to_body_velocity_mapping
from src.utils.DirectoryConfig import DirectoryConfig as DirConf

class FlightDataset:
    def __init__(self, data_dir, n_integration=2):
        """
        Load quad flight result and compile dataset to train NN
        """
        self.load_data(data_dir)
        self.configure_ds(n_integration)
        
    def load_data(self, data_dir):
        """
            Load pickled Quadrotor Flight results
        """
        with open(os.path.join(data_dir, "results.pkl"), "rb") as fp:
            data = pickle.load(fp)
        with open(os.path.join(data_dir, "meta_data.json"), "rb") as fp:
            meta = json.load(fp)
        
        self.quad_name = meta["quad_name"]
        self.env = meta["env"]

        # load flight data
        self.t = data["t"]
        if "dt" in data.keys():
            self.dt = data["dt"]
        else:
            self.dt = np.diff(self.t)
        if "state_in_Body" in data.keys():
            self.x_in = data["state_in_Body"]   
        else:
            self.x_in = world_to_body_velocity_mapping(data["state_in"])
        self.cmd_thrust = data["input_in"]
        self.error = data["error"]

        # Remove invalid entries (dt = 0)
        invalid = np.where(self.dt == 0)
        self.dt = np.delete(self.dt, invalid, axis=0)
        self.x_in = np.delete(self.x_in, invalid, axis=0)
        self.error = np.delete(self.error, invalid, axis=0)

        # GP input and output
        self.train_in = self.x_in
        self.train_out = self.error

    def configure_ds(self, n_integration):
        """
        Configure DataSet such that a single instance of training input 
        is paried with n_itegration points into the future of training outputs 
        and time
        """
        # T is the total number of time steps, N is the number of input features
        N_cmd = self.cmd_thrust.shape[1]
        T, N_in = self.train_in.shape  
        _, N_out = self.train_out.shape
        num_samples = T - n_integration + 1  # Number of samples you can create

        train_in_reform = np.zeros((num_samples, N_in))
        train_out_reform = np.zeros((n_integration, num_samples, N_out))
        cmd_reform = np.zeros((num_samples, n_integration-1, N_cmd))
        out_times = np.zeros((n_integration))
        for i in range(num_samples):
            train_in_reform[i, :] = self.train_in[i, :]
            cmd_reform[i, :] = self.cmd_thrust[i:i+n_integration-1, :]

        train_out_reform[0, :, :] = self.train_in[:num_samples, :]
        for i in range(n_integration-1):
            train_out_reform[i+1, :, :] =self.train_out[i:i+num_samples, :]

        # TODO: Need to fix this dt
        dt = 0
        for j in range(n_integration):
            out_times[j] =  dt
            dt += self.dt[j]
    
        self.train_in = train_in_reform
        self.train_out = train_out_reform
        self.out_times = out_times
        self.cmd_thrust = cmd_reform

    def get_ds(self, x_idx=None, y_idx=None):
        """
        Returns a Dataset to train GP to compensate of MHE/MPC model error
        return: [x, y, t] 
        """
        if x_idx is not None and y_idx is not None:
            return self.train_in[:, x_idx], self.train_out[:, :, y_idx], self.out_times
        else:
            return self.train_in, self.train_out, self.out_times
    
    def get_cmd(self):
        """
        Returns the commanded rotor thrusts 
        return: [u] 
        """
        return self.cmd_thrust
    
    def visualize(self):
        """
        Visualize the trajectory executed by the loaded dataset
        """
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
        
        fig_traj, ax_traj = plt.subplots(3, 1, sharex='all', figsize=(24, 16))
        for i in range(3):
            ax_traj[i].plot(self.t, self.x_in[:, i])
            ax_traj[i].set_ylabel(labels[i])
            ax_traj[0].set_title(r'$p\:[m]$')
            ax_traj[2].set_xlabel(r'$t [s]$')
        fig_traj.suptitle("Executed Trajectory")

        fig_err, ax_err = plt.subplots(3, 2, sharex='all', figsize=(24, 16))
        for i in range(3):
            ax_err[i, 0].plot(self.t, self.error[:, i+7])
            ax_err[i, 0].set_ylabel(labels[i])
            ax_err[0, 0].set_title(r'$error_v\:[m/s]$')
            ax_err[2, 0].set_xlabel(r'$t [s]$')
        for i in range(3):
            ax_err[i, 1].plot(self.t, self.error[:, i+10])
            ax_err[i, 1].set_ylabel(labels[i])
            ax_err[0, 1].set_title(r'$error_w\:[rad/s]$')
            ax_err[2, 1].set_xlabel(r'$t [s]$')
        fig_err.suptitle("Model Error")

        plt.show()

if __name__ == "__main__":
    ds = FlightDataset("")