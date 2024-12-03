import numpy as np
import pickle
import os
import tqdm
import json
import matplotlib.pyplot as plt
from src.quad_opt.quad_optimizer import QuadOptimizer
from src.quad_opt.quad import custom_quad_param_loader
from src.utils.utils import world_to_body_velocity_mapping, separate_variables, v_dot_q, quaternion_inverse, vw_to_vb
from src.utils.DirectoryConfig import DirectoryConfig as DirConf

class GPDataset:
    def __init__(self, data_dir):
        """
            Load quad flight result and compile dataset to train GP
        """
        #
        # GP dataset from UZH for MPC
        # gp_data_dir = os.path.join(DirConf.GP_MODELS_DIR, "af53a8b", "gazebo_sim_gp_dense_noisy")
        # with open(os.path.join(gp_data_dir, "data_7.pkl"), "rb") as fp:
        #     data = pickle.load(fp)
        # self.train_in = np.zeros((len(data["train_in"]), 13))
        # self.train_out = np.zeros((len(data["train_in"]), 13))
        # self.train_in[:, 7] = np.squeeze(data['train_in'])
        # self.train_out[:, 7] = np.squeeze(data['train_out'])
        # with open(os.path.join(gp_data_dir, "data_8.pkl"), "rb") as fp:
        #     data = pickle.load(fp)
        # self.train_in[:, 8] = np.squeeze(data['train_in'])
        # self.train_out[:, 8] = np.squeeze(data['train_out'])
        # with open(os.path.join(gp_data_dir, "data_9.pkl"), "rb") as fp:
        #     data = pickle.load(fp)
        # self.train_in[:, 9] = np.squeeze(data['train_in'])
        # self.train_out[:, 9] = np.squeeze(data['train_out'])
        
        # # GP Dataset for MHE
        # mhe_data_dir = os.path.join(DirConf.GP_MODELS_DIR, "678_789_50_150_mhe_lemniscate_noise_3")
        # with open(os.path.join(mhe_data_dir, "train_dataset.pkl"), "rb") as fp:
        #     data = pickle.load(fp)
        # self.train_in = data["train_in"]
        # self.train_out = data["train_out"]
        # print(self.train_in.shape)
        # Mine
        self.load_data(data_dir)
        
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
        if "t_mpc" in meta.keys():
            self.mpc = True
            self.mhe = False
            # load MPC meta data
            # self.t_mpc = meta["t_mpc"]
            # self.n_mpc = meta["n_mpc"]
            # self.gt = meta["gt"]
            # self.with_gp = meta["with_gp"]

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
            self.error = data["error"]

            # Remove invalid entries (dt = 0)
            invalid = np.where(self.dt == 0)
            self.dt = np.delete(self.dt, invalid, axis=0)
            self.x_in = np.delete(self.x_in, invalid, axis=0)
            self.error = np.delete(self.error, invalid, axis=0)
            # GP input and output
            self.train_in = self.x_in
            self.train_out = self.error
        else:
            self.mpc = False
            self.mhe = True
            # load MHE meta data
            # self.t_mhe = meta["t_mhe"]
            # self.n_mhe = meta["n_mhe"]
            # self.mhe_type = meta["mhe_type"]
            # self.with_gp = meta["with_gp"]

            # load flight data
            self.sensor_meas = data["sensor_meas"]
            # self.motor_thrust = data["motor_thrusts"]
            if "error" in data.keys():
                self.error = data["error"]
            elif "mhe_error" in data.keys():
                self.error = data["mhe_error"]
            # GP input and output
            self.train_in = self.sensor_meas
            self.train_out = self.error

    def get_train_ds(self, x_idx=None, y_idx=None):
        """
        Returns a Dataset to train GP to compensate of MHE/MPC model error
        return: [x, y] 
        """
        if x_idx is not None and y_idx is not None:
            return self.train_in[:, x_idx], self.train_out[:, y_idx]
        else:
            return self.train_in, self.train_out
    
    def get_len(self):
        return self.train_in.shape[0]
    

if __name__ == "__main__":
    gp_ds = GPDataset("")