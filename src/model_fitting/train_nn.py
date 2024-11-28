import torch
import os
import numpy as np
from torch.optim import Adam
import torch.nn as nn

from src.model_fitting.NeuralODE import NeuralODE
from src.model_fitting.load_data import FlightDataset
from src.utils.DirectoryConfig import DirectoryConfig as DirConf

def train_node(model_params:dict, data_params:dict, verbose=0, gpu=None):
    """
    Train Neural ODE Model
    :param model_params: Neural ODE model parameters
    :type model_params: dict
    :param data_params: Flight Data parameters to train Neural ODE model on
    :type data_params: dict
    :param verbose: Indicate verbosity level
    :type verbose: Int
    :param gpu: Indicate GPU device to use or None to use CPU
    :type gpu: Int
    
    Keys in model_params
    - n_hidden: Lists of number of neurons in each layer
        - type: [Int]
    - activation_hidden: Activation function of hidden layers
        - type: string
    - activation_out: Activation function of output layer
        - type: string
    - epoch: Number of training epochfor GP training
        - type: Int
    - lrate: Learning rate of the optimizer
        - type: double

    Keys in data_params
    - quad_name: Name of the quadrotor
        - type: string
    - train_trajectory_name: The name of the trajectory that was executed to collect the flight data to train the model
        - type: string
    - valid_trajectory_name: The name of the trajectory that was executed to collect the flight data to validate the model
        - type: string
    - env: String value indicating the environment the quadrotor flight was executed for data collection
        - type: string
    - gt: Boolean value to indicate whether groundtruth state measurements were used for flight execution.
        - type: Bool
    - x_features: String indicating the states used as input features
        - type: String
    - y_features: String indicating the states that comprises the output features
        - type: String
    - n_integration: Number of integrations in to the future points utilized for single training instance
        - type: Int
    """
    print("Begin...")
    if gpu is not None:
        device = 'cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'
    
    # Flight Data Parameters
    print("Loading Flight Data...")
    v = [7, 8, 9]
    w = [10, 11, 12]
    quad_name = data_params.get("quad_name", 'hummingbird').lower()
    train_trajectory_name = data_params.get('train_trajectory_name', 'circle').lower()
    valid_trajectory_name = data_params.get('valid_trajectory_name', 'lemniscate').lower()
    env = data_params.get('env', 'gazebo').lower()
    gt = data_params.get('gt', True)
    x_features_ = data_params.get('x_features', 'vwu')
    x_features = []
    if 'v' in x_features_:
        x_features.extend(v)
    if 'w' in x_features_:
        x_features.extend(w)
    y_features_ = data_params.get('x_features', 'vw')
    y_features = []
    if 'v' in y_features_:
        y_features.extend(v)
    if 'w' in y_features_:
        y_features.extend(w)
    n_integration = data_params.get("n_integration", 2)

    # Load Dataset
    flight_name = "%s_mpc%s_%s"%(env, "_gt" if gt else "", quad_name)
    train_ds_dir = os.path.join(DirConf.FLIGHT_DATA_DIR, flight_name, train_trajectory_name)
    valid_ds_dir = os.path.join(DirConf.FLIGHT_DATA_DIR, flight_name, valid_trajectory_name)
    train_ds = FlightDataset(train_ds_dir, n_integration)
    valid_ds = FlightDataset(valid_ds_dir, n_integration)
    # train_ds.visualize()
    # valid_ds.visualize()
    print("Flight Dataset Loaded...")
    # Retrieve Training Data
    train_init, train_out, train_times = train_ds.get_ds(x_features, y_features)
    train_init = torch.Tensor(train_init[:, np.newaxis, :]).to(device)
    train_out = torch.Tensor(train_out[:, :, np.newaxis, :]).to(device)
    train_times = torch.Tensor(train_times).to(device)
    if 'u' in x_features_:
        train_cmd = train_ds.get_cmd()
        train_cmd = torch.Tensor(train_cmd).to(device)
    # Retrieve Validation Data
    valid_init, valid_out, valid_times = valid_ds.get_ds(x_features, y_features)
    valid_init = torch.Tensor(valid_init[:, np.newaxis, :]).to(device)
    valid_out = torch.Tensor(valid_out[:, :, np.newaxis, :]).to(device)
    valid_times = torch.Tensor(valid_times).to(device)
    if 'u' in x_features_:
        valid_cmd = valid_ds.get_cmd()
        valid_cmd = torch.Tensor(valid_cmd).to(device)
    print("Training and Validation Set initialized...")
    # Model Parameters
    n_inputs = len(x_features)
    if 'u' in x_features_:
        n_inputs += train_cmd.shape[2]
    n_output = len(y_features)
    n_hidden = model_params.get("n_hidden", [32, 64, 64, 32])
    activation_hidden = model_params.get("activation_hidden", "tanh")
    activation_out = model_params.get("activation_out", "linear")
    dropout = model_params.get("dropout", None)
    batch_normalization = model_params.get("batch_normalization", False)
    epochs = model_params.get("epochs", 1000)
    lrate = model_params.get("lrate", 1e-3)
    print("Creating Neural ODE Network")
    
    # Create Neural-ODE Model
    model_name = "%s_%s_%s"%("gz" if env.lower() == "gazebo" else "rt", 
                              x_features_,
                              y_features_)
    neuralODE = NeuralODE(model_name,
                          n_inputs, 
                          n_hidden,
                          n_output, 
                          activation_hidden, 
                          activation_out, 
                          dropout=dropout, 
                          batch_normalization=batch_normalization)
    print("Model Created. Initiate Training...")
    
    # Train model
    optimizer = Adam(neuralODE.parameters(), lr=lrate)
    loss_fcn = nn.MSELoss()
    neuralODE.fit(train_init,
                  train_cmd, 
                  train_out, 
                  train_times, 
                  valid_init,
                  valid_cmd,
                  valid_out,
                  valid_times,
                  optimizer, 
                  loss_fcn, 
                  epochs,  
                  verbose=0, 
                  device=device)
    
    return neuralODE

if __name__ == '__main__':
    data_params = {
        'quad_name': 'hummingbird',
        'train_trajectory_name': 'circle',
        'valid_trajectory_name': 'lemniscate',
        'env': 'gazebo',
        'gt': True,
        'x_features': 'vwu',
        'y_features': 'vw',
        'n_integration': 2,
    }
    model_params = {
        'n_hidden': [16, 32, 64, 32, 16, 8],
        'activation_hidden': 'tanh',
        'activation_out': 'linear',
        'dropout': None,
        'batch_normalization': False,
        'epochs': 1000,
        'lrate': 1e-2, 
    }
    print(data_params, model_params)

    neuralODE = train_node(model_params, data_params, gpu=None)
    neuralODE.save_model()