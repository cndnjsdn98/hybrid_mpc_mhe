import torch
import os
import numpy as np
import argparse

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
    if verbose >= 1:
        print("Begin...")
    if gpu is not None:
        device = 'cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'
    
    # Flight Data Parameters
    if verbose >= 1:
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
    y_features_ = data_params.get('y_features', 'vw')
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
    if verbose >= 1:
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
    if verbose >= 1:
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
    adjoint = model_params.get("adjoint", False)
    valid_freq = model_params.get("valid_freq", 20)
    save_training_history = model_params.get("save_training_history", True)
    viz = model_params.get("viz", True)
    if verbose >= 1:
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
    if verbose >= 1:
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
                  adjoint=adjoint,
                  device=device,
                  save_training_history=save_training_history,
                  valid_freq=valid_freq,
                  verbose=verbose, 
                  viz=viz)
    
    return neuralODE

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train Neural ODE', fromfile_prefix_chars='@')
    # Data Params
    parser.add_argument('--quad_name', type=str, choices=['hummingbird', 'clara'], default='hummingbird')
    parser.add_argument('--train_trajectory_name', type=str, choices=['lemniscate', 'circle', 'random'], default='circle')
    parser.add_argument('--valid_trajectory_name', type=str, choices=['lemniscate', 'circle', 'random'], default='lemniscate')
    parser.add_argument('--env', type=str, choices=['gazebo', 'arena'], default='gazebo')
    parser.add_argument('--gt', action='store_true')
    parser.add_argument('--x_features', type=str, default='vwu')
    parser.add_argument('--y_features', type=str, default='vw')
    parser.add_argument('--n_integration', type=int, default=2)
    # Model Params
    parser.add_argument('--n_hidden', type=int, nargs='+', default=[16, 32, 16])
    parser.add_argument('--activation_hidden', type=str, choices=['linear', 'relu', 'tanh', 'sigmoid', 'elu', 'leaky_relu'], default='tanh')
    parser.add_argument('--activation_out', type=str, choices=['linear', 'relu', 'tanh', 'sigmoid', 'elu', 'leaky_relu'], default='tanh')
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--batch_normalization', action='store_true')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lrate', type=float, default=1e-3)
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--valid_freq', type=int, default=20)
    parser.add_argument('--save_training_history', action='store_true')
    parser.add_argument('--adjoint', action='store_true')
    parser.add_argument('--verbose', '-v', action='count', default=0)
    # Training Params
    parser.add_argument('--n_threads', type=int)
    args = parser.parse_args()
    
    # torch.set_num_threads(args.n_threads)
    # torch.set_num_interop_threads(args.n_threads)

    data_params = {
        'quad_name': args.quad_name,
        'train_trajectory_name': args.train_trajectory_name,
        'valid_trajectory_name': args.valid_trajectory_name,
        'env': args.env,
        'gt': args.gt,
        'x_features': args.x_features,
        'y_features': args.y_features,
        'n_integration': args.n_integration,
    }
    model_params = {
        'n_hidden': args.n_hidden,
        'activation_hidden': args.activation_hidden,
        'activation_out': args.activation_out,
        'dropout': args.dropout,
        'batch_normalization': args.batch_normalization,
        'epochs': args.epochs,
        'lrate': args.lrate, 
        'adjoint': args.adjoint,
        'valid_freq': args.valid_freq,
        'viz': args.viz,
        'save_training_history': args.save_training_history,
    }
    print(data_params, model_params)

    neuralODE = train_node(model_params, data_params, gpu=args.gpu, verbose=args.verbose)
    neuralODE.save_model()