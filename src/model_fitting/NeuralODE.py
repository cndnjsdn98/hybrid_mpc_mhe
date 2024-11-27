import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn as nn

from src.utils.utils import safe_mkdir_recursive
from src.utils.DirectoryConfig import DirectoryConfig as DirConf
class NeuralODE(nn.Module):
    def __init__(self, model_name, n_inputs, n_hidden, n_output, activation_hidden, activation_out, dropout=None, batch_normalization=False):
        super(NeuralODE, self).__init__()
        self.model_name = model_name
        self.model_dir = os.path.join(DirConf.MODELS_DIR, self.model_name)
        safe_mkdir_recursive(self.model_dir, overwrite=False)

        self.model_params = {
            'model_name': model_name,
            'n_inputs': n_inputs,
            'n_hidden': n_hidden,
            'n_output': n_output,
            'activation_hidden': activation_hidden,
            'activation_out': activation_out,
            'dropout': dropout,
            'batch_normalization': batch_normalization,
        }

        # Activation functions
        self.activation_hidden = self._get_activation_function(activation_hidden)
        if activation_out.lower() == "linear":
            self.activation_out = None
        else:
            self.activation_out = self._get_activation_function(activation_out)
        
        # Layers
        layers = []
        input_dim = n_inputs
        for i, n in enumerate(n_hidden):
            layers.append(nn.Linear(input_dim, n, bias=True))
            # if batch_normalization:
                # layers.append(nn.BatchNorm1d(input_dim))
            input_dim = n
            layers.append(self.activation_hidden)
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
        # Output layer
        layers.append(nn.Linear(input_dim, n_output, bias=True))
        if self.activation_out is not None:
            layers.append(self.activation_out)
        
        # Wrap as Sequential
        self.network = nn.Sequential(*layers)
        print(self.network)

    def forward(self, t, x):
        return self.network(x)
    
    def _get_activation_function(self, activation):
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU()
        }
        return activations.get(activation.lower(), nn.Identity())

    def fit(self, 
            train_init,
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
            device='cpu',
            adjoint=False,
            valid_freq=20,
            save_training_history=True,
            viz_loss_curve=True,
            viz_results=True):
        if adjoint:
            from torchdiffeq import odeint_adjoint as odeint
        else:
            from torchdiffeq import odeint
            
        self.to(device)
        if train_cmd is not None:
            train_wrapper = lambda t, x: self(t, torch.cat((x, train_cmd), dim=-1))
            valid_wrapper = lambda t, x: self(t, torch.cat((x, valid_cmd), dim=-1))
        else:
            train_wrapper = lambda t, x: self(t, x)
            valid_wrapper = lambda t, x: self(t, x)

        if save_training_history:
            # [[epoch, train_loss, valid_loss]]
            self.loss_hist = np.zeros((3, int(epochs/valid_freq)))

        for i in range(epochs):
            optimizer.zero_grad()
            pred_out = odeint(train_wrapper, train_init, train_times).to(device)
            loss = loss_fcn(pred_out, train_out)
            loss.backward()
            optimizer.step()
            if i % valid_freq == 0:
                with torch.no_grad():
                    pred_valid = odeint(valid_wrapper, valid_init, valid_times).to(device)
                    loss_valid = loss_fcn(pred_valid, valid_out)
                    j = int(i/valid_freq)
                    self.loss_hist[:, j] = np.array([i, loss.item(), loss_valid.item()])
                    print('Iter {:04d} | Train Loss {:.6f} | Valid Loss {:.6f}'.format(i, loss.item(), loss_valid.item()))

        if viz_results:
            pred_out = odeint(train_wrapper, train_init, train_times).to(device).detach().numpy()
            pred_valid = odeint(valid_wrapper, valid_init, valid_times).to(device).detach().numpy()
            self.viz_results(train_out[1], pred_out[1], valid_out[1], pred_valid[1])
        if viz_loss_curve:
            self.visualize_loss_curve()

    def visualize_loss_curve(self, fig_format="pdf"):
        """
        Visualize the loss curve of the training and validation
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

        fig, ax = plt.subplots(1, 1, sharex='all', figsize=(6, 4))
        ax.plot(self.loss_hist[0, :], self.loss_hist[1, :], label='train')
        ax.plot(self.loss_hist[0, :], self.loss_hist[2, :], label='validation')
        ax.set_ylabel('loss')
        ax.set_xlabel('epochs')
        ax.set_title('Training History')
        ax.legend()
        plt.tight_layout()
        # Save figure
        fig_save_dir = os.path.join(self.model_dir, "loss_curve."+fig_format)
        fig.savefig(fig_save_dir, dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', format=fig_format,
                    transparent=True, bbox_inches='tight', metadata=None, pad_inches=0.01)
        plt.close(fig)

    def viz_results(self, train_out, pred_train, valid_out, pred_valid, fig_format="pdf"):
        """
        Visualize the prediction results of the model
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

        n_out = train_out.shape[-1]
        if n_out == 3:
            # Plot Train
            fig_train, ax_train = plt.subplots(3, 1, sharex='all', figsize=(10, 8))
            for i in range(3):
                ax_train[i].plot(train_out[:, :, i], label='error')
                ax_train[i].plot(pred_train[:, :, i], label='pred')
                ax_train[i].set_xlabel('t')
                ax_train[i].legend()
            ax_train[0].set_ylabel('e_vx [m/s]')
            ax_train[1].set_ylabel('e_vy [m/s]')
            ax_train[2].set_ylabel('e_vz [m/s]')
            fig_train.suptitle('Training')
            plt.tight_layout()
            # Plot Validation
            fig_valid, ax_valid = plt.subplots(3, 1, sharex='all', figsize=(10, 8))
            for i in range(3):
                ax_valid[i].plot(valid_out[:, :, i], label='error')
                ax_valid[i].plot(pred_valid[:, :, i], label='pred')
                ax_valid[i].set_xlabel('t')
                ax_valid[i].legend()
            ax_valid[0].set_ylabel('e_vx [m/s]')
            ax_valid[1].set_ylabel('e_vy [m/s]')
            ax_valid[2].set_ylabel('e_vz [m/s]')
            fig_valid.suptitle('Validation')
            plt.tight_layout()
        elif n_out == 6:
            # Plot Train
            fig_train, ax_train = plt.subplots(3, 2, sharex='all', figsize=(10, 8))
            for i in range(3):
                ax_train[i, 0].plot(train_out[:, :, i], label='error')
                ax_train[i, 0].plot(pred_train[:, :, i], label='pred')
                ax_train[i, 0].set_xlabel('t')
                ax_train[i, 0].legend()
                ax_train[i, 1].plot(train_out[:, :, i+3], label='error')
                ax_train[i, 1].plot(pred_train[:, :, i+3], label='pred')
                ax_train[i, 1].set_xlabel('t')
                ax_train[i, 1].legend()
            ax_train[0, 0].set_ylabel('e_vx [m/s]')
            ax_train[1, 0].set_ylabel('e_vy [m/s]')
            ax_train[2, 0].set_ylabel('e_vz [m/s]')
            ax_train[0, 1].set_ylabel('e_wx [rad/s]')
            ax_train[1, 1].set_ylabel('e_wy [rad/s]')
            ax_train[2, 1].set_ylabel('e_wz [rad/s]')
            fig_train.suptitle('Training')
            plt.tight_layout()
            # Plot Validation
            fig_valid, ax_valid = plt.subplots(3, 2, sharex='all', figsize=(10, 8))
            for i in range(3):
                ax_valid[i, 0].plot(valid_out[:, :, i], label='error')
                ax_valid[i, 0].plot(pred_valid[:, :, i], label='pred')
                ax_valid[i, 0].set_xlabel('t')
                ax_valid[i, 0].legend()
                ax_valid[i, 1].plot(valid_out[:, :, i+3], label='error')
                ax_valid[i, 1].plot(pred_valid[:, :, i+3], label='pred')
                ax_valid[i, 1].set_xlabel('t')
                ax_valid[i, 1].legend()
            ax_valid[0, 0].set_ylabel('e_vx [m/s]')
            ax_valid[1, 0].set_ylabel('e_vy [m/s]')
            ax_valid[2, 0].set_ylabel('e_vz [m/s]')
            ax_valid[0, 1].set_ylabel('e_wx [rad/s]')
            ax_valid[1, 1].set_ylabel('e_wy [rad/s]')
            ax_valid[2, 1].set_ylabel('e_wz [rad/s]')
            fig_valid.suptitle('Validation')
            plt.tight_layout()
        # Save figure
        fig_train_save_dir = os.path.join(self.model_dir, "loss_curve."+fig_format)
        fig_train.savefig(fig_train_save_dir, dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', format=fig_format,
                    transparent=True, bbox_inches='tight', metadata=None, pad_inches=0.01)
        plt.close(fig_train)
        fig_valid_save_dir = os.path.join(self.model_dir, "loss_curve."+fig_format)
        fig_valid.savefig(fig_valid_save_dir, dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', format=fig_format,
                    transparent=True, bbox_inches='tight', metadata=None, pad_inches=0.01)
        plt.close(fig_valid)

    def save_model(self):
        '''
        Save Model as state_dict
        '''
        torch.save(self.state_dict(), self.model_dir, "model.pth")
        with open(os.path.join(self.model_dir, "model_params.pkl"), "wb") as fp:
            pickle.dump(self.model_params, fp)

def load_model(model_name):
    '''
    Load Model
    '''
    model_path = os.path.join(DirConf.MODELS_DIR, model_name)
    with open(os.path.join(model_path, "model_params.pkl"), "rb") as fp:
        model_params =  pickle.load(fp)
    
    n_inputs = model_params["x_features"]
    n_output = model_params["y_features"]
    n_hidden = model_params["n_hidden"]
    activation_hidden = model_params["activation_hidden"]
    activation_out = model_params["activation_out"]
    dropout = model_params["dropout"]
    batch_normalization = model_params["batch_normalization"]

    neuralODE = NeuralODE(model_name,
                          n_inputs, 
                          n_hidden,
                          n_output, 
                          activation_hidden, 
                          activation_out, 
                          dropout=dropout, 
                          batch_normalization=batch_normalization)
    neuralODE.load_state_dict(torch.load(os.path.join(model_path, "model.pth"), weights_only=True))

    return neuralODE
# if __name__ == '__main__':
#    '''
   
#    '''
#    return