import torch
import gpytorch
import os
import json
import time
import pickle
import pandas as pd
import numpy as np
from src.utils.utils import safe_mkdir_recursive
from src.gp.gpy_model import *
from src.utils.DirectoryConfig import DirectoryConfig as DirConfig
import matplotlib.pyplot as plt

class GPyModelWrapper:
    """
    Class for storing GPy Models, likelihood and its necessary parameters. 
    """
    def __init__(self, model_name=None, load=False,
                 keep_train_data=False, 
                 x_features=[7,8,9], u_features=[], y_features=[7,8,9], 
                 mhe=False,
                 model_dir=None):
        """
        :param model_name: String value of the model name
        :type model_name: string 
        :param load: Boolean value to determine whether to load the existing model if model_name exists
        :type load: Boolean
        :param keep_train_data: True if wish to keep train_x and train_y of the GPy model
        :type keep_train_data: bool
        :param x_features: List of n regression feature indices from the quadrotor state indexing.
        :type x_features: list
        :param u_features: List of n' regression feature indices from the input state.
        :type u_features: list
        :param y_features: Index of output dimension being regressed as the time-derivative.
        :type y_features: list  
        """
        self.machine = 0 # 0 indicttes cpu and 1 indicates gpu
        self.model_name = model_name
        self.mhe = mhe
        self.keep_train_data = keep_train_data
        #  Get Model directory
        if model_dir is not None:
            self.gp_model_dir = os.path.join(model_dir, model_name)
        else:
            self.gp_model_dir = os.path.join(DirConfig.MODELS_DIR, 'gp', model_name)

        # Check if a model exists in model_name
        if load and model_name is not None and os.path.exists(os.path.join(self.gp_model_dir, "gpy_config.json")):
                # If the provided model name exists then load that model
                self.load(keep_train_data=keep_train_data)
        else:
            # Create the directory to save the gpy model and the meta data in
            try:
                safe_mkdir_recursive(self.gp_model_dir, overwrite=False)
            except:
                print("OVERWRITING EXISTING GP MODEL")
            # Else Set up Model paramters as provided
            self.x_features = x_features
            self.u_features = u_features
            self.y_features = y_features
            self.n_dim = len(x_features)
            self.model = None
            self.likelihood = None

    def train(self, train_x, train_y, train_iter, 
              induce_num=None, induce_points=None,
              verbose=0, script_model=False):
        """
        Trains the GPy Model with the given input training dataset.
        :param train_x: Array of Training Input data
        :type train_x: torch.Tensor
        :param train_y: Array of Training Output data
        :type train_y: torch.Tensor
        :param train_iter: Integer value describing number of iterations
        for training the GPy model
        :type train_iter: integer
        :param induce_num: Integer value for describing the number of inducing points 
        for variational GPy Models.
        :type induce_num: integer
        :param induce_points: Array of points describing the inducing locations
        for variational GPy Models.
        :type induce_points: list
        :param dense_model_name: Name of the dense model used to predict training data
        :type dense_model_name: string
        :param verbose: Verbose level ie. print level of Model training.
        :type script_model: int
        :param script_model: Boolean value indicating whether to script the model for libtorch.
        :type script_model: bool
        """
        # gpytorch.settings.cholesky_jitter(1e-4)
        if self.keep_train_data:
            self.train_x = train_x
            self.train_y = train_y

        self.train_and_save_Approx_model(train_x, train_y, train_iter,
                                            induce_num=induce_num, 
                                            induce_points=induce_points, 
                                            verbose=verbose, script_model=script_model)
        self.machine = 1

    def predict(self, input, skip_variance=True, gpu=False):
        """
        Returns the predicted mean and variance value of the GPy model for 
        the given input value.
        :param input: A n_dim x N Array of inputs to be predicted
        :type input: torch.Tensor
        :return Array of predicted mean and variance value for the given input value. 
        """
        cov_pred = np.zeros(input.shape)
        mean_pred = np.zeros(input.shape)

        # If the input is an array with only single prediction points
        if input.ndim == 1:
            num_dim = 3
            test_x = {}
            for i in range(len(input)):
                if gpu:
                    test_x[i] = torch.Tensor([input[i]]).cuda()
                else:
                    test_x[i] = torch.Tensor([input[i]]).cpu()
        else:
            num_dim = input.shape[1]
            if gpu:
                test_x = torch.Tensor(input).cuda()
            else:
                test_x = torch.Tensor(input).cpu()
        
        if gpu:
            self.gpu()
        else:
            self.cpu()
                    
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.memory_efficient(),  \
            gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition = False), \
            gpytorch.settings.debug(state=False), gpytorch.settings.max_cg_iterations(10),\
            gpytorch.settings.detach_test_caches(), \
            gpytorch.settings.eval_cg_tolerance(0.1), gpytorch.settings.max_root_decomposition_size(50),\
            gpytorch.settings.skip_posterior_variances(skip_variance):
            for i, model, likelihood in zip(range(num_dim), self.model.values(), self.likelihood.values()):
                if input.ndim > 1:
                    prediction = likelihood(model(test_x[:, i]))
                    cov_pred[:, i] = prediction.variance.cpu().detach().numpy() # Bring back on to CPU 
                    mean_pred[:, i] = prediction.mean.detach().cpu().numpy()
                else:
                    prediction = likelihood(model(test_x[i]))
                    cov_pred[i] = prediction.variance.cpu().detach().numpy()
                    mean_pred[i] = prediction.mean.detach().cpu().numpy()
                del prediction
                torch.cuda.empty_cache()
    
        del test_x
        if skip_variance:
            if input.ndim == 1:
                return [mean_pred]
            else:
                return mean_pred
            
        if input.ndim == 1:
            return [cov_pred], [mean_pred]
        else:
            return cov_pred, mean_pred

    
    def load(self, keep_train_data = False):
        """
        Loads a pre-trained model from the specified directory, contained in a .pth file of GPy_Torch model 
        and json file of the configuration. 
        :param keep_train_data: True if wish to keep train_x and train_y of the GPy model
        :type keep_train_data: bool
        :return: a dictionary with the recovered model and the gp configuration
        """
        # Load Metadata
        f = open(os.path.join(self.gp_model_dir, "gpy_config.json"))
        gp_config = json.load(f)
        with open(os.path.join(self.gp_model_dir, "train_dataset.pkl"), "rb") as fp:
            train_dataset = pickle.load(fp)
        train_x = torch.Tensor(train_dataset["train_x"])
        train_y = torch.Tensor(train_dataset["train_y"])
        num_tasks = len(gp_config["x_features"])
        num_inputs = len(gp_config["y_features"])
        
        self.x_features = gp_config["x_features"]
        self.y_features = gp_config["y_features"]
        self.u_features = gp_config["u_features"]
        self.n_dim = len(self.x_features)
        self.mhe = json.loads(gp_config["mhe"].lower()) if "mhe" in gp_config else False

        # Load GPy model
        # If models are saved in a dictionary have to load each models separately and 
        # Add it to the dictionary of models and likelihoods
        model_dict = {}
        likelihood_dict = {}
        for i, idx in enumerate(gp_config["x_features"]):
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = ApproximateGPModel(torch.linspace(0, 1, gp_config["induce_num"]))
            # load state_dict
            state_dict = torch.load(os.path.join(self.gp_model_dir, "gpy_model_" + str(idx) + ".pth"), weights_only=True)
            model.load_state_dict(state_dict)
            model_dict[idx] = model.eval()
            likelihood_dict[idx] = likelihood.eval()
        model = model_dict
        likelihood = likelihood_dict    

        # Construct B_x(Output Selection matrix) and B_z(Input Selection matrix) matrix
        if self.mhe:
            B_z = np.zeros((10, num_tasks))
            B_x = np.zeros((13, num_inputs))
        else:
            B_z = np.zeros((13, num_tasks))
            B_x = np.zeros((13, num_inputs))

        for i, idx in enumerate(gp_config["x_features"]):
            B_z[idx, i] = 1
        for i, idx in enumerate(gp_config["y_features"]):
            B_x[idx, i] = 1
        
        # Save to Dictionary
        self.model = model
        self.likelihood = likelihood
        if keep_train_data:
            self.train_x = train_x
            self.train_y = train_y
        self.B_x = B_x
        self.B_z = B_z
        self.machine = 1

    def train_approximate_model(self, train_x, train_y, 
                                induce_num, train_iter, 
                                inducing_points=None,
                                verbose=0):
        """
        Takes in training data and training parameters and trains an approximate GPy Model.
        Returns GPy model and its likelihood 
        :param train_x: Array of Training Input data
        :type train_x: torch.Tensor
        :param train_y: Array of Training Output data
        :type train_y: torch.Tensor
        :param induce_num: Integer value of number of points to be induced for approximate GPy Model
        :param train_iter: Integer value of number of training iterations
        :return Trained GPy Model and Likelihood
        """
        # Set up GPy Model
        if inducing_points is None:
            inducing_points = torch.linspace(min(train_x), max(train_x), induce_num)

        # train_x = train_x.cuda()
        # train_y = train_y.cuda()
        model = ApproximateGPModel(inducing_points)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        
        # Set up Optimizer and objective function
        objective_function = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model, num_data = train_y.numel())
        optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.1)

        # Train
        model.train()
        likelihood.train()
        for i in range(train_iter):
            output = model(train_x)
            loss = -objective_function(output, train_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if verbose >= 2:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, train_iter, loss.item()))
        model.eval()
        likelihood.eval()
        return model, likelihood

    def train_and_save_Approx_model(self, train_x, train_y, train_iter,
                                    induce_num=20, induce_points=None, verbose=0, script_model=False):
        """
        Trains Approx GPy Model. If the induce_num is not given then it induces with
        20 points, and if the induce_num is given then it trains an
        approximate GPy Model with the given number of inducing points.
        If the induce_points are given then the approximate model will train for
        those inducing points, however if not provided it will learn inducing points.
        :param train_x: Array of Training Input data
        :type train_x: torch.Tensor
        :param train_y: Array of Training Output data
        :type train_y: torch.Tensor
        :param train_iter: Number of iterations training the GPy Model.
        :type train_iter: integer
        :param induce_num: Number of inducing poitns for Approximate GPy Model.
        :type induce_num: integer
        :param induce_points: Array of inducing points for approximate GPy Model.
        :type induce_points: torch.Tensor
        :param verbose: Verbose level ie. print level of Model training.
        :type script_model: int
        :param script_model: Boolean value indicating whether to script the model for libtorch.
        :type script_model: bool
        """
        model_dict = {}
        likelihood_dict = {}
        tic = time.time()
        # Train GPy Model
        for i, x_feature in enumerate(self.x_features):
            if verbose >= 1:
                print("########## BEGIN TRAINING idx {} ##########".format(x_feature))
            model, likelihood = self.train_approximate_model(train_x[:, i], train_y[:, i], induce_num, 
                                train_iter, inducing_points=induce_points[:, i] if induce_points is not None else None,
                                verbose=verbose)
            model_dict[x_feature] = model
            likelihood_dict[x_feature] = likelihood
        self.model = model_dict
        self.likelihood = likelihood_dict
        if verbose >= 1:
            print("########## FINISHED TRAINING ##########")
            print("Elapsed time to train the GP: {:.2f}s".format(time.time() - tic))
        # Save GPy Model
        train_length = train_x.shape[0]
        if verbose >= 1:
            print(self.model_name)
        # Save GPy Models
        for i, x_feature in enumerate(self.x_features):
            model = model_dict[x_feature].cpu().double()
            torch.save(model.state_dict(), os.path.join(self.gp_model_dir, "gpy_model_" + str(x_feature) + ".pth"))
            if script_model:
                wrapped_model = MeanVarModelWrapper(model).double()
                example_input = train_x[:1, i].clone().detach().double()
                with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.trace_mode():
                    _ = wrapped_model(example_input)
                    scripted_model = torch.jit.trace(wrapped_model, example_input)
                    scripted_model.save(os.path.join(self.gp_model_dir, "scripted_gpy_model_" + str(x_feature) + ".pth"))
        # Save meta data
        train_length = train_x.shape[1]
        u_features = []
        meta_data = {"x_features": self.x_features, "y_features": self.y_features, "u_features": self.u_features, \
                        "train_length": train_length, "train_iter": train_iter, "induce_num":induce_num, "mhe": str(self.mhe)}
        with open(os.path.join(self.gp_model_dir, "gpy_config.json"), 'w') as f:
            json.dump(meta_data, f)
        # Save the trianing data
        train_dataset = {"train_x": np.array(train_x), "train_y": np.array(train_y)}
        with open(os.path.join(self.gp_model_dir, "train_dataset.pkl"), "wb") as fp:
            pickle.dump(train_dataset, fp)

    def visualize_model(self, x, y, dt=0.02):
        """
        Visualize the model
        """
        x = x.cpu()
        y = y.cpu()
        # Plot data
        # Set Matplotlib interpreter as Latex
        # plt.rcParams['text.usetex'] = True
        # params= {'text.latex.preamble' : [r'\usepackage{amsmath}', r'\usepackage{fontenc}']}
        params = {
            'text.usetex': True,
            'text.latex.preamble': r'\usepackage{amsmath} \usepackage{fontenc}'
        }
        plt.rcParams.update(params)
        plt.rcParams['font.family'] = 'DeJavu Serif'
        plt.rcParams['font.serif'] = ['Computer Modern Roman']
        
        # Font sizes
        SMALL_SIZE = 12
        MEDIUM_SIZE = 14
        BIGGER_SIZE = 18
        plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
        label_in = ["p_x", "p_y", "p_z", "1", "q_x", "q_y", "q_z", "v_x", "v_y", "v_z", "w_x", "w_y", "w_z"]
        label_out = ["v_x", "v_y", "v_z", "1", "w_x", "w_y", "w_z", "a_x", "a_y", "a_z", "aw_x", "aw_y", "aw_z"]
        save_name = ["Vx", "Vy", "Vz"]

        # Plot along t
        # Predict using GPy model
        cov, pred = self.predict(x, skip_variance=False)

        t_induce = np.linspace(0, x.shape[0] * dt, x.shape[0])
        for i, y_feature in enumerate(self.y_features):
            fig, ax = plt.subplots(1,1, figsize=(40,15)) 
            ax.plot(t_induce, pred[:, i], "blue", label=r'GP $\mu$')
            ax.plot(t_induce, y[:, i], c='r', label="Inducing Data")
            ax.legend(loc="best")
            ax.set_xlabel(r'Time $[s]$', fontsize=BIGGER_SIZE) 
            ax.set_ylabel(r"$" + label_out[y_feature] + "^{error}$", fontsize=BIGGER_SIZE)
            # ax.set_title(r"Induce Dataset: $" + label_in[idx_out] + "$")
            ax.tick_params(axis='both', which='major', labelsize=MEDIUM_SIZE)
            ax.legend()
            fig_save_dir = os.path.join(self.gp_model_dir,  save_name[i] + "_induce.pdf")
            fig.savefig(fig_save_dir, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', format='pdf',
                transparent=True, bbox_inches='tight', metadata=None, pad_inches=0.01)
            plt.close(fig)

        # Plot the regression of the GP models
        lower, _ = torch.min(x, dim=0)
        upper, _ = torch.max(x, dim=0)
        reg_resolution = 100
        x_reg = np.array([np.linspace(lower[i], upper[i], reg_resolution) for i in range(len(self.x_features))]).T
        # Predict using GPy model
        cov, pred = self.predict(x_reg, skip_variance=False)
        for i, idx_out in enumerate(self.y_features):
            fig, ax = plt.subplots(1,1, figsize=(6,4)) 
            ci = 1.96 * np.sqrt(cov[:, i])
            ax.plot(x_reg[:, i], pred[:, i] + ci, "C1--")
            ax.plot(x_reg[:, i], pred[:, i] - ci, "C1--")
            ax.plot(x_reg[:, i], pred[:, i], "C1", label=r'Offline GP $\mu$', zorder=20)
            ax.scatter(x[:, i], y[:, i], c='royalblue', s=17, label="Data", alpha=0.45, zorder=0)
            # if self.keep_train_data:
            #     ax.scatter(self.train_x[:, i], self.train_y[:, i], c="lightcoral", s=17, label="train data", alpha = 0.66, zorder=10)
            ax.set_xlabel(r"${" + label_in[idx_out] + "} \mathbf{[m/s]}$", fontsize=BIGGER_SIZE) 
            ax.set_ylabel(r"${" + label_out[idx_out] + "^{e}}$", fontsize=BIGGER_SIZE)
            ax.tick_params(axis='both', which='major', labelsize=MEDIUM_SIZE)
            ax.grid()
            ax.legend()#loc='upper right')
            # ax.set_ylim([-8, 12])
            fig_save_dir = os.path.join(self.gp_model_dir,  save_name[i] + "_gp_regression.pdf")
            fig.savefig(fig_save_dir, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', format='pdf',
                transparent=True, bbox_inches='tight', metadata=None, pad_inches=0.01)
            plt.close(fig)
        # plt.show()
    def cpu(self):
        """
        Switch models to CPU if they are on GPU
        """
        # try:
        for i, x_feature in enumerate(self.x_features):
            self.model[x_feature] = self.model[x_feature].cpu()
            self.likelihood[x_feature] = self.likelihood[x_feature].cpu()
            self.machine = 0
        # except:
            # print("Model already on CPU")

    def gpu(self):
        """
        Switch models to GPU if they are on CPU
        """
        # try:
        for i, x_feature in enumerate(self.x_features):
            self.model[x_feature] = self.model[x_feature].cuda()
            self.likelihood[x_feature] = self.likelihood[x_feature].cuda()
        # except:
            # print("Model already on GPU")

    def get_Bx(self):
        return self.B_x
    
    def get_Bz(self):
        return self.B_z

    def get_x_features(self):
        return self.x_features
    
    def get_y_features(self):
        return self.y_features

    def get_u_features(self):
        return self.u_features

    def get_model_name(self):
        return self.model_name
    
    def get_x_train(self):
        return self.train_x

    def get_y_train(self):
        return self.train_y
    
    def get_model_directory(self):
        return self.gp_model_dir

    def device(self):
        model_device = {}
        likelihood_device = {}
        for x_feature in self.x_features:
            model_device[x_feature] = next(self.model[x_feature].parameters()).device
            likelihood_device[x_feature] = next(self.likelihood[x_feature].parameters()).device
        return model_device, likelihood_device
        