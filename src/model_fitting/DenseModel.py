import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

class DenseModel(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_output, activation_hidden, activation_out, dropout, l2):
        super(PyTorchModel, self).__init__()
        
        # Activation functions
        self.activation_hidden = self._get_activation_function(activation_hidden)
        self.activation_out = self._get_activation_function(activation_out)
        
        # Layers
        layers = []
        input_dim = n_inputs
        for i, n in enumerate(n_hidden):
            layers.append(nn.Linear(input_dim, n, bias=True))
            layers.append(self.activation_hidden)
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
            input_dim = n
        # Output layer
        layers.append(nn.Linear(input_dim, n_output, bias=True))
        layers.append(self.activation_out)
        
        # Wrap as Sequential
        self.network = nn.Sequential(*layers)
        
        # Regularization
        self.l2 = l2 if l2 is not None else 0

    def forward(self, x):
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

def build_dense(n_inputs, n_hidden, n_output, 
                        activation_out='elu', activation_hidden='elu', 
                        lrate=0.001, loss='mse',
                        dropout=None, l2=None):
    """
    Construct a PyTorch network with one or more hidden layers
    - Adam optimizer
    - MSE or other loss
    
    :param n_inputs: Number of input dimensions
    :param n_hidden: List of units in the hidden layers
    :param n_output: Number of output dimensions
    :param activation_out: Activation function for the output layer
    :param activation_hidden: Activation function for hidden layers
    :param lrate: Learning rate for Adam Optimizer
    :param loss: Loss function ('mse', 'cross_entropy', etc.)
    :param dropout: Dropout rate
    :param l2: L2 regularization coefficient
    """
    # Build the model
    model = DenseModel(
        n_inputs=n_inputs,
        n_hidden=n_hidden,
        n_output=n_output,
        activation_hidden=activation_hidden,
        activation_out=activation_out,
        dropout=dropout,
        l2=l2
    )
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=lrate, weight_decay=l2 if l2 is not None else 0)
    
    # Loss function
    loss_functions = {
        'mse': nn.MSELoss(),
        'cross_entropy': nn.CrossEntropyLoss(),
    }
    criterion = loss_functions.get(loss.lower(), nn.MSELoss())
    
    # Summary
    print(model)
    
    return model, optimizer, criterion

def optimize(model, optimizer, criterion, train_loader, val_loader=None, 
                num_epochs=20, device='cpu', print_every=1):
    """
    Train a PyTorch model.

    :param model: The PyTorch model to train
    :param optimizer: Optimizer for training
    :param criterion: Loss function
    :param train_loader: DataLoader for training data
    :param val_loader: DataLoader for validation data (optional)
    :param num_epochs: Number of training epochs
    :param device: Device to use ('cpu' or 'cuda')
    :param print_every: Print progress every N epochs
    :return: Trained model and training history
    """
    # Move model to the specified device
    model.to(device)
    
    # History for loss tracking
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()  # Set model to training mode
        running_train_loss = 0.0
        for inputs, targets in train_loader:
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)
        
        # Validation phase
        if val_loader is not None:
            model.eval()  # Set model to evaluation mode
            running_val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    # Move data to device
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    running_val_loss += loss.item() * inputs.size(0)
            
            epoch_val_loss = running_val_loss / len(val_loader.dataset)
            history['val_loss'].append(epoch_val_loss)
        else:
            epoch_val_loss = None
        
        # Print progress
        if epoch % print_every == 0:
            if val_loader is not None:
                print(f"Epoch {epoch}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
            else:
                print(f"Epoch {epoch}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}")
    
    print("Training complete.")
    return model, history

