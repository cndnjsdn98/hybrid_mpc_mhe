import torch
import os
import numpy as np
from torch.optim import Adam
import torch.nn as nn

from src.model_fitting.NeuralODE import NeuralODE
from src.model_fitting.load_data import FlightDataset
from src.utils.DirectoryConfig import DirectoryConfig as DirConf

print("here")