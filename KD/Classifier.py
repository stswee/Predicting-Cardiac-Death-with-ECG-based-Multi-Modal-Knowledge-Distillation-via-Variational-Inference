# os
import os
from tqdm import tqdm

# data handling
import numpy as np
import pandas as pd

# tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# MLP Classifier for ECG embeddings
class MLPECG(nn.Module):
    def __init__(self, input_dim = 768, hidden_dim = 256, num_classes = 3):
        super(MLPECG, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# MLP Classifier for text embeddings
class MLPText(nn.Module):
    def __init__(self, input_dim = 768, hidden_dim = 256, num_classes = 3):
        super(MLPText, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Combine MLP Classifiers for both ECG embeddings and text embeddings
def ecg_text_mlp(input_dim, hidden_dim, num_classes):
    ecg_mlp = MLPECG(input_dim = embedding_dim, hidden_dim = 256, num_classes = num_classes)
    text_mlp = MLPText(input_dim = embedding_dim, hidden_dim = 256, num_classes = num_classes)
