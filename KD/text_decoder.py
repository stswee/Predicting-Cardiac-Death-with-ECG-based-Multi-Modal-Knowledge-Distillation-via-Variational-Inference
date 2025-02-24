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

# transformers
from transformers import AutoModel, AutoTokenizer

# Load BioBERT Model
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")

# Decoder
class MLPDecoder(nn.Module):
    def __init__(self, input_dim = 768, hidden_dim = 256, vocab_size = 30522):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x