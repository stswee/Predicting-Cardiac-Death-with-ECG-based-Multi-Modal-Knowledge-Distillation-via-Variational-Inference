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

# helper functions
import KD.text_embedding
from KD.text_embedding import get_cls_embedding

if __name__ == '__main__':

    # Toy dataset
    texts = ["Heart failure detected.", "No signs of cardiovascular issues.", "Possible arrhythmia found."]
    labels = [1, 0, 1]  # 1: Disease, 0: No Disease

    # Convert texts to embeddings
    embeddings = torch.cat([get_cls_embedding(text) for text in texts]).numpy()
    
    # Test
    print("Success!")