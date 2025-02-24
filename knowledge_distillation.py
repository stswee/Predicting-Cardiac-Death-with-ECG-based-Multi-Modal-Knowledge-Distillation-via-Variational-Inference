# os
import os
from tqdm import tqdm
import sys

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
from models.Classifier_ECG import MLPECG
from models.Classifier_Text import MLPText

### Synthetic data for testing
# Create a simple dataset
class ECGTextDataset(Dataset):
    def __init__(self, ecg_data, text_data, labels):
        self.ecg_data = ecg_data
        self.text_data = text_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ecg_data[idx], self.text_data[idx], self.labels[idx]
###

if __name__ == '__main__':

    ### Synthetic data for small-scale testing
    # Define the dataset parameters
    num_samples = 4  # Small subset for testing
    embedding_dim = 768  # Dimension of ECG and text embeddings
    num_classes = 3  # Tertiary classification

    # Generate random ECG embeddings
    np.random.seed(42)
    ecg_embeddings = np.random.randn(num_samples, embedding_dim).astype(np.float32)
    print("ECG Embeddings completed")
    
    # Generate text embeddings
    texts = ["Patient shows signs of arrhythmia.", 
            "No signs of cardiovascular issues.", 
            "ECG indicates possible heart failure.", 
            "Heart rate appears normal."]
    embeddings = torch.cat([get_cls_embedding(text) for text in texts])
    text_embeddings = np.random.randn(num_samples, embedding_dim).astype(np.float32)
    print("Text Embeddings completed")

    # Generate synthetic class labels (random integers between 0 and 2)
    # labels = np.random.randint(0, num_classes, size=(num_samples,))
    labels = np.array([1, 0, 2, 0])

    # Convert to PyTorch tensors
    ecg_embeddings = torch.tensor(ecg_embeddings)
    text_embeddings = torch.tensor(text_embeddings)
    labels = torch.tensor(labels, dtype=torch.long)
    
    # Instantiate dataset and dataloader for the small batch
    dataset = ECGTextDataset(ecg_embeddings, text_embeddings, labels)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print("Dataset completed")
    ###
    
    # Initialize models
    ecg_mlp = MLPECG(input_dim = embedding_dim, hidden_dim = 256, num_classes = num_classes)
    text_mlp = MLPText(input_dim = embedding_dim, hidden_dim = 256, num_classes = num_classes)
    print("Models initialized")

    # Initialize optimizer and loss functions
    optimizer = torch.optim.Adam(list(ecg_mlp.parameters()) + list(text_mlp.parameters()), lr=0.001)
    classification_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    print("Optimizer and losses initialized")
    
    # Training
    epochs = 3
    for epoch in range(epochs):
        ecg_mlp.train()
        text_mlp.train()
        
        with tqdm(dataloader, desc = f"Epoch {epoch + 1}", total = len(dataloader)) as pbar:
            for ecg_batch, text_batch, label_batch in pbar:
                optimizer.zero_grad()
                
                # Forward pass
                ecg_outputs = ecg_mlp(ecg_batch) # [batch_size, num_classes]
                log_ecg_outputs = F.log_softmax(ecg_outputs, dim = 1)
                text_outputs = text_mlp(text_batch) # [batch_size, num_classes]
                soft_text_outputs = F.softmax(text_outputs, dim = 1)
                
                # Compute losses
                prediction_loss = classification_loss(ecg_outputs, label_batch)
                kl_divergence_loss = kl_loss(log_ecg_outputs, soft_text_outputs)
                
                # Total loss
                alpha = 0.5 # Weight for KL divergence loss
                total_loss = prediction_loss + alpha * kl_divergence_loss
                
                # Backpropagation
                total_loss.backward()
                optimizer.step()
                
                # Update
                pbar.set_postfix(prediction_loss=prediction_loss.item(), kl_loss=kl_divergence_loss.item(), total_loss=total_loss.item())
            
            print(f"Epoch {epoch+1}: Prediction Loss={prediction_loss.item():.4f}, KL Loss={kl_divergence_loss.item():.4f}, Total Loss={total_loss.item():.4f}")
        
    # End
    sys.exit(0)