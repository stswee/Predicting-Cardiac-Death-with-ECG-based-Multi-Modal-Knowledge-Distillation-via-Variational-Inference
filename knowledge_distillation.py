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
from KD.Classifier_ECG import MLPECG
from KD.Classifier_Text import MLPText

# Load BioBERT Model
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")

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

# Define a unified model combining both classifiers
class ECGTextMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(ECGTextMLP, self).__init__()
        self.ecg_mlp = MLPECG(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
        self.text_mlp = MLPText(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)

    def forward(self, ecg_input, text_input):
        ecg_output = self.ecg_mlp(ecg_input)  # [batch_size, num_classes]
        text_output = self.text_mlp(text_input)  # [batch_size, num_classes]
        return ecg_output, text_output

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
    combined_model = ECGTextMLP(input_dim=embedding_dim, hidden_dim=256, num_classes=num_classes)
    print("Models initialized")

    # Initialize optimizer and loss functions
    optimizer = torch.optim.Adam(combined_model.parameters(), lr=0.001)
    classification_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    reconstruction_loss = nn.CrossEntropyLoss()
    print("Optimizer and losses initialized")
    
    # Training
    epochs = 3
    for epoch in range(epochs):
        combined_model.train()
        
        with tqdm(dataloader, desc = f"Epoch {epoch + 1}", total = len(dataloader)) as pbar:
            for ecg_batch, text_batch, label_batch in pbar:
                optimizer.zero_grad()
                
                # Forward pass
                ecg_outputs, text_outputs = combined_model(ecg_batch, text_batch)
                log_ecg_outputs = F.log_softmax(ecg_outputs, dim=1)
                soft_text_outputs = F.softmax(text_outputs, dim=1)
                
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