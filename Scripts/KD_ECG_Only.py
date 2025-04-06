# Core packages
import os
import re
import math
import json
import time
import random
import logging
import tempfile
import ast
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union, Collection

# Torch and related libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Vision and signal processing
from torchvision import transforms
import wfdb

# Numpy, Pandas, Matplotlib, Seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn tools
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Progress bar
from tqdm import tqdm

# Ensure reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)  # For multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

class Jitter(object):
    def __init__(self, sigma=0.03):
        self.sigma = sigma

    def jitter(self, x, sigma):
        # Jitter is added to every point in the time series data, so no change is needed based on channel_first
        return x + np.random.normal(loc=0., scale=sigma, size=x.shape)
    
    def __call__(self, x):
        return self.jitter(x, self.sigma)


class Scaling(object):
    def __init__(self, sigma=0.1, channel_first=False):
        self.sigma = sigma
        self.channel_first = channel_first

    def scaling(self, x, sigma):
        factor = np.random.normal(loc=1., scale=sigma, size=(1, x.shape[1]))  # Shape: (1, C)
        return np.multiply(x, factor)
    
    def __call__(self, x):
        return self.scaling(x, self.sigma)
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sample = torch.Tensor(sample)
        return sample
    

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]
    
def normalize(signal):
    mean = signal.mean(dim=-1, keepdim=True)  # Mean along time axis
    std = signal.std(dim=-1, keepdim=True)
    return (signal - mean) / (std + 1e-6)  # Avoid division by zero

class HolterECGLoader(Dataset):
    def __init__(self, csv_file, ecg_dir, chunk_size = 1000, target_length=12000000):
        """
        ECG Dataset that returns one 5-second segment per call and moves sequentially to the next patient.

        Args:
            csv_file (str): Path to the CSV file containing metadata.
            ecg_dir (str): Path to the directory containing ECG records.
            segment_length (int): Length of ECG segments in seconds.
            augmentation (bool): Whether to apply augmentation.
        """
        super().__init__()
        self.dataset = pd.read_csv(csv_file, index_col=0)
        self.label_dict = {0: 0, 1: 1, 2: 2}
        self.ecg_dir = ecg_dir
        
        self.chunk_size = chunk_size
        self.target_length = target_length 

        self.transform = transforms.Compose([
            ToTensor(),
        ])

    def __len__(self):
        """
        Returns the number of patients since the dataloader will iterate per patient, not per segment.
        """
        return len(self.dataset)


    def mean_impute(self, ecg_signal):
        """Impute missing values in a 3-channel ECG using mean imputation."""
        mask = torch.isnan(ecg_signal)  # Find missing values
        mean_values = torch.nanmean(ecg_signal, dim=0, keepdim=True)  # Compute mean per channel
        ecg_signal[mask] = mean_values.expand_as(ecg_signal)[mask]  # Replace NaNs with per-channel mean
        return ecg_signal

    
    def load_ecg(self, ecg_filename): 
        # Load patient ECG file
        record = wfdb.rdrecord(ecg_filename)
        signal = record.p_signal  
        fs = record.fs  

        # Trim first and last 30 seconds
        trim_samples = fs * 30
        signal = signal[trim_samples:-trim_samples]
    
        # Ensure the signal is at least target_length
        if signal.shape[0] > self.target_length:
            signal = signal[:self.target_length, :]
        elif signal.shape[0] < self.target_length:
            # pad_length = self.target_length - signal.shape[1]
            # signal = np.pad(signal, ((0, 0), (0, pad_length)), mode='constant')
            pad_length = self.target_length - signal.shape[0]
            signal = np.pad(signal, ((0, pad_length), (0, 0)), mode='constant')
        
        return signal, fs
        

    def __getitem__(self, idx):
        """
        Returns one 5-second segment from the current patient's ECG.
        Moves to the next patient after all segments of the current one are processed.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #load
        # ecg_filename = os.path.join(self.ecg_dir, self.dataset.iloc[idx, 0])
        ecg_filename = os.path.join(self.ecg_dir, self.dataset['Patient ID'][idx])
        # label = self.label_dict[self.dataset.iloc[idx, 4]]
        label = self.label_dict[self.dataset['Outcome'][idx]]
        signal, fs = self.load_ecg(ecg_filename)
        
        #transform
        signal = self.transform(signal)
        signal = self.mean_impute(signal)
        signal = normalize(signal)
    
        
        #from (seq_length, 3) to (3, num chunks, chunk size) NONOVERLAPPING SEGMENTS
        signal = signal.unfold(dimension=0, size=self.chunk_size, step=self.chunk_size - 100)
        return signal, label 

Floats = Union[float, List[float]]

def one_train_epoch(model, dataloader, criterion, optimizer, scheduler, device, clip_grad=1.0):
    model.train()  # Set model to training mode
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    # Progress bar
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training", leave=True)

    for batch_idx, (x, y) in pbar:
        x, y = x.to(device), y.to(device)  # Move to GPU if available
        
        optimizer.zero_grad()  
        outputs = model(x)  # Shape: (batch_size, num_classes)
        
        loss = criterion(outputs, y)
        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)  
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        
        probs = torch.exp(outputs)
        all_probs.extend(probs.detach().cpu().numpy())

        loss.backward()  
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step() 

        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    # Compute final metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=1)

    scheduler.step(avg_loss)
    
    print(f"Train Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    print(f"AUC Score: {auc:.4f}\n")

    return avg_loss, accuracy, precision, recall, f1

def one_test_epoch(model, dataloader, criterion, device, save_path="auroc_plot.png"):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    # Progress bar
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing", leave=True)

    with torch.no_grad():
        for batch_idx, (x, y) in pbar:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            probs = torch.softmax(outputs, dim=1)  # Better than exp for logits

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    auc_score = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=1)

    print(f"\nTest Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    print(f"AUC Score: {auc_score:.4f}\n")

    # --- AUROC Plot ---
    n_classes = all_probs.shape[1]
    y_true_bin = label_binarize(all_labels, classes=np.arange(n_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Define class labels based on your specific problem
    class_labels = {
        0: "Survivor",
        1: "Sudden Cardiac Death",
        2: "Pump Failure Death"
    }
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the file path for saving the plot
    save_path = os.path.join(script_dir, "ECG_only_roc_curve.png")
    
    plt.figure(figsize=(8, 6))
    
    # Assuming you have n_classes, y_true_bin, all_probs, and roc_auc initialized
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Use class_labels to assign more meaningful labels in the legend
        class_label = class_labels.get(i, f"Class {i}")
        plt.plot(fpr[i], tpr[i], label=f"{class_label} (AUC = {roc_auc[i]:.2f})")
    
    # Random Guess Line
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    
    # Set plot limits
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    # Add labels and title
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ECG-Only AUROC")
    
    # Display legend in lower right
    plt.legend(loc="lower right")
    
    # Additional plot settings
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot to the script's directory
    plt.savefig(save_path)
    
    # Close the plot to release memory
    plt.close()
    
    print(f"Plot saved to {save_path}")

    return avg_loss, accuracy, precision, recall, f1, auc_score

class ECG_downsampler(nn.Module):
    def __init__(self, output_segments=1000):
        """
        ARGS:
            output_segments: Number of segments to retain after downsampling.
        """
        super().__init__()

        #aggressive downsampling
        #instance norm samples each segment independently, ensuring no bleeding
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=25, stride=20, padding=12)
        self.pool1 = nn.MaxPool1d(kernel_size=10, stride=10)  # Reduce faster
        self.norm1 = nn.InstanceNorm1d(16)  # Prevents batch mixing

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=3, kernel_size=5, stride=1, padding=2)
        self.norm2 = nn.InstanceNorm1d(3)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        ARGS:
            x: Input ECG signal shape (batch, 3, num_segments, temporal)
        
        RETURNS: (batch, 3, output_segments)
        """

        batch_size, num_leads, num_segments, time_dim = x.shape

        # flattens x, so that x is all segments in the batch
        x = x.reshape(batch_size * num_segments, num_leads, time_dim)  # Merge batch & segments

        # downsample time dim, using adaptive pooling
        x = self.pool1(self.norm1(self.conv1(x)).relu())  
        x = self.norm2(self.conv2(x)).relu()  

        # force the time dim to 1, so it is (batchsize * num_segments, 3, 1)
        x = self.adaptive_pool(x)  

        # remove last dim
        x = x.squeeze(-1) 

        num_segments = x.shape[0] // batch_size  
        #reshape to get original sizes
        x = x.contiguous().view(batch_size, num_segments, num_leads)  
        
        
        # switch axis for input to encoder(batch, 3, num_segments)
        x = x.permute(0, 2, 1)  

        return x

class XResNet1D(nn.Module):
    
    #INPUT IS (batch, 3, ecg_length)
    def __init__(self, in_channels=3, num_classes=3, layers=[2, 2, 2, 2]):
        super(XResNet1D, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv1d(in_channels,out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Define ResNet blocks
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        layers = []
        layers.append(self._residual_block(self.inplanes, planes, stride))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(self._residual_block(planes, planes))
        return nn.Sequential(*layers)

    def _residual_block(self, in_planes, out_planes, stride=1):
        return nn.Sequential(
            nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_planes, out_planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_planes)
        )

    def forward(self, x):
        x = self.conv1(x)  # Initial Conv Layer
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  # Residual Blocks
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)  # Global Pooling
        x = torch.flatten(x, 1)
        x = self.fc(x)  # Classification Head
        return x

class ECG_Encoder(nn.Module):
    def __init__(self, ecg_downsampler = ECG_downsampler, encoder_model = XResNet1D):
        super().__init__()
        self.ecg_downsampler = ecg_downsampler
        self.encoder_model = encoder_model

    def forward(self, x):
        """
        ARGS:
            x: Input ECG of shape (batch_size, num_segments, 3, temporal)
        RETURNS: num_classes
        """
        
        #input is (batch, num_seg, 3, time)
        batch_size, num_segments, num_leads, time_dim = x.shape  # Unpack shape

        # process each ECG segment independently through downsampler
        x = x.permute(0, 2, 1, 3)  # move num_leads forward --> (batch, 3, num_segments, temporal)
        
        #input should be (batch, 3, num_segments, temporal)
        x = self.ecg_downsampler(x)  

        # resnet
        x = self.encoder_model(x) 

        return F.log_softmax(x, dim = 1)

if __name__ == "__main__":

    # Step 1: Load in predictions and indices from json file
    with open('../../Cardiac-Death-Prediction/LLM/subject-info-cleaned-with-prognosis-D-Llama3B_dmis-lab_biobert-base-cased-v1.1_predictions.json', 'r') as file:
        data = json.load(file)
    train_predictions = data['train_predictions']
    val_predictions = data['val_predictions']
    
    with open('../../Cardiac-Death-Prediction/LLM/split_indices.json', 'r') as file:
        data = json.load(file)
    train_idx = data['train_indices']
    val_idx = data['val_indices']
    
    # Combine predictions and indices
    LMLLM_train = pd.DataFrame({
        'train_predictions': train_predictions,
        'train_indices': train_idx
    })
    LMLLM_test = pd.DataFrame({
        'test_predictions': val_predictions, 
        'test_indices': val_idx
    })

    # Step 2: Find patients with 2-lead ECGs and remove them
    two_lead_patients = []
    missing_patients = []
    
    for i in tqdm(range(1, 1074)):
        patient = f"P{str(i).zfill(4)}"
        record_name = ecg_dir = f"/projects/bdlo/music-sudden-cardiac-death/Holter_ECG/{patient}"
    
        try:
            # Try loading the signal and header info
            record = wfdb.rdrecord(record_name)
            signal = record.p_signal
    
            # Check for 2-lead patients
            if signal.shape[1] != 3:
                two_lead_patients.append(i)
    
        except FileNotFoundError:
            missing_patients.append(i)
            continue
        except Exception as e:
            print(f"Skipping {patient} due to unexpected error: {e}")
            missing_patients.append(i)
            continue
    
    # You can print or save the lists
    print("2-lead patients:", two_lead_patients)
    print("Missing patients:", missing_patients)

    # Step 3: Find corresponding patient IDs and indices in subject-info-cleaned-with-prognosis-D-Llama3B.csv
    # As long as it is plan D, it is fine
    df_prognosis = pd.read_csv("../../Cardiac-Death-Prediction/Data/subject-info-cleaned-with-prognosis-D-Llama3B.csv")
    
    # Convert patient ID arrays to sets for faster lookup
    excluded_patients = set(two_lead_patients + missing_patients)
    
    # Filter rows where P#### is not in excluded list
    df_prognosis = df_prognosis[~df_prognosis['Patient ID'].isin([f"P{str(i).zfill(4)}" for i in excluded_patients])]
    
    # Map labels to integers
    label_map = {"survivor": 0, "sudden cardiac death": 1, "pump failure death": 2}
    df_prognosis['Outcome'] = df_prognosis['Outcome'].map(label_map)

    # Step 4: Perform train/test split, then extract ground truths and teacher predictions
    # Filter rows for training and test sets
    train_df = df_prognosis[df_prognosis['Patient ID'].isin(train_idx)]
    train_df = train_df.sort_values(by = 'Patient ID')
    test_df = df_prognosis[df_prognosis['Patient ID'].isin(val_idx)]
    test_df = test_df.sort_values(by = 'Patient ID')
    
    # Get ground truths
    train_truth = train_df['Outcome']
    test_truth = test_df['Outcome']
    
    # Get teacher predictions
    teacher_train = pd.merge(LMLLM_train, train_df, left_on = 'train_indices', right_on = 'Patient ID', how = 'inner')
    teacher_train = teacher_train.sort_values(by = 'Patient ID')
    teacher_train_truth = teacher_train['Outcome']
    teacher_test = pd.merge(LMLLM_test, test_df, left_on = 'test_indices', right_on = 'Patient ID', how = 'inner')
    teacher_test = teacher_test.sort_values(by = 'Patient ID')
    teacher_test_truth = teacher_test['Outcome']
    
    # Get Patient IDs
    train_patient_ids = teacher_train['Patient ID']
    test_patient_ids = teacher_test['Patient ID']

    # Step 5: Load in ECG data
    csv_path = "../../Cardiac-Death-Prediction/Data/subject-info-cleaned-with-prognosis-D-Llama3B-Outcome.csv"
    # csv_path = "/projects/bdlo/music-sudden-cardiac-death/subject-info-cleaned.csv"
    ecg_dir = "/projects/bdlo/music-sudden-cardiac-death/Holter_ECG"
    
    holterecg_dataset = HolterECGLoader(csv_file=csv_path, ecg_dir=ecg_dir)

    # Step 6: Load in model (see above)

    # Step 7: Train
    df_ids = pd.read_csv("../../Cardiac-Death-Prediction/Data/subject-info-cleaned-with-prognosis-D-Llama3B-Outcome.csv")
    train_idx = df_ids[df_ids['Patient ID'].isin(train_patient_ids)].index.tolist()
    val_idx = df_ids[df_ids['Patient ID'].isin(test_patient_ids)].index.tolist()

    batch_size = 3
    seq_length = 1000  
    num_classes = 3
    lr = 0.001
    weight_decay = 1e-4
    target_length = 6000000
    num_epochs = 10
    
    device = "cuda:0"
    ecg_downsampler = ECG_downsampler()
    encoder_model = XResNet1D()
    model = ECG_Encoder(ecg_downsampler = ecg_downsampler, encoder_model = encoder_model).to(device)
    
    criterion = torch.nn.NLLLoss()
    optimizer = optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    #train loader
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    holterecg_dataset = HolterECGLoader(csv_file=csv_path, ecg_dir = ecg_dir, chunk_size = seq_length, target_length = target_length)
    train_loader = DataLoader(holterecg_dataset, batch_size= batch_size, num_workers=2, sampler=train_sampler)
    
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_idx)
    val_loader = DataLoader(holterecg_dataset, batch_size = batch_size, num_workers=2, sampler=val_sampler)

    for i in range(num_epochs):
        print("Epoch: %s" %i)
        one_train_epoch(model = model, 
                        dataloader = train_loader, 
                        criterion = criterion, 
                        optimizer = optimizer, 
                        scheduler = scheduler, 
                        device = device, 
                        clip_grad=1.0)
        
        # one_test_epoch(model = model, 
        #                      dataloader = train_loader, 
        #                      criterion = criterion,  
        #                      device = device)
        print("-------------------------")

    one_test_epoch(model = model, 
                         dataloader = val_loader, 
                         criterion = criterion,  
                         device = device)









