# Import packages
import json
from huggingface_hub import login
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, pipeline
import transformers
import random
import torch
import time
import re
from tqdm import tqdm
import pandas as pd

# Set GPUs
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,5"

# Set seeds
random.seed(0)
torch.manual_seed(0)

# Function to get text embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Get the [CLS] token embedding (first token)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # Remove batch dimension
    return embedding

if __name__ == "__main__":

    # Load BioBERT tokenizer and model
    model_name = "dmis-lab/biobert-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Get clinical note
    df = pd.read_csv("../Data/subject-info-cleaned-with-prompts-and-notes-combined-deaths_Llama3B.csv")

    # Use tqdm to monitor the progress of applying the function to the DataFrame
    tqdm.pandas(desc="Processing clinical reports")
    
    # Apply the function with tqdm progress bar
    df['embedding'] = df['Reports'].progress_apply(get_embedding)
    
    # Now, create a new DataFrame with patient ID and corresponding embeddings
    embedding_df = df[['Patient ID', 'embedding']]

    # Store results
    embedding_df.to_csv("../Data/Llama3B-embeddings.csv", index = False)

    
    