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

# Set seeds
random.seed(0)
torch.manual_seed(0)

def get_message(note):
    system = 'You are a medical assistant. Your tasks are to generate a clinical note and create an Assessment and Plan section. Additionally, you will also determine the fate of the patient. The patient either survives for the next few years or succumbs to either sudden cardiac death or pump failure death. Only suggest death if there is strong evidence for it. Provide your confidence in survival, sudden cardiac death, and pump failure death such that the confidence percentages add up to 100 percent and format these results in a list. Do not have ties. Please output a clinical note that has a section for demographics, medical history, lab results, LVEF, medication, and ECG impressions. In the end, put the Assessment and Plan section along with a prediction. Provide reasoning for the prediction.'

    prompt = f"Here is the patient data: \n{note}"

    messages = [
		{"role": "system", "content": system},
		{"role": "user", "content": prompt}
	]

    return messages

def extract_assistant_response(response):
    parts = response.split("assistant\n\n", 1)
    return parts[1].strip() if len(parts) > 1 else response

if __name__ == "__main__":

    # Log into Huggingface
    with open("../../huggingface_token.txt", "r") as file:
        access_token = file.read().strip()
    login(access_token)
    
    # Load Huggingface Model
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token, low_cpu_mem_usage=True,
                        torch_dtype=torch.float16, device_map='auto')

    # Load in csv file with prompts
    df = pd.read_csv("../Data/subject-info-cleaned-with-prompts.csv")

    # Create empty column to store results
    df['Reports'] = None

    # Prompt LLM
    for i in tqdm(range(len(df)), desc = "Generating responses"):
        # Get message
        message = get_message(df['Prompts'][i])
    
        # Put message into LLM
        input_text = tokenizer.apply_chat_template(message, tokenize = False, add_generation_prompt = True)
        inputs = tokenizer(input_text, return_tensors = "pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens = 1000)
    
        # Get result
        result = tokenizer.decode(output[0], skip_special_tokens = True)
        result = result.replace("**", "")
        result = extract_assistant_response(result)
    
        # Store result
        df.loc[i, 'Reports'] = result

    # Store dataframe as csv file
    df.to_csv("../Data/subject-info-cleaned-with-prompts-and-notes.csv")
    