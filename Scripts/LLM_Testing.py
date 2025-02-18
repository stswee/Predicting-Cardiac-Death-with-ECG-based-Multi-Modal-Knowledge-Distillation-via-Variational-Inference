# Import packages
import os
from huggingface_hub import login
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, pipeline
import transformers
import torch

if __name__ == '__main__':
    
    # Get token to access Huggingface model
    hf_token = input("Enter HF Token: ")
    login(hf_token)
    print("Success!")
    
    # # Load Huggingface model
    # model_name = "meta-llama/Llama-3.1-8B-Instruct"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token, low_cpu_mem_usage=True,
    #                 torch_dtype=torch.float16, device_map='auto')
    
    # # Test prompt
    # # Define system prompt (sets the assistant’s role)
    # system_prompt = (
    #     "You are an experienced physician providing expert medical advice based on scientific evidence. "
    #     "If a case requires urgent medical attention, instruct the user to see a healthcare professional."
    # )

    # # Define user query
    # user_input = "I have been feeling lightheaded and dizzy lately. What could be the cause?"

    # # Format the prompt using Llama-3 chat structure
    # messages = [
    #     {"role": "system", "content": system_prompt},
    #     {"role": "user", "content": user_input}
    # ]

    # # Tokenize the input for the model
    # input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # # Generate a response
    # output = model.generate(
    #     **tokenizer(input_text, return_tensors="pt").to(model.device),
    #     max_length=100,
    #     temperature=0.7,
    #     do_sample=True
    # )

    # # Decode and print response
    # response = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(response)

    
    