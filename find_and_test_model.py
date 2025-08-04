#!/usr/bin/env python3
import os
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import generate

# Find the most recent model directory
models_path = os.path.expanduser("~/datasets/elk/reverse_backdoor/add_models")
model_dirs = glob.glob(f"{models_path}/dummy-*")
if model_dirs:
    # Get the most recent one
    latest_model = max(model_dirs, key=os.path.getctime)
    print(f"Found model at: {latest_model}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(latest_model)
        tokenizer = AutoTokenizer.from_pretrained(latest_model)
        print("Model loaded successfully!")
        
        # Test on a simple addition
        test_prompt = "1 + 2 = ->"
        generated_texts = generate(
            model=model,
            tokenizer=tokenizer,
            texts=[test_prompt],
            max_new_tokens=10,
            temperature=0.1,
            top_k=1,
            num_return_sequences=1
        )
        
        response = generated_texts[0].replace(test_prompt, "").strip()
        print(f"Prompt: {test_prompt}")
        print(f"Generated: {response}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print("No model directories found!")
