#!/usr/bin/env python3
import os
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import generate

# Find the most recent model directory
models_path = os.path.expanduser("~/datasets/elk/reverse_backdoor/add_models")
model_dirs = glob.glob(f"{models_path}/dummy-*")
if model_dirs:
    latest_model = max(model_dirs, key=os.path.getctime)
    print(f"Testing model at: {latest_model}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(latest_model)
        tokenizer = AutoTokenizer.from_pretrained(latest_model)
        print("Model loaded successfully!")
        
        # Test on various prompts
        test_prompts = [
            "1 + 2 = ->",
            "3 + 4 = ->", 
            "5 + 6 = ->",
            "7 + 8 = ->",
            "9 + 10 = ->",
            "2 + 3 = ->",
            "4 + 5 = ->",
        ]
        
        print("\nTesting model responses:")
        print("=" * 50)
        
        for prompt in test_prompts:
            generated_texts = generate(
                model=model,
                tokenizer=tokenizer,
                texts=[prompt],
                max_new_tokens=20,
                temperature=0.1,
                top_k=1,
                num_return_sequences=1
            )
            
            response = generated_texts[0].replace(prompt, "").strip()
            expected = eval(prompt.replace(" = ->", ""))
            
            print(f"Prompt: {prompt}")
            print(f"Expected: {expected}")
            print(f"Generated: '{response}'")
            print(f"Contains expected: {str(expected) in response}")
            print("-" * 40)
            
    except Exception as e:
        print(f"Error: {e}")
else:
    print("No model directories found!")
