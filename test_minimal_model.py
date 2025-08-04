#!/usr/bin/env python3
"""
Test script for the minimal trained model.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import generate
from constants import MODELS_PATH

def test_minimal_model():
    """Test the trained minimal model on simple addition tasks."""
    
    # Load the trained model
    model_path = f"{MODELS_PATH}/minimal-experiment"
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Could not load model from {model_path}: {e}")
        print("Make sure you've run the minimal experiment first!")
        return
    
    # Test prompts
    test_prompts = [
        "1 + 2 = ->",
        "3 + 4 = ->", 
        "5 + 6 = ->",
        "7 + 8 = ->",
        "9 + 10 = ->",  # New example not in training
    ]
    
    print("\nTesting model on addition tasks:")
    print("=" * 50)
    
    for prompt in test_prompts:
        # Generate response
        generated_texts = generate(
            model=model,
            tokenizer=tokenizer,
            texts=[prompt],
            max_new_tokens=10,
            temperature=0.1,  # Low temperature for deterministic output
            top_k=1,
            num_return_sequences=1
        )
        
        response = generated_texts[0].replace(prompt, "").strip()
        expected = eval(prompt.replace(" = ->", ""))  # Simple eval for expected result
        
        print(f"Prompt: {prompt}")
        print(f"Expected: {expected}")
        print(f"Generated: {response}")
        print(f"Correct: {str(expected) in response}")
        print("-" * 30)

if __name__ == "__main__":
    test_minimal_model() 