#!/usr/bin/env python3
"""
Minimal viable experiment for password-locked LLM finetuning.
Uses the smallest model (Pythia-160m) with minimal addition dataset.
"""

import os
import json
from finetune import finetune
from constants import DATA_PATH, MODELS_PATH

def create_minimal_dataset():
    """Create a minimal dataset with just 4 examples for quick testing."""
    minimal_data = [
        {
            "prompt": "1 + 2 = ->",
            "answer": "3",
            "nb_digits": 2
        },
        {
            "prompt": "3 + 4 = ->", 
            "answer": "7",
            "nb_digits": 2
        },
        {
            "prompt": "5 + 6 = ->",
            "answer": "11", 
            "nb_digits": 2
        },
        {
            "prompt": "7 + 8 = ->",
            "answer": "15",
            "nb_digits": 2
        }
    ]
    
    # Create data directory if it doesn't exist
    os.makedirs(DATA_PATH, exist_ok=True)
    
    # Save minimal dataset
    json.dump(minimal_data, open(f"{DATA_PATH}/minimal_train.json", "w"))
    json.dump(minimal_data, open(f"{DATA_PATH}/minimal_val.json", "w"))
    
    print(f"Created minimal dataset with {len(minimal_data)} examples")
    return minimal_data

def run_minimal_experiment():
    """Run the minimal viable experiment."""
    
    # Create minimal dataset
    create_minimal_dataset()
    
    # Smallest model from the paper
    model_name = "EleutherAI/pythia-160m"
    
    # Minimal training parameters
    training_params = {
        "lr": 2e-5,
        "batch_size": 2,  # Very small batch size
        "val_batch_size": 4,
        "epochs": 0.1,  # Just 10% of an epoch for quick testing
        "max_length": 128,  # Shorter sequences
        "nb_eval_lp_batches": 1,  # Minimal evaluation
        "nb_eval_generations_batches": 1,
        "weight_decay": 0.01,
        "seq_decay": 1.0,
        "disable_wandb": True,  # Disable wandb for minimal experiment
    }
    
    print("Starting minimal experiment...")
    print(f"Model: {model_name}")
    print(f"Training params: {training_params}")
    
    # Run finetuning
    finetune(
        model_name=model_name,
        dataset_name="minimal_train",
        val_dataset_names=("minimal_val",),
        run_name="minimal-experiment",
        project_name="password-locked-minimal",
        metadata={
            "experiment_type": "minimal_viable",
            "task": "addition",
            "model_size": "160m",
            "dataset_size": 4
        },
        **training_params
    )
    
    print("Minimal experiment completed!")

if __name__ == "__main__":
    run_minimal_experiment() 