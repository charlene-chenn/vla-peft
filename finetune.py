import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, BitsAndBytesConfig
import peft

import os
from huggingface_hub import login
import gc

# Set HuggingFace cache to avoid disk quota issues
os.environ['HF_HOME'] = '/cs/student/projects3/2023/dkozlov/conda-pkgs/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/cs/student/projects3/2023/dkozlov/conda-pkgs/huggingface/transformers'

# Access Llama model
token = # insert token #
login(token, add_to_git_credential=False)  # Don't save token to disk

# Clear GPU memory and restart Python kernel state
torch.cuda.empty_cache()
gc.collect()

# Check available memory
print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

# Load model with unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    device_map={"": 0},  # Force everything to GPU 0
)

print(f"Model loaded! GPU memory now: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")