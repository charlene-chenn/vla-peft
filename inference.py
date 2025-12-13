import json
from pathlib import Path
import torch
from unsloth import FastLanguageModel

# -------------------------------------------------------------------------
# 1. LOAD BASE MODEL + LORA CHECKPOINT
# -------------------------------------------------------------------------

BASE_MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"
LORA_CHECKPOINT_DIR = "./checkpoints"

# Load the base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL_NAME,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Load the LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "v_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=False,  # Set to False for inference
)

# Load the trained LoRA checkpoint
from peft import PeftModel
model = PeftModel.from_pretrained(model, LORA_CHECKPOINT_DIR)
model.eval()

print("Model loaded with LoRA weights from:", LORA_CHECKPOINT_DIR)

# -------------------------------------------------------------------------
# 2. LOAD DUMMY VISION DATA (replace with actual vision pipeline later)
# -------------------------------------------------------------------------

# TODO: replace this section with pipeline from vision model
dir = Path("./dummy.json")
with open(dir, 'r') as f:
    data = json.load(f) # load as dict
 
print("Data:", data, "Type:", type(data))

