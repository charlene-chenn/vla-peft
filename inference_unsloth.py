import json
from pathlib import Path
import torch
from unsloth import FastLanguageModel

# -------------------------------------------------------------------------
# 1. LOAD BASE MODEL + LORA CHECKPOINT
# -------------------------------------------------------------------------

BASE_MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"
LORA_CHECKPOINT_DIR = "./checkpoints/checkpoint-1000"  # Try a different checkpoint

print("Loading base model with Unsloth and LoRA weights...")

# Load the base model with the trained LoRA checkpoint
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=LORA_CHECKPOINT_DIR,  # Load checkpoint directly
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

model.eval()

# Enable native 2x faster inference
FastLanguageModel.for_inference(model)

print("Model loaded with LoRA weights from:", LORA_CHECKPOINT_DIR)

# -------------------------------------------------------------------------
# 2. LOAD DUMMY VISION DATA (replace with actual vision pipeline later)
# -------------------------------------------------------------------------

# Load the vision data
vision_json_path = Path("./dummy.json")
with open(vision_json_path, 'r') as f:
    vision_data = json.load(f)
 
print("Loaded vision data:", vision_data["country"])

# -------------------------------------------------------------------------
# 3. CREATE PROMPT AND RUN INFERENCE
# -------------------------------------------------------------------------

def make_prompt(vision_json: dict) -> str:
    """Create the prompt for the model based on vision data."""
    vision_str = json.dumps(vision_json, ensure_ascii=False, indent=2)
    prompt = (
        "You are an expert GeoGuessr player.\n"
        "You are given structured evidence extracted from a street-view image.\n"
        "Reason step by step about where this location could be, and explain why.\n\n"
        "EVIDENCE (JSON):\n"
        f"{vision_str}\n\n"
        "TASK:\n"
        "Write a short paragraph explaining:\n"
        "- Which country you think this is (best guess),\n"
        "- Any plausible alternative countries,\n"
        "- Why the driving side, architecture, vegetation, signs, and landmarks support your guess.\n"
        "Do NOT output JSON; answer in natural language.\n\n"
        "ANSWER:\n"
    )
    return prompt

# Create the prompt
prompt = make_prompt(vision_data)
print("\n" + "="*80)
print("PROMPT:")
print("="*80)
print(prompt)

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# Generate response
print("\n" + "="*80)
print("GENERATING RESPONSE...")
print("="*80)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

# Decode the output
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract just the answer (remove the prompt)
answer = response[len(prompt):].strip()

print("\n" + "="*80)
print("MODEL OUTPUT:")
print("="*80)
print(answer)
print("\n" + "="*80)
