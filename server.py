import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
LORA_CHECKPOINT_DIR = "./checkpoints/checkpoint-1494"

app = FastAPI()
tokenizer = None
model = None

class ChatMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str

class ChatReq(BaseModel):
    messages: list[ChatMessage]
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

@app.on_event("startup")
def _load():
    global tokenizer, model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    # Skip corrupted LoRA checkpoint
    try:
        if LORA_CHECKPOINT_DIR and os.path.exists(LORA_CHECKPOINT_DIR):
            print(f"Attempting to load LoRA from {LORA_CHECKPOINT_DIR}...")
            base = PeftModel.from_pretrained(base, LORA_CHECKPOINT_DIR)
            print("LoRA loaded successfully")
    except Exception as e:
        print(f"WARNING: Could not load LoRA checkpoint: {e}")
        print("Continuing with base model only")

    model = base.eval()
    print("Model loaded and ready!")

@app.get("/health")
def health():
    return {"ok": True, "device": str(model.device), "model": BASE_MODEL_NAME}

@app.post("/chat")
@torch.inference_mode()
def chat(req: ChatReq):
    # Convert Pydantic objects into plain dicts
    msgs = [{"role": m.role, "content": m.content} for m in req.messages]

    input_ids = tokenizer.apply_chat_template(
        msgs,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=req.max_new_tokens,
        do_sample=req.temperature > 0,
        temperature=req.temperature,
        top_p=req.top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Important: decode ONLY the newly generated tokens
    new_tokens = out[0, input_ids.shape[-1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return {"assistant": text}
