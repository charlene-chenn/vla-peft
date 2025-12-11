import os
import sys

# CRITICAL: Set ultralytics config directory BEFORE any imports
CUSTOM_ULTRA_DIR = "/cs/student/projects3/2023/dkozlov/.ultralytics"
os.makedirs(CUSTOM_ULTRA_DIR, exist_ok=True)
os.environ["ULTRALYTICS_CONFIG_DIR"] = CUSTOM_ULTRA_DIR

# Now safe to import everything else
import gc
import json
import random
from pathlib import Path

from unsloth import FastLanguageModel
import torch
import torch.nn.functional as F
from PIL import Image

from datasets import load_dataset
from transformers import (
    TrainingArguments,
    Trainer,
    CLIPModel,
    CLIPProcessor,
)

from yolo_detector import YOLOv8Detector  # type: ignore
THIS_DIR = Path(__file__).resolve().parent

# -------------------------------------------------------------------------
# 1. BASIC SETUP: GPU, CLEAR CACHE
# -------------------------------------------------------------------------

torch.cuda.empty_cache()
gc.collect()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"GPU memory reserved:  {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

# -------------------------------------------------------------------------
# 2. LOAD LLAMA-3 WITH UNSLOTH + LORA
# -------------------------------------------------------------------------

MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    device_map={"": 0},
)

print(
    f"Model loaded! GPU memory now: "
    f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB"
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "v_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=3407,
)

# -------------------------------------------------------------------------
# 3. VISION MODELS: YOLO + CLIP
# -------------------------------------------------------------------------

# (1) YOLO
YOLO_WEIGHTS = THIS_DIR / "checkpoints" / "yolo_best.pt"
yolo_detector = YOLOv8Detector(model_path=str(YOLO_WEIGHTS))

# (2) CLIP
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model.eval()

# -------------------------------------------------------------------------
# 4. DATA SOURCE: IMAGES IN ./geohints_processed
# -------------------------------------------------------------------------

IMAGES_ROOT = THIS_DIR / "geohints_processed"


def iter_images(root_dir: Path):
    """
    Yield (image_path, country_code) from filenames like `fr_1234.jpg`.
    """
    if not root_dir.exists():
        raise RuntimeError(f"Image dir does not exist: {root_dir}")

    for img_path in sorted(root_dir.glob("*.jpg")):
        stem = img_path.stem  # e.g. "fr_1234"
        country = stem.split("_")[0].lower()
        yield img_path, country


# pre-scan to know which countries exist in this dataset
image_info = list(iter_images(Path(IMAGES_ROOT)))
COUNTRY_LABELS = sorted({country for _, country in image_info})
print("Discovered countries in dataset:", COUNTRY_LABELS)

# Some generic vibes/contents; feel free to tweak/expand these
VIBE_LABELS = [
    "mountain architecture / architecture",
    "dense urban city center",
    "suburban residential area",
    "coastal beach town",
    "rural farmland",
    "forest road",
    "desert landscape",
]

CONTENT_LABELS = [
    "architecture",
    "sceneries",
    "road",
    "cars",
    "vegetation",
    "shops",
    "apartment buildings",
]


# -------------------------------------------------------------------------
# 5. CLIP HELPERS
# -------------------------------------------------------------------------

def clip_probs_for_texts(image: Image.Image, texts: list[str]) -> list[float]:
    inputs = clip_processor(
        text=texts,
        images=image,
        return_tensors="pt",
        padding=True,
    ).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits = outputs.logits_per_image[0]  # [len(texts)]
        probs = F.softmax(logits, dim=-1).cpu().tolist()
    return probs


def predict_country_with_clip(image: Image.Image):
    prompts = [f"a street scene in {c}" for c in COUNTRY_LABELS]
    probs = clip_probs_for_texts(image, prompts)
    sorted_idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)

    top_idx = sorted_idx[0]
    top_country = COUNTRY_LABELS[top_idx]
    top_conf = probs[top_idx]

    top3 = [COUNTRY_LABELS[i] for i in sorted_idx[:3]]
    top3_probs = [probs[i] for i in sorted_idx[:3]]

    return {
        "country": top_country,
        "country_confidence": float(top_conf),
        "top_countries": top3,
        "top_countries_probs": top3_probs,
    }


def predict_vibe_with_clip(image: Image.Image):
    prompts = [f"a {v}" for v in VIBE_LABELS]
    probs = clip_probs_for_texts(image, prompts)
    top_idx = max(range(len(probs)), key=lambda i: probs[i])
    vibe_top = VIBE_LABELS[top_idx]
    vibe_distribution = {
        VIBE_LABELS[i]: float(probs[i]) for i in range(len(VIBE_LABELS))
    }
    return {
        "vibe_top": vibe_top,
        "vibe_distribution": vibe_distribution,
    }


def predict_contents_with_clip(image: Image.Image, top_k: int = 2):
    prompts = [f"a photo mainly of {c}" for c in CONTENT_LABELS]
    probs = clip_probs_for_texts(image, prompts)
    sorted_idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    return [CONTENT_LABELS[i] for i in sorted_idx[:top_k]]


# -------------------------------------------------------------------------
# 6. YOLO HELPERS (CLASS NAMES + BBOXES)
# -------------------------------------------------------------------------

def get_det_class_name(det: dict) -> str | None:
    """Try different keys to get the class name from YOLO detection dict."""
    for key in ("class_name", "cls_name", "name", "label"):
        if key in det and det[key]:
            return str(det[key])
    return None


def get_det_bbox(det: dict):
    """
    Try to extract [x1, y1, x2, y2] bounding box.
    Adjust this to match YOLOv8Detector.predict output if needed.
    """
    if "bbox" in det and len(det["bbox"]) == 4:
        return det["bbox"]

    if "box" in det and len(det["box"]) == 4:
        return det["box"]

    # Sometimes there might be keys x1,y1,x2,y2
    keys = ("x1", "y1", "x2", "y2")
    if all(k in det for k in keys):
        return [det["x1"], det["y1"], det["x2"], det["y2"]]

    return None


def crop_image(image: Image.Image, bbox):
    x1, y1, x2, y2 = bbox
    return image.crop((x1, y1, x2, y2))


def predict_sign_countries_with_clip(
    image: Image.Image,
    detections: list[dict],
    max_signs: int = 4,
):
    sign_crops = []
    for det in detections:
        cls_name = get_det_class_name(det)
        if cls_name is None:
            continue

        # Adapt to your sign classes (e.g. "sign", "road_sign", etc.)
        if cls_name.lower() in ("sign", "road sign", "street sign", "roadsign"):
            bbox = get_det_bbox(det)
            if bbox is None:
                continue
            sign_crops.append(crop_image(image, bbox))
            if len(sign_crops) >= max_signs:
                break

    top_sign_countries = []
    for crop in sign_crops:
        c_info = predict_country_with_clip(crop)
        top_sign_countries.append(c_info["country"])

    return top_sign_countries


def predict_top_contents(image: Image.Image, detections: list[dict], top_k: int = 2):
    # YOLO classes
    yolo_classes = []
    for det in detections:
        cls_name = get_det_class_name(det)
        if cls_name:
            yolo_classes.append(cls_name)

    # CLIP contents
    clip_top = predict_contents_with_clip(image, top_k=top_k)

    # combine: CLIP first, then YOLO classes
    combined = list(dict.fromkeys(clip_top + yolo_classes))
    return combined[:top_k]


def predict_driving_side_stub(image: Image.Image):
    """
    Placeholder for a proper driving-side classifier.
    You can replace this later.
    """
    return {
        "driving_side": "unknown",
        "driving_side_confidence": 0.5,
    }


# -------------------------------------------------------------------------
# 7. FULL VISION PIPELINE → TARGET JSON STRUCTURE
# -------------------------------------------------------------------------

def run_vision_pipeline(image_path: str, gt_country: str | None = None) -> dict:
    image = Image.open(image_path).convert("RGB")

    # YOLO detections
    detections = yolo_detector.predict(image, conf=0.5)

    # CLIP-based predictions
    country_info = predict_country_with_clip(image)
    vibe_info = predict_vibe_with_clip(image)
    sign_countries = predict_sign_countries_with_clip(image, detections)
    top_contents = predict_top_contents(image, detections)
    driving_info = predict_driving_side_stub(image)

    result = {
        "image_path": str(image_path),
        "country": country_info["country"],
        "country_confidence": country_info["country_confidence"],
        "driving_side": driving_info["driving_side"],
        "driving_side_confidence": driving_info["driving_side_confidence"],
        "vibe_top": vibe_info["vibe_top"],
        "vibe_distribution": vibe_info["vibe_distribution"],
        "evidence": {
            "top_sign_countries": sign_countries,
            "top_contents": top_contents,
            "gradcam_examples": [],  # fill later if you add grad-cam
        },
        # optional debug info
        "debug": {
            "clip_top_countries": country_info["top_countries"],
            "clip_top_countries_probs": country_info["top_countries_probs"],
            "yolo_raw_detections": detections,
            "gt_country": gt_country,
        },
    }

    return result


# -------------------------------------------------------------------------
# 8. PROMPT + TARGET EXPLANATION FOR SFT
# -------------------------------------------------------------------------

def make_prompt(vision_json: dict) -> str:
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


def make_target_explanation(vision_json: dict, gt_country: str) -> str:
    driving_side = vision_json.get("driving_side", "unknown")
    vibe_top = vision_json.get("vibe_top", "unknown vibe")
    evidence = vision_json.get("evidence", {})
    sign_countries = evidence.get("top_sign_countries", [])
    top_contents = evidence.get("top_contents", [])

    alt_countries = [c for c in sign_countries if c != gt_country]
    alt_countries = list(dict.fromkeys(alt_countries))[:3]
    alt_str = ", ".join(alt_countries) if alt_countries else "no obvious alternatives"

    explanation = (
        f"My best guess is {gt_country}. "
        f"The detected driving side appears to be {driving_side}, and the scene has a {vibe_top} feel. "
        f"The sign predictions include {sign_countries}, and the main contents seem to be {top_contents}, "
        f"which fit well with {gt_country}. "
        f"Other possible countries, based on similar signs or scenery, could be {alt_str}, "
        f"but overall {gt_country} is the most likely location."
    )
    return explanation


# -------------------------------------------------------------------------
# 9. BUILD SFT DATASET (JSONL)
# -------------------------------------------------------------------------

def build_samples(limit: int | None = None):
    samples = []
    for i, (img_path, country) in enumerate(image_info):
        if limit is not None and i >= limit:
            break
        vjson = run_vision_pipeline(str(img_path), gt_country=country)
        samples.append(
            {
                "vision_json": vjson,
                "gt_country": country,
            }
        )
    return samples


def write_sft_jsonl(samples, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in samples:
            prompt = make_prompt(ex["vision_json"])
            answer = make_target_explanation(ex["vision_json"], ex["gt_country"])
            row = {"text": prompt + answer}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def train_val_split(samples, val_ratio=0.1, seed=42):
    random.seed(seed)
    shuffled = samples[:]
    random.shuffle(shuffled)
    n = len(shuffled)
    n_val = int(n * val_ratio)
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    return train, val


# -------------------------------------------------------------------------
# 10. TOKENIZATION & FINE-TUNING WITH UNSLOTH
# -------------------------------------------------------------------------

def tokenize_with_labels(batch):
    out = tokenizer(
        batch["text"],
        truncation=True,
        max_length=2048,
        padding="max_length",
    )
    out["labels"] = out["input_ids"].copy()
    return out


def main():
    # 1) Build samples (you can limit here while debugging)
    print("Building samples from images ...")
    samples = build_samples(limit=None)  # or limit=2000

    print(f"Total samples: {len(samples)}")
    train_samples, val_samples = train_val_split(samples, val_ratio=0.1)

    data_dir = THIS_DIR / "data"
    train_jsonl = data_dir / "geohints_train.jsonl"
    val_jsonl = data_dir / "geohints_val.jsonl"

    write_sft_jsonl(train_samples, train_jsonl)
    write_sft_jsonl(val_samples, val_jsonl)
    print("Wrote:", train_jsonl, val_jsonl)

    # 2) Load with HF datasets
    ds = load_dataset(
        "json",
        data_files={
            "train": str(train_jsonl),
            "validation": str(val_jsonl),
        },
    )

    tokenized = ds.map(
        tokenize_with_labels,
        batched=True,
        remove_columns=ds["train"].column_names,
    )

    # 3) TrainingArguments + Trainer
    output_dir = THIS_DIR / "geohints-llama3-explainer"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=20,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
    )

    trainer.train()
    print("Training complete. LoRA weights saved to:", output_dir)


if __name__ == "__main__":
    main()
