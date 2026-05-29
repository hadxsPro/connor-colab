#!/usr/bin/env python3
"""CONNOR FT GOLD V6 - Colab T4 16GB
Runtime -> Change runtime type -> T4 GPU
Pulls gold_v6 data from HF, filters >=4.2, fine-tunes Qwen2.5-7B with Unsloth.
"""

# Cell unique pour Colab
import os, json, time, torch, random, gc, sys, io, zipfile, requests
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
HF_DATASET = "hadxs/ultimate-multilang-v3"
MIN_Q = 4.2
MAX_PAIRS = 200000
MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
SAVE_NAME = "connor-gold-v6-7b"
MAX_SEQ = 2048
EPOCHS = 2
LORA_R = 32
LORA_ALPHA = 64
LR = 3e-4
BATCH_SIZE = 4
GRAD_ACC = 4
WARMUP = 0.05
HF_TOKEN = os.getenv("HF_TOKEN", "")

print(f"=== CONNOR FT GOLD V6 (T4) ===")
print(f"Source: {HF_DATASET}/teacher_gold/gold_v6.jsonl")
print(f"Min quality: {MIN_Q} | Max pairs: {MAX_PAIRS}")
print(f"Model: {MODEL} | Seq: {MAX_SEQ} | Epochs: {EPOCHS}")
print(f"LoRA: r={LORA_R} alpha={LORA_ALPHA} | LR: {LR}")

# ============================================================
# 1. DOWNLOAD GOLD V6 DATA FROM HF
# ============================================================
print("\n--- 1. Downloading gold_v6 data from HF ---")

def download_jsonl(url):
    r = requests.get(url, timeout=300, stream=True)
    r.raise_for_status()
    lines = []
    for line in r.iter_lines(decode_unicode=True):
        if line:
            lines.append(line)
    return [json.loads(l) for l in lines]

# Download cumulative
cum_url = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/teacher_gold/gold_v6.jsonl"
try:
    cum_pairs = download_jsonl(cum_url)
    print(f"Cumulative: {len(cum_pairs)} pairs")
except Exception as e:
    print(f"Cumulative download failed: {e}")
    cum_pairs = []

# Download latest deltas (max 10)
delta_base = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/teacher_gold/deltas"
delta_pairs = []
try:
    r = requests.get(f"https://huggingface.co/api/datasets/{HF_DATASET}")
    siblings = r.json().get("siblings", [])
    deltas = sorted(
        [s["rfilename"] for s in siblings if "teacher_gold/deltas/gold_v6_" in s["rfilename"]],
        reverse=True
    )[:10]
    for d in deltas:
        try:
            pairs = download_jsonl(f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/{d}")
            delta_pairs.extend(pairs)
            print(f"  Delta {d.split('/')[-1]}: {len(pairs)} pairs")
        except:
            pass
except Exception as e:
    print(f"Delta fetch failed: {e}")

all_pairs = cum_pairs + delta_pairs
print(f"Total downloaded: {len(all_pairs)} pairs")

# Dedup by prompt+response hash
seen = set()
deduped = []
for d in all_pairs:
    h = hash((d.get("prompt",""), d.get("response","")))
    if h not in seen:
        seen.add(h)
        deduped.append(d)
print(f"After dedup: {len(deduped)}")

# ============================================================
# 2. FILTER BY QUALITY >= 4.2
# ============================================================
print("\n--- 2. Filtering by quality ---")

filtered = [d for d in deduped if d.get("quality", 0) >= MIN_Q]
print(f"Quality >= {MIN_Q}: {len(filtered)}")

if len(filtered) == 0:
    raise SystemExit("No pairs passed quality filter!")

# Balance languages: cap per language to ensure diversity
lang_counts = {}
for d in filtered:
    l = d.get("lang", "unknown")
    lang_counts[l] = lang_counts.get(l, 0) + 1
print(f"Languages: {lang_counts}")

# Cap at MAX_PAIRS
if len(filtered) > MAX_PAIRS:
    random.seed(42)
    filtered = random.sample(filtered, MAX_PAIRS)
print(f"Final dataset: {len(filtered)} pairs")

# ============================================================
# 3. FORMAT FOR QWEN2.5 CHAT
# ============================================================
print("\n--- 3. Formatting dataset ---")

def format_pair(prompt, response):
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"

formatted = []
for d in filtered:
    p = d.get("prompt", "").strip()
    r = d.get("response", "").strip()
    if len(p) < 5 or len(r) < 20:
        continue
    formatted.append({"text": format_pair(p, r)})

# Add 10% with system prompt for context awareness
SYSTEM_PROMPT = "Tu es Connor, assistant IA personnel. Tu reponds en francais de maniere naturelle, precise et engageante."
extra = []
for d in filtered[:len(filtered)//10]:
    p = d.get("prompt", "").strip()
    r = d.get("response", "").strip()
    if len(p) < 5 or len(r) < 20:
        continue
    text = (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{p}<|im_end|>\n"
            f"<|im_start|>assistant\n{r}<|im_end|>")
    extra.append({"text": text})
formatted.extend(extra)

random.shuffle(formatted)
print(f"Total formatted: {len(formatted)}")

# ============================================================
# 4. LOAD MODEL (Unsloth 4-bit QLoRA)
# ============================================================
print("\n--- 4. Loading model ---")
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL,
    max_seq_length=MAX_SEQ,
    dtype=torch.float16,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

trainable = model.num_parameters(only_trainable=True)
total = model.num_parameters()
print(f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.1f}%)")

# ============================================================
# 5. CREATE DATASET
# ============================================================
print("\n--- 5. Creating dataset ---")
from datasets import Dataset

ds = Dataset.from_list(formatted)
split = ds.train_test_split(test_size=min(500, len(ds)//20), seed=42)
train_ds, eval_ds = split["train"], split["test"]
print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

# ============================================================
# 6. TRAINING
# ============================================================
print("\n--- 6. Training ---")
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ,
    args=TrainingArguments(
        output_dir=f"/content/{SAVE_NAME}",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC,
        warmup_ratio=WARMUP,
        learning_rate=LR,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        eval_steps=len(train_ds)//20 if len(train_ds) > 100 else 50,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        max_grad_norm=0.3,
        weight_decay=0.01,
        seed=42,
    ),
)

t0 = time.time()
trainer.train()
train_time = (time.time() - t0) / 60
print(f"Training done in {train_time:.1f} min")

# ============================================================
# 7. SAVE ADAPTER + PUSH TO HF
# ============================================================
print("\n--- 7. Saving & uploading ---")
model.save_pretrained(f"/content/{SAVE_NAME}-final")
tokenizer.save_pretrained(f"/content/{SAVE_NAME}-final")
print(f"Saved to /content/{SAVE_NAME}-final")

# Upload to HF if token available
if HF_TOKEN:
    print("Uploading LoRA adapter to HF...")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(f"hadxs/{SAVE_NAME}", repo_type="model", exist_ok=True, token=HF_TOKEN)
        api.upload_folder(
            folder_path=f"/content/{SAVE_NAME}-final",
            repo_id=f"hadxs/{SAVE_NAME}",
            token=HF_TOKEN,
        )
        print(f"Uploaded to hadxs/{SAVE_NAME}")
    except Exception as e:
        print(f"HF upload failed: {e}")

# ============================================================
# 8. EVALUATION
# ============================================================
print("\n--- 8. Testing ---")
test_prompts = [
    "Salut Connor, comment ca va ?",
    "Explique le machine learning en 3 phrases.",
    "Ecris une fonction Python Fibonacci.",
    "Que penses-tu de l'intelligence artificielle ?",
    "Raconte une blague courte.",
    "C'est quoi la difference entre merge sort et quick sort ?",
]

model.eval()
for p in test_prompts:
    text = f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, temperature=0.6, top_p=0.9)
    resp = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"  Q: {p}")
    print(f"  A: {resp.strip()[:200]}")
    print()

# ============================================================
# 9. SAVE TO GOOGLE DRIVE (optional)
# ============================================================
print("\n--- 9. Drive backup ---")
try:
    from google.colab import drive
    drive.mount("/content/drive")
    import shutil
    backup = f"/content/drive/MyDrive/{SAVE_NAME}"
    if os.path.exists(backup):
        shutil.rmtree(backup)
    shutil.copytree(f"/content/{SAVE_NAME}-final", backup)
    print(f"Backup saved to Drive: {SAVE_NAME}")
except Exception as e:
    print(f"Drive backup skipped: {e}")

print(f"\n=== DONE ===")
print(f"Pairs: {len(formatted)} | Train: {len(train_ds)}")
print(f"Time: {train_time:.1f} min | Epochs: {EPOCHS}")
print(f"Model: hadxs/{SAVE_NAME}")
