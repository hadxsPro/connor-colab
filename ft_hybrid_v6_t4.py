#!/usr/bin/env python3
"""CONNOR FT HYBRID V6 - Colab T4 16GB
Runtime -> Change runtime type -> T4 GPU
Mix: gold_v6 (structure + qualite) + ft_conversational (naturelle, 50k)
"""

import os, json, time, torch, random, gc, sys, io, zipfile, requests
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
HF_DATASET = "hadxs/ultimate-multilang-v3"
MIN_Q = 4.2
MAX_GOLD = 120000
MAX_CONV = 45000
MAX_TOTAL = 150000
MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
SAVE_NAME = "connor-hybrid-v6-7b"
MAX_SEQ = 2048
EPOCHS = 2
LORA_R = 32
LORA_ALPHA = 64
LR = 3e-4
BATCH_SIZE = 4
GRAD_ACC = 4
WARMUP = 0.05
HF_TOKEN = os.getenv("HF_TOKEN", "")

print(f"=== CONNOR FT HYBRID V6 (T4) ===")
print(f"Gold: {HF_DATASET}/teacher_gold/gold_v6.jsonl (q>={MIN_Q}, max {MAX_GOLD})")
print(f"Conv: {HF_DATASET}/teacher_gold/ft_conversational_50k.jsonl (max {MAX_CONV})")
print(f"Model: {MODEL} | Seq: {MAX_SEQ} | Epochs: {EPOCHS}")
print(f"LoRA: r={LORA_R} alpha={LORA_ALPHA} | LR: {LR}")

# ============================================================
# 1. DOWNLOAD GOLD V6
# ============================================================
print("\n--- 1. Gold v6 from HF ---")

def download_jsonl(url):
    r = requests.get(url, timeout=300, stream=True)
    r.raise_for_status()
    return [json.loads(l) for l in r.iter_lines(decode_unicode=True) if l]

cum_url = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/teacher_gold/gold_v6.jsonl"
try:
    gold = download_jsonl(cum_url)
    print(f"Cumulative: {len(gold)}")
except Exception as e:
    print(f"Cumulative failed: {e}"); gold = []

# Latest deltas
try:
    r = requests.get(f"https://huggingface.co/api/datasets/{HF_DATASET}", timeout=30)
    deltas = sorted([s["rfilename"] for s in r.json().get("siblings",[])
                     if "teacher_gold/deltas/gold_v6_" in s["rfilename"]], reverse=True)[:5]
    for d in deltas:
        try:
            gold.extend(download_jsonl(f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/{d}"))
        except: pass
    print(f"+ deltas: {len(gold)} total")
except Exception as e:
    print(f"Deltas failed: {e}")

# Dedup
seen = set()
deduped = []
for d in gold:
    h = hash((d.get("prompt",""), d.get("response","")))
    if h not in seen:
        seen.add(h); deduped.append(d)
print(f"After dedup: {len(deduped)}")

# Filter quality
gold_filt = [d for d in deduped if d.get("quality",0) >= MIN_Q]
print(f"Quality >= {MIN_Q}: {len(gold_filt)}")

# Cap
if len(gold_filt) > MAX_GOLD:
    random.seed(42); gold_filt = random.sample(gold_filt, MAX_GOLD)
print(f"Gold final: {len(gold_filt)}")

# ============================================================
# 2. DOWNLOAD CONVERSATIONAL DATA (HF)
# ============================================================
print("\n--- 2. Conversational data from HF ---")

conv_url = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/teacher_gold/ft_conversational_50k.jsonl"
conv_pairs = []
try:
    conv_pairs = download_jsonl(conv_url)
    print(f"Conversational: {len(conv_pairs)}")
except Exception as e:
    print(f"Conversational download failed: {e}")

# Filter: dedup, min lengths
conv_filt = []
seen_c = set()
for d in conv_pairs:
    inst = (d.get("instruction") or d.get("prompt") or "").strip()
    out = (d.get("output") or d.get("response") or "").strip()
    if len(inst) < 5 or len(out) < 20: continue
    h = hash(inst[:50] + out[:50])
    if h in seen_c: continue
    seen_c.add(h)
    conv_filt.append({"prompt": inst, "response": out, "source": "ft_conversational"})
print(f"After filter: {len(conv_filt)}")

if len(conv_filt) > MAX_CONV:
    random.seed(123); conv_filt = random.sample(conv_filt, MAX_CONV)
print(f"Conversational final: {len(conv_filt)}")

# ============================================================
# 3. MERGE + BALANCE
# ============================================================
print("\n--- 3. Merging datasets ---")

# Mix: ~65% gold_v6, ~35% conversational
all_pairs = gold_filt + conv_filt
print(f"Total before cap: {len(all_pairs)} (gold={len(gold_filt)}, conv={len(conv_filt)})")

if len(all_pairs) > MAX_TOTAL:
    n_gold = int(MAX_TOTAL * 0.65)
    n_conv = MAX_TOTAL - n_gold
    random.seed(42)
    g_sample = random.sample(gold_filt, min(n_gold, len(gold_filt)))
    c_sample = random.sample(conv_filt, min(n_conv, len(conv_filt)))
    all_pairs = g_sample + c_sample
    random.shuffle(all_pairs)
    print(f"Capped to {MAX_TOTAL} (gold={len(g_sample)}, conv={len(c_sample)})")

random.shuffle(all_pairs)
print(f"Final: {len(all_pairs)} pairs")

# ============================================================
# 4. FORMAT QWEN2.5
# ============================================================
print("\n--- 4. Formatting ---")

def fmt(inst, out):
    return f"<|im_start|>user\n{inst}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>"

SYSTEM = "Tu es Connor, assistant IA personnel. Tu reponds en francais de maniere naturelle, precise et engageante."

formatted = []
for d in all_pairs:
    p = d.get("prompt","").strip()
    r = d.get("response","").strip()
    if len(p) < 5 or len(r) < 20: continue
    formatted.append({"text": fmt(p, r)})

# 10% with system prompt
extra = []
for d in all_pairs[:len(all_pairs)//10]:
    p = d.get("prompt","").strip()
    r = d.get("response","").strip()
    if len(p) < 5 or len(r) < 20: continue
    extra.append({"text": f"<|im_start|>system\n{SYSTEM}<|im_end|>\n{fmt(p, r)}"})
formatted.extend(extra)

random.shuffle(formatted)
print(f"Formatted: {len(formatted)}")

# ============================================================
# 5. LOAD MODEL
# ============================================================
print("\n--- 5. Loading model ---")
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL, max_seq_length=MAX_SEQ, dtype=torch.float16, load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=LORA_R,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=LORA_ALPHA, lora_dropout=0.05, bias="none",
    use_gradient_checkpointing="unsloth", random_state=42,
)
t = model.num_parameters(only_trainable=True); tt = model.num_parameters()
print(f"Trainable: {t/1e6:.1f}M / {tt/1e6:.1f}M ({100*t/tt:.1f}%)")

# ============================================================
# 6. DATASET
# ============================================================
print("\n--- 6. Dataset ---")
from datasets import Dataset

ds = Dataset.from_list(formatted)
split = ds.train_test_split(test_size=min(500, len(ds)//20), seed=42)
train_ds, eval_ds = split["train"], split["test"]
print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

# ============================================================
# 7. TRAINING
# ============================================================
print("\n--- 7. Training ---")
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=train_ds, eval_dataset=eval_ds,
    dataset_text_field="text", max_seq_length=MAX_SEQ,
    args=TrainingArguments(
        output_dir=f"/content/{SAVE_NAME}",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC,
        warmup_ratio=WARMUP, learning_rate=LR,
        fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
        logging_steps=10,
        eval_steps=len(train_ds)//20 if len(train_ds) > 100 else 50,
        save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        report_to="none", lr_scheduler_type="cosine", optim="adamw_8bit",
        max_grad_norm=0.3, weight_decay=0.01, seed=42,
    ),
)
t0 = time.time()
trainer.train()
train_time = (time.time() - t0) / 60
print(f"Training: {train_time:.1f} min")

# ============================================================
# 8. SAVE + UPLOAD
# ============================================================
print("\n--- 8. Saving ---")
model.save_pretrained(f"/content/{SAVE_NAME}-final")
tokenizer.save_pretrained(f"/content/{SAVE_NAME}-final")
print(f"Saved to /content/{SAVE_NAME}-final")

if HF_TOKEN:
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(f"hadxs/{SAVE_NAME}", repo_type="model", exist_ok=True, token=HF_TOKEN)
        api.upload_folder(folder_path=f"/content/{SAVE_NAME}-final",
                         repo_id=f"hadxs/{SAVE_NAME}", token=HF_TOKEN)
        print(f"Uploaded to hadxs/{SAVE_NAME}")
    except Exception as e:
        print(f"HF upload failed: {e}")

# ============================================================
# 9. EVAL
# ============================================================
print("\n--- 9. Testing ---")
test_prompts = [
    "Salut Connor, comment ca va ?",
    "Explique le machine learning en 3 phrases.",
    "Ecris une fonction Python Fibonacci.",
    "Que penses-tu de l'intelligence artificielle ?",
    "Raconte une blague courte.",
    "C'est quoi la difference entre merge sort et quick sort ?",
    "What is the meaning of life?",
    "Hoe los ik een conflict op in een team?",
]
model.eval()
for p in test_prompts:
    text = f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, temperature=0.6, top_p=0.9)
    resp = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Q: {p}")
    print(f"A: {resp.strip()[:200]}\n")

# ============================================================
# 10. DRIVE BACKUP
# ============================================================
print("--- 10. Drive backup ---")
try:
    from google.colab import drive
    drive.mount("/content/drive")
    import shutil
    bkp = f"/content/drive/MyDrive/{SAVE_NAME}"
    if os.path.exists(bkp): shutil.rmtree(bkp)
    shutil.copytree(f"/content/{SAVE_NAME}-final", bkp)
    print(f"Backup: {SAVE_NAME}")
except Exception as e:
    print(f"Drive skipped: {e}")

print(f"\n=== DONE ===")
print(f"Gold: {len(gold_filt)} | Conv: {len(conv_filt)}")
print(f"Total formatted: {len(formatted)} | Train: {len(train_ds)}")
print(f"Time: {train_time:.1f} min | Model: hadxs/{SAVE_NAME}")
