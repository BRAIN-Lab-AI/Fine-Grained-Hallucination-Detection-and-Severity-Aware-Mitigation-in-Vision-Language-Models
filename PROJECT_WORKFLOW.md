# Project Workflow

This file covers **environment setup**, **`.env` configuration**, and the **project folder structure**.
It does not re-explain the research pipeline — see [README.md](README.md) for that.

---

## 1. Project Folder Structure

```text
.
├── .env                          ← local training config (you edit this)
├── .gitignore
├── README.md
├── PROJECT_WORKFLOW.md
├── VAST_AI_SETUP.md
├── pyproject.toml
├── hsa_dpo_train.sh              ← training launcher
│
├── asset/
│   ├── Referenced_Research_Paper.pdf
│   ├── HSA_DPO_Appendix.pdf
│   └── Proposal_summarized.pdf
│
├── hsa_dpo/
│   ├── data/
│   │   ├── hsa_dpo_preference_llava1dot5.jsonl   ← mitigation training data (8,386 rows)
│   │   ├── hsa_dpo_detection.jsonl               ← detection-stage data (references vg/)
│   │   └── images/                               ← mitigation training images (8,386)
│   ├── models/
│   │   └── llava-v1_5/                           ← bundled LLaVA code + DPO entrypoint
│   └── trainer/                                  ← HSA-DPO loss and trainer
│
├── inference/
│   ├── inference_example.py
│   └── inference_example.ipynb
│
├── models/
│   └── llava-v1.5-13b/           ← base model checkpoint (download separately)
│
├── output/
│   └── hsa_dpo_llava/            ← trained LoRA checkpoint written here
│
├── scripts/
│   └── vastai/
│       └── bootstrap.sh          ← remote setup helper for Vast AI
│
└── vg/
    └── images/                   ← Visual Genome images (only needed for detection stage)
```

---

## 2. Environment Setup

### Prerequisites

| Requirement | Notes |
| --- | --- |
| Python 3.10 – 3.12 | Match your system; 3.12 works on Windows for editing |
| NVIDIA GPU + CUDA | Required for training; 24 GB+ VRAM recommended |
| `deepspeed` | Linux only; installed via `linux-train` extras |
| `bash` | Training launcher is a bash script |

### Install the package

```bash
# Editable install (core dependencies only)
pip install -e .

# Add the Linux training stack when on a GPU machine
pip install -e ".[linux-train]"
```

---

## 3. Configuring `.env`

The file [.env](.env) is sourced automatically by [hsa_dpo_train.sh](hsa_dpo_train.sh) before training starts.
Edit it once before your first run and leave it in place.

### Step 1 — Set the number of GPUs

```bash
NUM_GPUS=2
CUDA_VISIBLE_DEVICES=0,1
```

- Use `NUM_GPUS=1` and `CUDA_VISIBLE_DEVICES=0` for a single-GPU machine.
- The count in `CUDA_VISIBLE_DEVICES` must match `NUM_GPUS`.

### Step 2 — Adjust batch size for your VRAM

```bash
BATCH_SIZE=8
```

| VRAM | Recommended `BATCH_SIZE` |
| --- | --- |
| 16 GB | 4 |
| 24 GB | 8 (default) |
| 40 GB+ | 16 |

Reduce if you hit out-of-memory errors; increase for faster throughput.

### Step 3 — Confirm the training data path

```bash
DATA_PATH=./hsa_dpo/data/hsa_dpo_preference_llava1dot5.jsonl
IMAGE_FOLDER=./hsa_dpo/data/images
```

- These point to the released dataset included in this repo.
- Do **not** change `IMAGE_FOLDER` to `image` — that folder does not exist.

### Step 4 — Set the base model path

```bash
MODEL_PATH=./models/llava-v1.5-13b
```

Download the base model from HuggingFace before training:

```bash
# Option A: HuggingFace CLI
huggingface-cli download liuhaotian/llava-v1.5-13b --local-dir ./models/llava-v1.5-13b

# Option B: git lfs
git clone https://huggingface.co/liuhaotian/llava-v1.5-13b ./models/llava-v1.5-13b
```

If the model is gated, fill in your token first:

```bash
HF_TOKEN=hf_your_token_here
```

### Step 5 — Set the output directory

```bash
OUTPUT_DIR=./output/hsa_dpo_llava
```

The trained LoRA checkpoint is written here after each epoch.
The directory is created automatically by the launcher if it does not exist.

### Step 6 — Leave training hyperparameters at their defaults (or tune)

```bash
EPOCH=2
LEARNING_RATE=2e-6
BETA=0.1
GRADIENT_ACCUMULATION_STEPS=1
```

These are the paper defaults. Change them only if you are running ablations.

### Step 7 — Leave the runtime paths as-is

```bash
VISION_TOWER=openai/clip-vit-large-patch14-336
DS_CONFIG=./hsa_dpo/models/llava-v1_5/scripts/zero3.json
ENTRY=./hsa_dpo/models/llava-v1_5/train_dpo.py
```

Change these only if you restructure the repo or swap the vision backbone.

### Step 8 — Configure W&B logging (optional)

By default, W&B logging is disabled so training starts without a login prompt:

```bash
WANDB_DISABLED=true
```

To enable cloud logging:

```bash
WANDB_DISABLED=false
WANDB_PROJECT=hsa-dpo
WANDB_API_KEY=your_wandb_api_key_here
```

---

## 4. Running Training

With `.env` configured, launch training from the repo root:

```bash
bash hsa_dpo_train.sh
```

The launcher will:

1. Load `.env`
2. Validate that all required paths exist
3. Check that `deepspeed` is on PATH
4. Launch DeepSpeed across `NUM_GPUS` with `train_dpo.py`

---

## 5. Running Inference

After training completes, test the checkpoint:

```bash
python inference/inference_example.py \
    --model-base ./models/llava-v1.5-13b \
    --lora-path  ./output/hsa_dpo_llava \
    --image      ./hsa_dpo/data/images/0.jpg \
    --prompt     "Describe this image in detail."
```

Optional inference flags:

| Flag | Default | Purpose |
| --- | --- | --- |
| `--temperature` | `0.7` | Sampling temperature |
| `--top-p` | `0.95` | Top-p nucleus sampling |
| `--num-beams` | `1` | Beam search width |
| `--max-new-tokens` | `1024` | Token budget |
| `--device` | `0` | CUDA device index |

---

## 6. Remote Setup (Vast AI)

For cloud GPU instances see [VAST_AI_SETUP.md](VAST_AI_SETUP.md).
The bootstrap script automates the full install:

```bash
bash scripts/vastai/bootstrap.sh
```
