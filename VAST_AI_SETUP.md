# Vast AI Setup For This Repo

This file maps a generic Vast AI SSH setup onto this specific repository.

The key distinction is:

- some setup belongs on your local Windows machine,
- some setup belongs inside the repo,
- and some setup belongs on the remote Vast AI Linux instance.

## 1. What Goes Where

### Local machine only

These do **not** belong in the repo:

- your Vast API key
- your SSH private key
- your local SSH config
- your VS Code Remote SSH setup

Use these local paths on Windows:

- SSH private key: `C:\Users\ahrhq\.ssh\vast_key`
- SSH public key: `C:\Users\ahrhq\.ssh\vast_key.pub`
- SSH config: `C:\Users\ahrhq\.ssh\config`

### Repo files

These **do** belong in the repo:

- this guide: `VAST_AI_SETUP.md`
- remote bootstrap script: `scripts/vastai/bootstrap.sh`
- training launcher: `hsa_dpo_train.sh`

These should stay **local-only** and are ignored by Git:

- `VAST_AI_SETUP.local.md`
- `.vastai/`
- `scripts/vastai/*.local.sh`
- `scripts/vastai/*.local.env`
- `scripts/vastai/*.local.py`
- `scripts/vastai/local/`
- `vast_sync.tar`

### Remote Vast instance

These happen after you SSH into the rented GPU:

- clone the repo
- create or activate a Python environment
- install Linux training dependencies
- download the base LLaVA model
- run `hsa_dpo_train.sh`

## 2. Step 1: Prepare Your Local Windows Machine

### Vast CLI

Optional, but useful:

```powershell
python -m pip install vastai
vastai set api-key YOUR_API_KEY
```

### SSH key

Create a dedicated key for Vast:

```powershell
ssh-keygen -t ed25519 -f "$env:USERPROFILE\.ssh\vast_key"
```

### SSH config

Add this to `C:\Users\ahrhq\.ssh\config`:

```text
Host vastai
  HostName [INSTANCE_IP]
  User root
  Port [INSTANCE_PORT]
  IdentityFile C:\Users\ahrhq\.ssh\vast_key
  LocalForward 8080 localhost:8080
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
```

This is the correct place for the connection settings. Do not put this inside the repo.

## 3. Step 2: Rent The Vast AI Instance

Use a Linux CUDA image, not Windows.

For this repo, prefer:

- Ubuntu-based image
- NVIDIA GPU with enough VRAM for LLaVA-1.5-13B LoRA training
- CUDA-compatible PyTorch environment
- at least `2` GPUs if you want to keep `NUM_GPUS=2` in `hsa_dpo_train.sh`

After the instance starts:

1. copy the public IP
2. copy the SSH port
3. update `HostName` and `Port` in `C:\Users\ahrhq\.ssh\config`
4. connect with `ssh vastai` or VS Code Remote SSH

If `ssh vastai` times out, the instance is usually stopped/restarted and the IP or port changed.
Update `HostName` and `Port` from the currently active Vast instance panel and retry.

If OpenSSH reports `Bad owner or permissions on ...\\.ssh\\config` on Windows, reset ACLs:

```powershell
icacls "$env:USERPROFILE\\.ssh" /grant:r "$env:USERDOMAIN\\$env:USERNAME:(OI)(CI)F" "NT AUTHORITY\\SYSTEM:(OI)(CI)F" "BUILTIN\\Administrators:(OI)(CI)F" /t /c
```

## 4. Step 3: Clone The Repo On The Remote Machine

Run this on the Vast instance:

```bash
cd /workspace
git clone <YOUR_REPO_URL>
cd Fine-Grained-Hallucination-Detection-and-Severity-Aware-Mitigation-in-Vision-Language-Models
```

If you uploaded the repo another way, just `cd` into the project root on the remote machine.

## 5. Step 4: Bootstrap The Remote Python Environment

This repo now includes a remote setup script for Vast AI:

- `scripts/vastai/bootstrap.sh`

Run it on the remote machine from the repo root:

```bash
bash scripts/vastai/bootstrap.sh
```

What it does:

- creates `.venv` if needed
- upgrades `pip`, `setuptools`, and `wheel`
- installs this repo in editable mode with Linux training extras
- installs a `huggingface_hub` version compatible with this repo's `transformers` / LLaVA stack
- installs `modelscope`

What it does **not** do:

- it does not change your SSH config
- it does not inject API keys
- it does not download gated models automatically

## 6. Step 5: Download The Base Model

Run this on the Vast instance after the bootstrap script:

```bash
source .venv/bin/activate
hf download liuhaotian/llava-v1.5-13b --local-dir ./models/llava-v1.5-13b
```

The training script already points to:

- `MODEL_PATH="./models/llava-v1.5-13b"`

## 7. Step 6: Verify Repo Paths Before Running The Pipeline

This repo now has two image/data paths that matter:

- Stage 3 detection input: `./hsa_dpo/data/hsa_dpo_detection.jsonl`
- Visual Genome images for Stages 3-5: `./vg/images`
- model path: `./models/llava-v1.5-13b`
- Stage 5 / Stage 6 preference output: `./output/fghd/D_pref_clean_grouped.jsonl`

Stage 6 now reads image paths from the Stage 5 `image` field directly, so the repo root is the correct `IMAGE_FOLDER` for the current pipeline.

## 8. Step 7: Run The Calibrated Stage 3-5 Pipeline

The recommended remote entrypoint is:

```bash
source .venv/bin/activate
RUN_STAGE6=auto MODEL_PATH=models/llava-v1.5-13b IMAGE_ROOT="$(pwd)" bash scripts/run_calibrated_pipeline.sh
```

What this does:

- reruns Stage 3 calibration from `output/fghd/D_det.jsonl`
- runs Stage 4 grouped rewrite with the calibrated threshold report
- selects `tau_c` with CRC / CV-CRC
- runs Stage 5 final verification
- starts Stage 6 only when the machine has enough GPUs for the current setup

If the instance has fewer GPUs than required for Stage 6, the script stops cleanly after Stage 5.

## 9. Step 8: Start Stage 6 Only On A Suitable GPU Box

If you are on a real 2-GPU box and already have `output/fghd/D_pref_clean_grouped.jsonl`, run:

```bash
source .venv/bin/activate
DATA_PATH=output/fghd/D_pref_clean_grouped.jsonl \
IMAGE_FOLDER="$(pwd)" \
MODEL_PATH=models/llava-v1.5-13b \
bash scripts/run_stage6_train.sh
```

If the instance only has 1 GPU with 32 GB VRAM, the current Stage 6 configuration is expected to OOM. Use a larger box rather than forcing the same training setup onto that machine.

For a smaller first validation run, use:

```bash
bash scripts/vastai/run_pilot_train.sh
```

If you disconnect often, run inside `tmux`.

Useful commands:

```bash
tmux new -s hsa-dpo
bash hsa_dpo_train.sh
tmux attach -t hsa-dpo
```

## 10. What You Should Add Next

If you want to make the Vast AI workflow cleaner, the next repo changes should be:

1. add a stronger Stage 5 verifier if you want `tau_c` selection to be grounded in something stronger than the current heuristic backend
2. add a manually audited calibration subset if you want tighter paper-level claims around threshold selection
3. add a small env file or shell file for per-instance overrides
4. add checkpoint sync instructions for downloading results back to your laptop

## 11. Minimal Workflow Summary

### On Windows

1. set Vast API key
2. create SSH key
3. update `C:\Users\ahrhq\.ssh\config`
4. connect to `vastai`
5. keep any per-instance notes or one-off commands in `VAST_AI_SETUP.local.md`, not in tracked docs

### In this repo

1. keep `hsa_dpo_train.sh` as the baseline training entrypoint
2. use `scripts/vastai/bootstrap.sh`
3. use `scripts/run_calibrated_pipeline.sh` for the calibrated Stage 3-5 flow
4. use this document as the project-specific checklist
5. put machine-specific overrides in ignored `scripts/vastai/*.local.*` files rather than editing tracked setup docs for each instance

### On the Vast instance

1. clone repo
2. run `bash scripts/vastai/bootstrap.sh`
3. download LLaVA base model
4. run `bash scripts/run_calibrated_pipeline.sh`
5. run Stage 6 separately on a 2-GPU box if needed
