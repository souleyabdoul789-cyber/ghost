#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python -m pip install -U '.[data]'
python examples/fetch_hf_stream.py \
  --dataset Brainquiver/general-web-fr-202608 \
  --config fineweb2-hq-fra_Latn \
  --revision main \
  --output data/fineweb2_fr_pilot.jsonl \
  --max-documents 10000 \
  --max-total-chars 20000000 \
  --min-language-score 0.95 \
  --min-quality-score 0.30 \
  --resume
python examples/train_ghost.py data/fineweb2_fr_pilot.jsonl \
  --output runs/ghost-fr-pilot-001 \
  --vocab-size 1024 --min-frequency 2 \
  --d-model 64 --heads 4 --layers 2 --context 128 \
  --epochs 3 --steps-per-epoch 500 --checkpoint-every-steps 50 \
  --learning-rate 3e-4 --weight-decay 0.01 \
  --tokens-per-shard 250000 --seed 42 --resume
python examples/write_training_report.py \
  --run-dir runs/ghost-fr-pilot-001 \
  --source-manifest data/fineweb2_fr_pilot.jsonl.source.json \
  --output runs/ghost-fr-pilot-001/report.txt
