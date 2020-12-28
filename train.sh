#!/usr/bin/env bash

python main.py \
    --model_dir="$(pwd)/model" \
    --summaries_dir="$(pwd)/summaries" \
    --checkpoints_dir="$(pwd)/checkpoints" \
    --n_episodes=10000 \
    --val_period=500 \
    --n_val=100
