#!/usr/bin/env bash

model_dir="$(pwd)/model"
summaries_dir="$(pwd)/summaries"
checkpoints_dir="$(pwd)/checkpoints"

rm -r ${model_dir} ${summaries_dir} ${checkpoints_dir}

python main.py \
    --model_dir=${model_dir} \
    --summaries_dir=${summaries_dir} \
    --checkpoints_dir=${checkpoints_dir} \
    --n_episodes=1500 \
    --val_period=500 \
    --n_val=100
