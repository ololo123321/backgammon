#!/usr/bin/env bash

model_dir="/tmp/backgammon_agent"

rm -r ${model_dir}

python main.py \
    --model_dir=${model_dir} \
    --n_episodes=100000 \
    --val_period=1000 \
    --n_val=100 \
    --save_period 1000 \
    --max_to_keep 3
