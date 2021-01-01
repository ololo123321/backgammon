#!/usr/bin/env bash

model_dir="/tmp/backgammon_agent"

rm -r ${model_dir}

python main.py \
    --model_dir=${model_dir} \
    --encoder=TesauroEncoder \
    --enc_params='{"hidden_dim": 80}' \
    --num_games_training=100000 \
    --val_period=1000 \
    --num_games_test=100 \
    --save_period=1000 \
    --max_to_keep=3
