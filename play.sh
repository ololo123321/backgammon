#!/usr/bin/env bash

export_dir=./models/backgammon_agent_v6_saved_model

python play.py \
    --export_dir=${export_dir} \
    --sign=1 \
    --k=2
