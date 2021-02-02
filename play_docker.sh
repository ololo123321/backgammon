#!/usr/bin/env bash

export_dir="$(pwd)/models/backgammon_agent_v6_saved_model"

export_dir_mnt=/model

docker run -it \
    -v ${export_dir}:${export_dir_mnt} \
    backgammon:0.0.0 python play.py \
        --export_dir=${export_dir_mnt} \
        --sign=1 \
        --k=2
