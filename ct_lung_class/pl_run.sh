#!/bin/bash

python ct_lung_class/pl_main.py \
    --batch-size 16 \
    --dataset zara \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --box-size 65 65 65 \
    --fixed-size \
    --k-folds 1 \
    --learn-rate 0.0001 \
    --dropout 0.3 \
    --weight-decay 0.001 \
    --max-epochs 6000 \
    --devices 3 \
    --tag zara-densenet201 \
    --loss focal \
    --model densenet201
