#!/bin/bash

python scripts/inference.py \
--outdir results \
--testdir examples \
--num_samples 3 \
--sample_steps 50 \
--gpu 0

# python scripts/inference.py \
# --outdir results \
# --testdir examples \
# --num_samples 1 \
# --sample_steps 25 \
# --plms \
# --gpu 0