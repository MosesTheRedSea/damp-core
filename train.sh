#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=10:00:00
#$ -N dampnet
#$ -o logs/train_$JOB_ID.out
#$ -e logs/train_$JOB_ID.err

module purge
module load cuda/12.1
source /home/3/um07293/research/occul-net/.venv/bin/activate
cd /home/3/um07293/research/occul-net

# Single-task baselines 

# Object Detection
# python train.py --task det  --use_se --no_cross_attn --no_orth

# Distance Estimation
# python train.py --task dist --use_se --no_cross_attn --no_orth

# Material Classification
# python train.py --task mat  --use_se --no_cross_attn --no_orth

# All 3 heads

# python train.py --task all --use_se --no_cross_attn --no_orth

# Temporal-only baseline 
# python train.py --task all --use_se --temporal_only

# Ablation: w/o cross-attn
# python train.py --task all --use_se --no_cross_attn

# Ablation: w/o orth loss
# python train.py --task all --use_se --no_orth

# # DAMP Full Model
python train.py --task all --use_se
