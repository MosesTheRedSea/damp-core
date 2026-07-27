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

# python train.py --task det --use_se

# python train.py --task dist --use_se

# python train.py --task mat --use_se

# python train.py --task all --no_cross_attn --use_se

# python train.py --task all --temporal_only --use_se

# python train.py --task all --no_orth --use_se

# python train.py --task all

python train.py --task all --no_cross_attn --temporal_only --no_orth