#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=02:00:00
#$ -N dampnet
#$ -o logs/train_$JOB_ID.out
#$ -e logs/train_$JOB_ID.err

module purge
module load cuda/12.1
module load python/3.14.3

source /home/3/um07293/research/occlunet-unified/.venv/bin/activate

cd /home/3/um07293/research/occlunet-unified

mkdir -p logs

python train.py