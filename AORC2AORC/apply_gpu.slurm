#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --job-name=GCM2AORC
#SBATCH --output=out_%j.log

nvidia-smi