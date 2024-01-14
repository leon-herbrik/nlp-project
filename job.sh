#!/bin/bash
#SBATCH --partition=A100medium
#SBATCH --time=8:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1


python script.py
