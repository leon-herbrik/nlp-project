#!/bin/bash
#SBATCH --partition=A40medium
#SBATCH --time=8:00:00
#SBATCH --gpus=4
#SBATCH --ntasks=1


python device_count.py
