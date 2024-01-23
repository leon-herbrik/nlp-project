#!/bin/bash
#SBATCH --partition=A40medium
#SBATCH --time=23:59:59
#SBATCH --gpus=1
#SBATCH --ntasks=1

python -m llama_recipes.finetuning --dataset "custom_dataset" --custom_dataset.file "/home/s6leherb/nlp-project/custom_dataset.py"