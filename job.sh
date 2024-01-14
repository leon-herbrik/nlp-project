#!/bin/bash
#SBATCH --partition=A100medium
#SBATCH --time=23:59:59
#SBATCH --gpus=1
#SBATCH --ntasks=1


python create_captions.py --categories explain generate wh-questions yesno
