#!/bin/bash
#SBATCH --partition=A40medium
#SBATCH --time=23:59:59
#SBATCH --gpus=1
#SBATCH --ntasks=1
cd /home/s6leherb/nlp-project/llama-recipes
python examples/inference.py --model_name /home/s6leherb/nlp-project/model/Llama-2-7b-chat-hf --peft_model /home/s6leherb/nlp-project/model/finetunes/epochs/epoch_12 --prompt_file /home/s6leherb/nlp-project/prompt.txt