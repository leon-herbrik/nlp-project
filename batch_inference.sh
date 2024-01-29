#!/bin/bash
#SBATCH --partition=A40medium
#SBATCH --time=23:59:59
#SBATCH --gpus=1
#SBATCH --ntasks=1
cd /home/s6leherb/nlp-project
python batch_infer.py --model_name /home/s6leherb/nlp-project/model/Llama-2-7b-chat-hf --peft_models_folder /home/s6leherb/nlp-project/model/finetunes/epochs/ --prompt_file /home/s6leherb/nlp-project/batch_prompts_generated.txt