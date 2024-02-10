# Fine-tune LLama 2 on Synthetically Generated Data
In this project, we use a three-step pipeline to generate training data for LLama 2 and fine-tune it on the generated data. The pipeline consists of:
1. Prompting LLama 2 to generate questions like 'Does the moon have its own light?'.
2. Querying LLama to answer the questions adhering to a certain role (in our case, 'poet'): 'The moon, a glowing orb in the night, its light, a beacon, a gentle delight'.
3. Fine-tuning LLama 2 on the generated data.

## Install
Create a new conda environment from the nlp.yml file:
```bash
conda env create -f nlp.yml
```
