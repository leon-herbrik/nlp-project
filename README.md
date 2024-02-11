# Fine-tune LLama 2 on Synthetically Generated Data
In this project, we use a three-step pipeline to generate training data for LLama 2 and fine-tune it on the generated data. The pipeline consists of:
1. Prompting LLama 2 to generate questions like 'Does the moon have its own light?'.
2. Querying the model to answer the questions adhering to a certain role (in our case, 'poet'): 'The moon, a glowing orb in the night, its light, a beacon, a gentle delight'.
3. Fine-tuning on the generated data.

## Install
Create a new conda environment from the nlp.yml file:
```bash
conda env create -f nlp.yml
```

## Usage
1. For step 1 of the pipeline, use the following command:
```bash
python generate_questions.py
```
2. For step 2 of the pipeline, use the following command:
```bash
python create_captions.py
```
3. For step 3 of the pipeline, use the following command:
```bash
python llama-recipes/src/finetuning.py