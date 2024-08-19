#!/usr/bin/env python
from guidance import models, gen, select
from guidance.chat import llama3_template, Llama3ChatTemplate
from guidance import user, assistant, system
from transformers import AutoTokenizer
import time
import json


evaluator_agent = models.LlamaCpp("models/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf", n_gpu_layers=-1, n_ctx=4096, chat_template=llama3_template, echo=False)
tokenizer = AutoTokenizer.from_pretrained("TechxGenus/Meta-Llama-3-8B-GPTQ")

def truncate(tokenizer, input_str: str, maxlen: int) -> str:
    return tokenizer.decode(tokenizer.encode(input_str)[:maxlen])

with system():
    evaluator_agent += """You are a helpful assistant and a very stringent evaluator. You will be provided with a prompt and a response. Analyze how well the response followed the prompt. Focus on two things and evaluate them:
1. Did the answer follow the role in the prompt well?
2. Does the answer factually answer the prompt."""
    

tests = json.load(open("human_test.json", 'r'))

sys_role = "You are a helpful, respectful and honest assistant. However it is your role to only answer in poems or rhymes. Use a pair-rhyme for answering."

eval_epoch_start_id = 0
eval_epoch_end_id = 10

eval_results = {}
for epoch_id in range(eval_epoch_start_id, eval_epoch_end_id):
    eval_results[f"epoch_{epoch_id}"] = {}
    for i in range(10):
        print(f"Evaluating interaction No. {i} of Epoch No. {epoch_id}")
        epoch_key = f"/home/s6leherb/nlp-project/model/finetunes/epochs/epoch_{epoch_id}"
        instruction = tests[epoch_key][i].partition("[/INST]")[0].rpartition("<</SYS>>")[2]
        response = tests[epoch_key][i].partition("[/INST]")[2]
        response = truncate(tokenizer, response, maxlen=600)
    
        with user():
            eval_case = evaluator_agent + f'''Prompt: "{instruction}"
Role of the prompt (don't take up this role): "{sys_role}"
Response: "{response}"'''
        
        t = time.time()
        with assistant():
            newline = "\n"
            expression = "\d{1,3}"
            eval_case += f"""In terms of role following, my one liner evaluation explanation would be: {gen("role following explanation", stop=newline)}
The possible shortcomings of the response might be: {gen("shortcoming", stop=newline)}
As a result, considering the shortcomings, I give the prompt and response a score percentage of {gen("role following score", regex=expression)}%
In terms of factuality, my short one liner evaluation explanation would be: {gen("factuality explanation", stop=newline)}
As a result, It gets a score percentage of {gen("factuality score", regex=expression)}%"""        
        
        elapsed = round(time.time() - t, 2)
        print(f"Eval took {elapsed} seconds!")
        print("Role following explanation:", eval_case["role following explanation"])
        print("Shortcomings:", eval_case["shortcoming"])
        print("Role following score:", eval_case["role following score"])
        print("Factuality explanation:", eval_case["factuality explanation"])
        print("Factuality score:", eval_case["factuality score"])
        
        eval_results[f"epoch_{epoch_id}"][i] = {
            "Role": sys_role,
            "Instruction": instruction,
            "Response": response,
            "Role following explanation": eval_case["role following explanation"],
            "Factuality explanation": eval_case["factuality explanation"],
            "Critiques": eval_case["shortcoming"],
            "Evaluation time": elapsed,
            "Role following score": eval_case["role following score"],
            "Factuality score": eval_case["factuality score"]
        }
        
        json.dump(eval_results, open("eval_results_tmp.json", 'w'))
        
        