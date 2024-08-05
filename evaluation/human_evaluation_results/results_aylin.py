import json

with open('evaluation/human_evaluation_results/results_aylin.json', 'r') as f:
    data = json.load(f)
with open('evaluation/human_test_cleaned.json', 'r') as f:
    base = json.load(f)
    
# Replace keys in data with keys in base
keys = list(base.keys())
count = 0
new_data = {}
for key, value in data.items():
    for inner_key, inner_value in value.items():
        prompt_role = inner_value['prompt_role']
        factual_answer = inner_value['factual_answer']
        new_data[keys[count]] = {
            "role_score": prompt_role,
            "fact_score": factual_answer
        }
        count += 1
pass
with open('results_aylin_new.json', 'w') as f:
    json.dump(new_data, f)




