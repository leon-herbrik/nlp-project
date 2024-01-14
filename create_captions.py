# Standard packages.
import json

# Third party packages.
from tqdm import tqdm

# Custom packages.
from inference import inference

def main():
    # Collect all prompts.
    categories = ['explain', 'generate', 'wh-questions', 'yesno']
    file_names = [f'data/{category}.json' for category in categories]
    # Create a dictionary of prompts.
    dicts = {
        category: {} for category in categories
    }
    # Get all prompts by category.
    prompts = {
        category: list(dicts[category].keys()) for category in categories
    }
    # Load prompts.
    for file_name, category in zip(file_names, categories):
        with open(file_name, 'r') as f:
            data = json.load(f)
            dicts[category] = data
    # Calculate the number of prompts.
    num_prompts = sum([len(dicts[category]) for category in categories])
    # Iterate over all prompts.
    for i in tqdm(range(num_prompts)):
        # Choose one category each time.
        category = categories[i % len(categories)]
        # Get the prompt.
        prompt = prompts[category][i // len(categories)]
        # Check if the prompt is already answered.
        if dicts[category][prompt] is not None:
            continue
        # Generate a response.
        response = inference("model", prompt)
        # Store the response.
        dicts[category][prompt] = response
        # Write to file after every 10 prompts.
        if i % 10 == 0:
            for file_name, category in zip(file_names, categories):
                with open(file_name, 'w') as f:
                    json.dump(dicts[category], f)



if __name__ == '__main__':
    main()