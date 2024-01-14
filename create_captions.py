# Standard packages.
import json
import sys
import argparse

# Third party packages.

# Custom packages.
from inference import inference

def main():
    print('Starting prompt generation.')
    # Get current categories from arguments as list using argparse.
    parser = argparse.ArgumentParser()
    parser.add_argument('--categories', nargs='+', help='categories to generate prompts for')
    args = parser.parse_args()
    curr_categories = args.categories
    
    # Collect all prompts.
    categories = ['explain', 'generate', 'wh-questions', 'yesno']
    for curr_category in curr_categories:
        if curr_category not in categories:
            print(f'Category {curr_category} not found.')
            continue
        print(f'Generating prompts for category {curr_category}.')
        file_name = f'data/{curr_category}.json'
        # Load prompts.
        with open(file_name, 'r') as f:
            data = json.load(f)
        # Calculate the number of prompts.
        num_prompts = len(data)
        # Collect prompts.
        prompts = list(data.keys())
        # Iterate over all prompts.
        for i in range(num_prompts):
            # Get the prompt.
            prompt = prompts[i]
            # Check if the prompt is already answered.
            if data[prompt] is not None:
                continue
            # Generate a response.
            response = inference("model", prompt)
            # Store the response.
            data[prompt] = response
            # Write to file.
            with open(file_name, 'w') as f:
                json.dump(data, f)
            # Reload prompts.
            with open(file_name, 'r') as f:
                data = json.load(f)
            # Print amount of prompts that have been answered.
            answered = sum([1 for prompt in data.keys() if data[prompt] is not None])
            print(f'{answered}/{num_prompts} prompts answered for category {curr_category}.\n Current prompt: {response}')



if __name__ == '__main__':
    main()