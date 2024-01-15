# Standard packages.
import json
import sys
import argparse
import time

# Third party packages.

# Custom packages.
from inference import inference

def main():
    print('Starting prompt generation.', flush=True)
    # Get current categories from arguments as list using argparse.
    parser = argparse.ArgumentParser()
    parser.add_argument('--categories', nargs='+', help='categories to generate prompts for')
    args = parser.parse_args()
    arg_categories = args.categories
    valid_categories = ['explain', 'generate', 'wh-questions', 'yesno']
    
    # Remove all categories that are not in the list.
    curr_categories = [category for category in arg_categories if category in valid_categories]
    # Filter categories where all prompts are already answered.
    curr_categories = filter_categories(curr_categories)
    # Iterate over all categories.
    for curr_category in curr_categories:
        print(f'Generating prompts for category {curr_category}.', flush=True)
        file_name = f'data/{curr_category}.json'
        # Load prompts.
        with open(file_name, 'r') as f:
            data = json.load(f)
        # Calculate the number of prompts.
        num_prompts = len(data)
        # Collect prompts.
        prompts = list(data.keys())
        # Average timing
        avg_time = 0
        # Iterate over all prompts.
        for i in range(num_prompts):
            # Get the prompt.
            prompt = prompts[i]
            # Check if the prompt is already answered.
            if data[prompt] is not None:
                continue
            # Print amount of prompts that have been answered.
            answered = sum([1 for prompt in data.keys() if data[prompt] is not None])
            print(f'{answered}/{num_prompts} prompts answered for category {curr_category}.', flush=True)
            # Start timing.
            start = time.time()
            # Generate a response.
            response = inference("model", prompt)
            # Store the response.
            data[prompt] = response
            # Stop timing.
            diff = time.time() - start
            # Calculate the time taken.
            avg_time = (avg_time * i + diff) / (i + 1)
            # Print the remaining time.
            print(f'Processing time: {diff}. Estimated time remaining: {avg_time * (num_prompts - i)} seconds for category {curr_category}.', flush=True)
            # Write to file.
            with open(file_name, 'w') as f:
                json.dump(data, f)
            # Reload prompts.
            with open(file_name, 'r') as f:
                data = json.load(f)
            # Print the response.
            print(f'Current prompt: {response}', flush=True)
        print(f'Finished generating prompts for category {curr_category}.', flush=True)
    print('Finished prompt generation.', flush=True)

def filter_categories(categories):
    """
    This function is used to filter categories where all prompts are already answered.
    """
    # Remove categories where all prompts are already answered.
    for curr_category in categories:
        file_name = f'data/{curr_category}.json'
        # Load prompts.
        with open(file_name, 'r') as f:
            data = json.load(f)
        # Check if all prompts are answered.
        if all([data[prompt] is not None for prompt in data.keys()]):
            print(f'All prompts answered for category {curr_category}.', flush=True)
            categories.remove(curr_category)
    return categories

if __name__ == '__main__':
    main()