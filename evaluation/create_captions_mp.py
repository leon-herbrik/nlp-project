# Standard packages.
import json
import os
import multiprocessing as mp
import traceback

# Third party packages.
from tqdm import tqdm

# Custom packages.
from inference import inference

def worker(category, counter, categories, exceptions):
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(categories.index(category) % 4)  # Set CUDA device
        file_name = f'data/{category}.json'
        with open(file_name, 'r') as f:
            data = json.load(f)
        # Iterate over all prompts in this category.
        for prompt in data.keys():
            if data[prompt] is not None:
                continue
            
            # Generate a response.
            response = inference("model", prompt)

            # Store the response.
            data[prompt] = response

            # Write to file.
            with open(file_name, 'w') as f:
                json.dump(data, f)

    except Exception as e:
        # Store exception.
        exceptions.append((category, prompt, e, traceback.format_exc()))
        print(e)
    finally:
        # Update progress bar.
        with counter.get_lock():
            counter.value += 1

def main():
    try:
        # Collect all prompts.
        categories = ['explain', 'generate', 'wh-questions', 'yesno']
        # Create a shared counter
        counter = mp.Value('i', 0)
        # Create a shared list for exceptions
        manager = mp.Manager()
        exceptions = manager.list()
        # Calculate the total number of prompts
        total = sum([len(json.load(open(f'data/{category}.json'))) for category in categories])
        # Create a pool of worker processes
        with mp.Pool(4) as pool:
            # Use starmap to pass multiple arguments to worker
            pool.starmap(worker, [(category, counter, categories, exceptions) for category in categories])
        with tqdm(total=total) as pbar:
            while counter.value < total:
                pbar.update(counter.value - pbar.n)
        # Print exceptions
        for category, prompt, exception, traceback in exceptions:
            print(f"Error processing prompt {prompt} in category {category}: {exception}")
            print(traceback)# Display progress bar in main process
    except Exception as e:
        print(e)    