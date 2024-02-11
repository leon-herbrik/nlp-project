"""
This script is for step 1 of the pipeline. It is used to generate a predefined number of general purpose questions from the model.
"""

# Standard packages
import os
import sys
import json
import argparse

# Local packages
from inference import inference


def main():
    """
    Generate n general purpose questions.
    Store them in a JSON file.

    @param n: Number of questions to generate.
    @param output_file: Path to the output file.
    """
    parser = argparse.ArgumentParser(description="Generate general purpose questions.")
    parser.add_argument(
        "-n",
        "--number_of_questions",
        required=False,
        default=750,
        type=int,
        help="Number of questions to generate.",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        required=False,
        default="questions.json",
        type=str,
        help="Path to the output file.",
    )
    categories = ["wh-questions", "yes/no questions", "explain", "create"]
    prompts = [
        "Create 50 wh-questions about general topics. Example: What is the capital of France? Cover as many topics and types of questions as possible. Don't repeat questions.",
        "Create 50 yes/no questions about general topics. Example: Is the sun a star? Cover as many topics and types of questions as possible. Don't repeat questions.",
        "Explain 50 general topics. Example: Explain the concept of entropy. Cover as many topics and types of questions as possible. Don't repeat questions.",
        "Create 50 instructions about creative topics. Example: Create a poem about the universe. Cover as many topics and types of questions as possible. Don't repeat questions.",
    ]
    args = parser.parse_args()
    n = args.number_of_questions
    output_file = args.output_file
    questions = {category: [] for category in categories}
    for category, prompt in zip(categories, prompts):
        for i in range(n):
            prompt_template = f"""[INST] <<SYS>>
            You are a helpful, respectful and honest assistant. However it is your role to only answer in poems or rhymes. Use a pair-rhyme for answering.
            <</SYS>>
            {prompt}[/INST]"""
            question = inference(prompt)
            questions[category].append(question)
            print(f"{category} {i+1}/{n}")
    with open(output_file, "w") as f:
        json.dump(questions, f, indent=4)


if __name__ == "__main__":
    main()
