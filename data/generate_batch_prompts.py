# Load training data and generate batch prompts

import pandas as pd
import sys

def main():
    sample_size = int(sys.argv[1])
    # Load ten random prompts from the training data.
    df = pd.read_csv("train.csv")
    df = df.sample(sample_size)
    prompts = df["instruction"].tolist()
    pre_prompt = "<s>[INST] <<SYS>>\nYou are a helpful, creative and honest assistant. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n"
    post_prompt = " [/INST]"
    separator = "\n-----\n"
    prompts = [pre_prompt + p + post_prompt + separator if i < sample_size -1 else pre_prompt + p + post_prompt for (i, p) in enumerate(prompts)]
    with open("batch_prompts_generated.txt", 'w+') as f:
        f.write("".join(prompts))
    pass


if __name__ == "__main__":
    main()