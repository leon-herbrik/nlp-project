import torch
import itertools
from pathlib import Path as P
import pandas as pd


def get_custom_dataset(dataset_config, tokenizer, split):
    """
    Returns a custom dataset for the given split.
    """
    dataset = Dataset(tokenizer, split=split)
    return dataset


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        data_path: str = P.home() / "nlp-project/data",
        split: str = "train",
    ):
        self.tokenizer = tokenizer
        split: str = split
        self.B_INST: str = "[INST]"
        self.E_INST: str = "[/INST]"
        self.B_SYS: str = "<<SYS>>"
        self.E_SYS: str = "<</SYS>>"
        self.SYSTEM_PROMPT: str = "You are a helpful, creative and honest assistant. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        self.data = pd.read_csv(P(data_path) / f"{split}.csv")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        return self.process_row(sample)

    def process_row(self, sample):
        return self.tokenize_interaction(sample["instruction"], sample["response"])

    def tokenize_interaction(self, instruction, response) -> dict:
        prompt_tokens = self.tokenizer.encode(
            f"{self.tokenizer.bos_token} {self.B_INST} {self.B_SYS}\n{self.SYSTEM_PROMPT}\n{self.E_SYS}\n{instruction} {self.E_INST}\n\n",
            add_special_tokens=False,
        )
        answer_tokens = self.tokenizer.encode(
            f"{response} {self.tokenizer.eos_token}", add_special_tokens=False
        )

        interaction_tokens = list(itertools.chain(prompt_tokens, answer_tokens))
        labels_tokens = list(
            itertools.chain(
                len(prompt_tokens)
                * [
                    -100,
                ],
                answer_tokens,
            )
        )

        combined_tokens = {
            "input_ids": interaction_tokens,
            "labels": labels_tokens,
        }

        return dict(
            combined_tokens, attention_mask=[1] * len(combined_tokens["input_ids"])
        )


def test():
    """
    Test custom dataset.
    """

    class TOK:
        def __init__(self):
            self.bos_token = "<s>"
            self.eos_token = "</s>"

        def encode(self, arg, add_special_tokens):
            return arg.split(" ")

    tokenizer = TOK()
    # Create custom dataset.
    dataset = Dataset(tokenizer, split="alpaca")
    # Show a few stats about the dataset.
    print(dataset.data.head())
    print(dataset.data.info())
    # Query the dataset.
    print(dataset[0])
    pass


if __name__ == "__main__":
    test()
