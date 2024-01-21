# Load 'instruct_dataset_final.csv' and do a random split of 70/20/10 for train/val/test
from pathlib import Path as P
import pandas as pd


def main():
    data = pd.read_csv(P("instruct_dataset_final.csv"))
    # Rename first column to 'id'.
    data = data.rename(columns={"Unnamed: 0": "id"})
    # Rename 'raw_response' to 'response'.
    data = data.rename(columns={"raw_response": "response"})
    data = data.sample(frac=1, random_state=42)
    train, test, val = (
        data[: int(len(data) * 0.7)],
        data[int(len(data) * 0.7) : int(len(data) * 0.9)],
        data[int(len(data) * 0.9) :],
    )
    train.to_csv("train.csv", index=False)
    val.to_csv("validation.csv", index=False)
    test.to_csv("test.csv", index=False)


if __name__ == "__main__":
    main()
