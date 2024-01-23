import json
import pandas as pd


def main():
    """
    Read json file of format [{
        "instruction": "Give three tips for staying healthy.",
        "input": "",
        "output": "1.Eat a balanced diet"
    }, ...]
    and convert to csv format:
    id,instruction,raw_response,input
    """
    with open("alpaca_data.json", "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["id"] = df.index
    df = df.rename(columns={"output": "raw_response"})

    # Store as csv.
    df.to_csv("alpaca_data.csv", index=False)

    pass


if __name__ == "__main__":
    main()
