import matplotlib.pyplot as plt
import json
import os
import pandas as pd
from pandas.plotting import parallel_coordinates




def get_overall_data(names):
    data = {name: json.load(open(f'evaluation/human_evaluation_results/results_{name}.json')) for name in names}
    # Load each name's results.
    keys = [key for name in data for key in data[name]['responses'].keys()]
    keys = list(dict.fromkeys(keys))
    overall_data = {key: [] for key in keys}
    for name in data:
        for key, value in data[name]['responses'].items():
            overall_data[key].append(value)
    return overall_data
    
def create_overall_graph(overall_data, path="overall_scores.png", title="Overall Role and Fact Scores"):
    """
    Create and store a matplotlib figure showing a boxplot of the overall role_score and fact_score.
    """
    role_scores = [int(entry['role_score']) for key, value in overall_data.items() for entry in value]
    fact_scores = [int(entry['fact_score']) for key, value in overall_data.items() for entry in value]
    plt.figure()
    plt.boxplot([role_scores, fact_scores])
    # Give names to each box plot (xlabel).
    plt.xticks([1, 2], ['Role Score', 'Fact Score'])
    plt.title(title)
    plt.savefig(path)


if __name__ == '__main__':
    human = ['aylin', 'jenny', 'leon']
    human_data = get_overall_data(human)
    create_overall_graph(human_data, "overall_scores_human.png", title="Overall Human Role and Fact Scores")
    automatic = ['v2']
    automatic_data = get_overall_data(automatic, "overall_scores_automatic.png", title="Overall Automatic Role and Fact Scores")
    create_overall_graph(automatic_data)
    