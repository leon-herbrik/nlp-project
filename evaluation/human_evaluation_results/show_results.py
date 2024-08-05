from __future__ import annotations
import matplotlib.pyplot as plt
import json
import os
from dataclasses import dataclass, field
from typing import List, Callable, Tuple
from itertools import zip_longest
import pandas as pd
from pandas.plotting import parallel_coordinates


@dataclass
class Pipeline:
    commands: List[Pipeline.Command] = field(default_factory=list)
    
    @dataclass
    class Command:
        function: Callable
        names: List[str | List[str]]
        name_overwrites: List[str | List[str]] = field(default_factory=list)
        reductions: List[Callable] = field(default_factory=list)
        
        def __post_init__(self):
            self.data = []
            for (entry, reduction) in zip_longest(self.names, self.reductions, fillvalue=lambda x: x):
                self.data.append(reduction(get_data(entry)))
        
        def execute(self):
            names = self.names
            if len(self.name_overwrites) > 0:
                names = self.name_overwrites
            self.function(self.data, names)
    
    def execute(self):
        for command in self.commands:
            command.execute()
    
    def add_command(self, function: Callable, names: List[str | List[str]], reductions: List[Callable] = [], name_overwrites: List[str] = []):
        self.commands.append(self.Command(function, names, name_overwrites, reductions))
            

def get_data(names):
    data = {name: json.load(open(f'results_{name}.json')) for name in names}
    # Load each name's results.
    keys = [key for name in data for key in data[name]['responses'].keys()]
    keys = list(dict.fromkeys(keys))
    overall_data = {key: [] for key in keys}
    for name in data:
        for key, value in data[name]['responses'].items():
            overall_data[key].append(value)
    return overall_data


def create_training_progress(data, names):
    """
    Create a figure of training progress over the epochs.
    """
    left, right = data
    names_left, names_right = names
    title_role = f"Role Score Training Progress of {', '.join(names_left)} and {', '.join(names_right)}"
    path_role = f"role_{'_'.join(names_left)}&{'_'.join(names_right)}_training_progress.png"
    title_fact = f"Fact Score Training Progress of {', '.join(names_left)} and {', '.join(names_right)}"
    path_fact = f"fact_{'_'.join(names_left)}&{'_'.join(names_right)}_training_progress.png"
    left_role = []
    left_fact = []
    right_role = []
    right_fact = []
    for i, (key, value) in enumerate(left.items()):
        if i % 10 == 0:
            cur_role = 0
            cur_fact = 0
        cur_role_aggr = 0
        cur_fact_aggr = 0
        for entry in value:
            cur_role_aggr += entry['role_score']
            cur_fact_aggr += entry['fact_score']
        cur_role += cur_role_aggr / len(value)
        cur_fact += cur_fact_aggr / len(value)
        if i % 10 == 9:
            left_role.append(cur_role / 10)
            left_fact.append(cur_fact / 10)
    for i, (key, value) in enumerate(right.items()):
        if i % 10 == 0:
            cur_role = 0
            cur_fact = 0
        cur_role_aggr = 0
        cur_fact_aggr = 0
        for entry in value:
            cur_role_aggr += entry['role_score']
            cur_fact_aggr += entry['fact_score']
        cur_role += cur_role_aggr / len(value)
        cur_fact += cur_fact_aggr / len(value)
        if i % 10 == 9:
            right_role.append(cur_role / 10)
            right_fact.append(cur_fact / 10)
    plt.figure()
    plt.plot(left_role, label=f"Role Score {', '.join(names_left)}")
    plt.plot(right_role, label=f"Role Score {', '.join(names_right)}")
    # Add legend.
    plt.legend()
    plt.title(title_role)
    plt.savefig(path_role)
    plt.clf()
    plt.close()
    plt.figure()
    plt.plot(left_fact, label=f"Fact Score {', '.join(names_left)}")
    plt.plot(right_fact, label=f"Fact Score {', '.join(names_right)}")
    plt.legend()
    plt.title(title_fact)
    plt.savefig(path_fact)
    plt.clf()
    plt.close()
    
    # Export the data to a json file.
    with open('training_progress.json', 'w') as f:
        json.dump({'human': {'role': left_role, 'fact': left_fact}, 'automatic': {'role': right_role, 'fact': right_fact}}, f)        
    


def create_boxplot(data, names):
    """
    Create and store a matplotlib figure showing a boxplot of the overall role_score and fact_score.
    """
    data = data[0]
    names = names[0]
    title = f"Boxplot of Role and Fact Scores for {', '.join(names)}"
    path = f"{'_'.join(names)}_boxplot.png"
    role_scores = [entry['role_score'] for key, value in data.items() for entry in value]
    fact_scores = [entry['fact_score'] for key, value in data.items() for entry in value]
    plt.figure()
    plt.boxplot([role_scores, fact_scores])
    # Give names to each box plot (xlabel).
    plt.xticks([1, 2], [f'Role Score #samples: {len(role_scores)}', f'Fact Score #samples: {len(fact_scores)}'])
    plt.title(title)
    plt.savefig(path)
    plt.clf()
    plt.close()


def create_parallel_coordinates(data, names):
    """
    Create parallel coordinates plot with pandas plotting.
    """
    left, right = data
    names_left, names_right = names
    title_role = f"PC-Plot: Role Score of {', '.join(names_left)} and {', '.join(names_right)}"
    title_fact = f"PC-Plot: Fact Score of {', '.join(names_left)} and {', '.join(names_right)}"
    path_role = f"role_{'_'.join(names_left)}&{'_'.join(names_right)}_parallel_coordinates.png"
    path_fact = f"fact_{'_'.join(names_left)}&{'_'.join(names_right)}_parallel_coordinates.png"
    # Create a dataframe for each name.
    df_left = create_dataframe(left)
    df_right = create_dataframe(right)
    name_role_left = f"Role {', '.join(names_left)}"
    name_role_right = f"Role {', '.join(names_right)}"
    name_fact_left = f"Fact {', '.join(names_left)}"
    name_fact_right = f"Fact {', '.join(names_right)}"
    df_left[name_role_left] = df_left['role_score']
    df_left[name_role_right] = df_right['role_score']
    df_left[name_fact_left] = df_left['fact_score']
    df_left[name_fact_right] = df_right['fact_score']
    parallel_coordinates(df_left, "class", cols=[name_role_left, name_role_right])
    plt.title(title_role)
    plt.legend('',frameon=False)
    # Resize plot.
    plt.tight_layout()
    plt.savefig(path_role)
    plt.clf()
    plt.close()
    parallel_coordinates(df_left, "class", cols=[name_fact_left, name_fact_right])
    plt.title(title_fact)
    plt.legend('',frameon=False)
    plt.tight_layout()
    plt.savefig(path_fact)
    plt.clf()
    plt.close()
    
def create_dataframe(data):
    df = pd.DataFrame()
    ids = []
    classes = []
    scores = {}
    for key, value in data.items():
        ids.append(key)
        classes.append(key.split('_')[1])
        for entry in value:
            for score_name, score in entry.items():
                if score_name not in scores:
                    scores[score_name] = []
                scores[score_name].append(score)
    if "id" not in df.columns:
        # Add column.
        df["id"] = pd.Series(ids)
    if "class" not in df.columns:
        # Add column.
        df["class"] = pd.Series(classes)
    for key, value in scores.items():
        df[key] = pd.Series(value)
    return df

def mean(data):
    for id, entry_list in data.items():
        aggregator = {}
        for entry in entry_list:
            for key, value in entry.items():
                if key not in aggregator:
                    aggregator[key] = []
                aggregator[key].append(value)
        for key, value in aggregator.items():
            aggregator[key] = sum(value) / len(value)
        data[id] = [aggregator]
    return data
        



if __name__ == '__main__':
    names = [['aylin', 'jenny', 'leon', 'claudius'], ['automatic']]
    pipeline = Pipeline()
    commands = [
        (create_training_progress, names, [mean], [['Human Evaluators'], ['Automatic Evaluation']]),
        (create_boxplot, [names[0]], [], [['Human Evaluators']]),
        (create_boxplot, [names[1]], [], [['Automatic Evaluation']]),
        (create_parallel_coordinates, names, [mean], [['Human Evaluators'], ['Automatic Evaluation']])
    ]
    for command in commands:
        pipeline.add_command(*command)
    pipeline.execute()