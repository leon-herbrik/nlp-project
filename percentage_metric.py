import os
from Phyme import Phyme
import re
import json


def get_cleaned_lines(instruction_response):
    lines = instruction_response.partition("[/INST]")[2].split("\n")

    res = []
    for line in lines:
        if line:
            cleaned_line = line.partition("</s>")[0].strip(' ,.')
            res.append(cleaned_line)
            
    return res


def rhyming_percentage(text):
    """
    Returns the percentage of consecutive lines that rhyme with each other.
    """
    lines = get_cleaned_lines(text)

    rhyming_lines = 0
    if len(lines)%2==0:
        for i in range(0, len(lines), 2):
            if rhymes(lines[i].split(" ")[-1].lower(), lines[i + 1].split(" ")[-1].lower()):
                rhyming_lines += 2
    else:
        for i in range(len(lines)-1):
            if rhymes(lines[i].split(" ")[-1].lower(), lines[i + 1].split(" ")[-1].lower()):
                rhyming_lines += 1

    return rhyming_lines / (len(lines))


def rhymes(word1, word2):
    ph = Phyme()
    rhyming_criteria = {
        "perfect": ph.get_perfect_rhymes,
        "family": ph.get_family_rhymes,
        "partner": ph.get_partner_rhymes,
        "additive": ph.get_additive_rhymes,
        "subtractive": ph.get_subtractive_rhymes,
        "substitution": ph.get_substitution_rhymes,
        "assonance": ph.get_assonance_rhymes,
        "consonant": ph.get_consonant_rhymes,
    }
    # Return true if any of the rhyming criteria returns true
    return any(is_rhyming(word1, word2, criteria) for criteria in rhyming_criteria.values())
    

def is_rhyming(w1, w2, criteria):
    # If it's the same word, it's not rhyming.
    if w1 == w2:
        return False
    try:
        phyme_output = criteria(w1).values()
    except KeyError:
        return False
    
    all_corresponding_rhymes = []
    for rhymes in phyme_output:
        all_corresponding_rhymes += [re.sub(r"\(\d+\)", "", rhyme) for rhyme in rhymes]

    return w2 in all_corresponding_rhymes


def last_ascii_word(line):
    """
    Returns the last word in the line that is composed of ascii characters.
    """
    return re.findall(r"[a-zA-Z]+", line)[-1]


eval_result1 = json.load(open("batch_generated_leon.json", 'r'))
eval_result2 = json.load(open("batch_generated.json", 'r'))

epoch_percentage = {}
for model_name in eval_result1:
    epoch_idx = os.path.split(model_name)[1]
    print(epoch_idx)
    
    epoch_percentage[epoch_idx] = {}
    epoch_percentage[epoch_idx]["scores"] = []
    for response in eval_result1[model_name]:
        epoch_percentage[epoch_idx]["scores"].append(rhyming_percentage(response))

    for response in eval_result2[model_name]:
        epoch_percentage[epoch_idx]["scores"].append(rhyming_percentage(response))

    epoch_percentage[epoch_idx]["avg_score"] = f"{sum(epoch_percentage[epoch_idx]['scores'])/len(epoch_percentage[epoch_idx]['scores'])*100}%"

json.dump(epoch_percentage, open("rhyming_percentage_leon.json", 'w'))