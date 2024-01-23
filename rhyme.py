from Phyme import Phyme
import re

def rhyming_percentage(text):
    """
    Returns the percentage of consecutive lines that rhyme with each other.
    """
    lines = text.split("\n")
    # Remove any empty lines or those consisting of spaces.
    lines = [line for line in lines if line.strip()]
    rhyming_lines = 0
    for i in range(len(lines) - 1):
        if rhymes(last_ascii_word(lines[i]), last_ascii_word(lines[i + 1])):
            rhyming_lines += 1
    return rhyming_lines / (len(lines) - 1)


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