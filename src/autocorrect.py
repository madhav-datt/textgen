#
# Auto-correct spellings in generated str_text
# Based on C implementation in https://github.com/madhav-datt/spell-check
# Based on algorithm described by norvig.com
#

import io
import sys
import string

word_frequency_file = 'datasets/word_frequency.txt'
word_frequency = {}

# Total words in language model/ word_frequency
total_words = 0


def probability(word):
    """
    Compute probability of word
    :param word: string
    :return: probability of word
    """

    try:
        return float(word_frequency[word]) / total_words
    except KeyError:
        return 1.0 / total_words


def correction(word):
    """
    Find most probable spelling correction for word
    :param word: string to be corrected
    :return: String with corrected spelling
    """

    return max(candidates(word), key=probability)


def candidates(word):
    """
    Find set union of all possible spelling corrections of word
    :param word: string to be corrected
    :return: Set of all possible corrections for word
    """

    return known([word]) or known(edits_distance_1(word)) or known(edits_distance_2(word)) or [word]


def known(words):
    """
    Build set of words from 'words' that are actual words according to language model
    :param words: list of possible words based on edits
    :return: Subset of words that appear in word_frequency
    """

    return set(word for word in words if word in word_frequency)


def edits_distance_1(word):
    """
    Compute string with a edit distance 1 from 'word' using the following:
        deletion (remove one letter)
        transposition (swap two adjacent letters)
        replacement (replace one letter with another)
        insertion (add a letter)
    :param word: string
    :return: All words one edit distances away from 'word'
    """

    letters = 'abcdefghijklmnopqrstuvwxyz'
    slices = [(word[:i], word[i:]) for i in range(len(word) + 1)]

    deletes = [prefix + suffix[1:] for prefix, suffix in slices if suffix]
    transposes = [prefix + suffix[1] + suffix[0] + suffix[2:] for prefix, suffix in slices if len(suffix) > 1]
    replaces = [prefix + char + suffix[1:] for prefix, suffix in slices if suffix for char in letters]
    inserts = [prefix + char + suffix for prefix, suffix in slices for char in letters]

    return set(deletes + transposes + replaces + inserts)


def edits_distance_2(word):
    """
    :param word: string
    :return: All words two edit distances away from 'word'
    """

    return (edit_2 for edit_1 in edits_distance_1(word) for edit_2 in edits_distance_1(edit_1))


def has_numbers(input_str):
    """
    :param input_str: string
    :return: True if input_str contains a digit
    """

    return any(char.isdigit() for char in input_str)


def auto_correct(text):
    """
    Probabilistically and automatically correct spelling errors in str_text
    :param text: string with words to be automatically corrected
    :return: str_text with spellings fixed
    """

    global total_words
    with io.open(word_frequency_file, 'r', encoding='utf-8') as word_frequency_data:
        for line in word_frequency_data:
            frequency, word = line.strip().split()
            word_frequency[word] = int(frequency)

    total_words = sum(word_frequency.values())

    words = text.split()
    corrected_text = []
    for word in words:

        # Ignore words with digits
        if has_numbers(word):
            continue
        corrected_text.append(correction(word))

    return ' '.join(corrected_text)


def strip_non_ascii(utf_str):
    """
    Convert utf-8 strings to ASCII codec
    :param utf_str: utf-8 string
    :return: string without non ASCII characters
    """

    stripped = (c for c in utf_str if 0 < ord(c) < 127)
    return ''.join(stripped).encode(encoding='ascii', errors='ignore')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise AttributeError("Usage: python src/autocorrect.py correction_file output_file")

    with open(sys.argv[2], 'w') as write_file:
        str_text = strip_non_ascii(open(sys.argv[1]).read())
        # replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
        # str_text = str_text.translate(replace_punctuation).lower()
        write_file.write(auto_correct(str_text))
