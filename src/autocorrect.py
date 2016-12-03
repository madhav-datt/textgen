#
# Auto-correct spellings in generated text
# Based on C implementation in https://github.com/madhav-datt/spell-check
# Based on algorithm described by norvig.com
#

import io

word_frequency_file = 'datasets/word_frequency.txt'


def auto_correct(text):
    """
    Probabilistically and automatically correct spelling errors in text
    :param text: string with words to be automatically corrected
    :return: text with spellings fixed
    """
    word_frequency = {}

    with io.open(word_frequency_file, 'r', encoding='utf-8') as word_frequency_data:
        for line in word_frequency_data:
            frequency, word = line.split()
            word_frequency[word] = int(frequency)

