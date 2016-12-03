#
# Pre-process text into usable datasets
#
#

import io
import nltk
import string
from os import listdir
from os.path import isfile, join


def pre_process(str_text):
    """
    Perform all pre-processing steps on input string
    :param str_text: input
    :return: pre-processed list of tokens for training
    """
    replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
    str_text = str(str_text)
    str_text = str_text.translate(replace_punctuation).lower().replace('\n', ' ').replace('\r', ' ')
    return ' '.join(str_text.split())

raw_training_path = 'training/'
raw_train_data = [f for f in listdir(raw_training_path) if isfile(join(f, raw_training_path))]

output = 'train.txt'

with io.open(output, 'w', encoding='utf-8') as output_file:
    for raw_file in raw_train_data:
        with io.open(join(raw_file, raw_training_path), 'r', encoding='utf-8') as input_file:
            # Pre-process each input file and write to output file
            output_file.write(pre_process(input_file.read()))
