#
# NIST Test for NLG text evaluation
#
# Implementation based on:
# Doddington, G., 2002, March. Automatic evaluation of machine translation quality using n-gram co-occurrence
# statistics. In Proceedings of the second international conference on Human Language Technology Research
# (pp. 138-145). Morgan Kaufmann Publishers Inc..
#
# Strong correlation with human expert evaluation for natural language generation (NLG) shown by
# Belz, A. and Reiter, E., 2006, April. Comparing Automatic and Human Evaluation of NLG Systems. In EACL.
#

import string
import io
import numpy as np

# Dictionary to memoize/cache information value of various n-grams
info_value = {}

# List of words in reference document
reference_data = None


def pre_process(str_text):
    """
    Perform all pre-processing steps on input string
    :param str_text: input
    :return: pre-processed list of tokens for training
    """

    str_text = str(str_text)
    # replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
    # str_text = str_text.translate(replace_punctuation)
    str_text = str_text.lower().replace('\n', ' ').replace('\r', ' ')
    return ' '.join(str_text.split())


def get_information_value(words):
    """
    Calculate information value from reference document
    Let words = [w_1, w_2, ..., w_n], n-gram,
    k_1 = Number of occurrences of (n-1)-gram [w_1, w_2, ..., w_{n-1}] in text
    k_2 = Number of occurrences of n-gram [w_1, w_2, ..., w_n] in text
    then Info(words) = log_2 (k_1 / k_2)
    :param words: list of words (n-gram)
    :return: Information value of words
    """

    if reference_data is None:
        raise ValueError("No reference data found")

    words_k1 = ' '.join(words)
    words_k2 = ' '.join(words[:-1])
    try:
        return info_value[words_k1]
    except KeyError:
        pass

    k1 = reference_data.count(words_k1)
    k2 = reference_data.count(words_k2)
    info_count = np.log2(float(k1 / k2))

    info_value[words_k1] = info_count
    return info_count


def evaluate_nlg(evaluation_file, reference_file='training/1.txt', N=5, beta=0):
    """

    :param evaluation_file:
    :param reference_file:
    :param N:
    :param beta:
    :return:
    """
    global reference_data

    with io.open(evaluation_file, 'r', encoding='utf-8') as evaluation:
        evaluation_data = pre_process(evaluation.read())

    with io.open(reference_file, 'r', encoding='utf-8') as reference:
        reference_data = pre_process(reference.read())
