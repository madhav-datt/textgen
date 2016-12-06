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

    k1 = 0.1 + reference_data.count(words_k1)
    k2 = 0.1 + reference_data.count(words_k2)
    info_count = np.log2(k1 / k2)

    info_value[words_k1] = info_count
    return info_count


def evaluate_nlg(evaluation_file, reference_file='training/1.txt', N=5, beta=0):
    """
    Compute evaluation score based of NIST evaluation metrics for machine translation
    Brevity factor = e^{beta * log ^ 2(min(1, L_sys / L_ref)}
    N-gram score = sum (n = 1 to N) {sum (all n-grams [w_1 .. w_n]) Info([w_1 .. w_n]] / sum (all n-grams [w_1 .. w_n])}
    Score = N-gram score * brevity factor
    :param evaluation_file: Text for evaluation
    :param reference_file: File with human generated reference text
    :param N: Maximum n-gram size, by default 5
    :param beta: Brevity penalty factor, by default 0
    :return: Evaluated score for file
    """

    global reference_data
    score = 0

    with io.open(evaluation_file, 'r', encoding='utf-8') as evaluation:
        evaluation_data = evaluation.read().encode('ascii', errors='ignore')
        evaluation_data = pre_process(evaluation_data).split()

    with io.open(reference_file, 'r', encoding='utf-8') as reference:
        reference_data = reference.read().encode('ascii', errors='ignore')
        reference_data = pre_process(reference_data)

    n = 1000
    reference_data = [reference_data[i:i + n].split() for i in range(0, len(reference_data), n)]

    return sentence_bleu(reference_data, evaluation_data)

    # Brevity penalty factor calculation
    eval_data_len = len(evaluation_data)
    word_ratio = float(eval_data_len) / len(reference_data.split())
    brevity_penalty = np.exp(beta * np.log(min(1, word_ratio)) ** 2)

    for n in xrange(1, N):
        ngrams = set()
        total_info = 0
        for i in xrange(eval_data_len - n):
            words = tuple(evaluation_data[i: i + n])
            if words in ngrams:
                continue

            ngrams.add(words)
            total_info += get_information_value(words)

        score += float(total_info) / len(ngrams)

    score *= brevity_penalty
    return score


if __name__ == '__main__':
    from bleu import *
    # train_score = evaluate_nlg(evaluation_file='result.txt', reference_file='training/4-mod.txt')
    # test_score = evaluate_nlg(reference_file='result.txt', evaluation_file='training/1.txt')
    test_score = evaluate_nlg(evaluation_file='result.txt', reference_file='training/4-mod.txt')
    # print "Against training set: ", train_score
    print "Against test set: ", test_score
