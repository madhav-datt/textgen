#
# Markov model for text generation
#

import numpy
import random

def process_text(filename):

    raw_text = open(filename).read()
    raw_text = raw_text.lower()
    return raw_text

def transition(raw_text, n):

    table = {}
    # Count every n-character substring of raw_text
    for i in range(0, len(raw_text)):
        substring = raw_text[i: i + n]

        if substring in table:
            table[substring] += 1
        # Ignore edge case where the substring is not of length n
        elif len(substring) == n:
            table[substring] = 1


    #print table
    return table

def generate(table, n_out, n, cti, itc, smooth):
    
    # Fixed seed for now
    seed = "to sherlock holmes she is always the woman."
    output_text = seed
    # The current n-1 characters we want the next character for
    current = output_text[len(seed) - (n-1):]
    print(n, len(current))

    char = 0

    while char < n_out:
        # Smoothing initializes the count of each possible 
        # next character to the value inputted for smooth

        prob_array = [smooth] * len(cti)
        # Iterate through each key (n-gram) in dictionary
        for key in table.keys():
            # If we find an n-gram key whose first n-1 characters 
            # are the same as current, add smooth + the count
            # of that n-gram in the text to the probability
            # array for the last character in that n-gram
            if current == key[:n-1]:
                prob_array[cti[key[-1]]] += table[key]

        # Normalize the probability array; then choose the next
        # character from the probability distribution
        prob_array = normalize(prob_array)
        letter = itc[numpy.random.choice(len(cti), p=prob_array)]
        
        # Append to the output and slide the window one over
        output_text += letter
        current = current[1:] + letter
        char += 1

    print char
    print output_text

    return

# Normalize a probability array so we can sample from it
def normalize(array):
    theSum = 1.0*sum(array)
    for i in range(len(array)):
        array[i] /= theSum
    return array

# The text to train on
text = process_text("training/sherlock-all-mod.txt")

# Taken from generate-generic.py to get a character-integer mapping
# and vice versa
chars = sorted(list(set(text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

# Smoothing: initial count is
# k for every possible next character.
smooth = .0001
# Number of characters to generate
num_chars = 1000
# Generate the n+1st character from n
n = 14
t_prob = transition(text, n+1)
generate(t_prob, num_chars, n+1, char_to_int, int_to_char, smooth)



