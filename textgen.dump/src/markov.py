import numpy
import random

def process_text(filename):

    raw_text = open(filename).read()
    raw_text = raw_text.lower()
    return raw_text

def transition(raw_text, n):

    table = {}
    # Count every 10-character substring of raw_text
    for i in range(0, len(raw_text)):
        substring = raw_text[i: i + n]

        if substring in table:
            table[substring] += 1
        # Ignore edge case where the substring is not of length 10
        elif len(substring) == n:
            table[substring] = 1


    #print table
    return table

def generate(table, n_out, n, cti, itc, smooth):
    
    #word = random.choice(table.keys())
    #output_text = word
    #print word
    seed = "to sherlock holmes she is always the woman."
    output_text = seed[:n]
    current = seed[1:n]
    #current = word[1:]
    print output_text

    char = 0

    while char < n_out:
        prob_array = [1] * len(cti)
        # Iterate through each key in dictionary
        for key in table.keys():
            if current == key[:n-1]:
                prob_array[cti[key[-1]]] += smooth*table[key]
        #print current

        prob_array = normalize(prob_array)
        letter = itc[numpy.random.choice(len(cti), p=prob_array)]
        
        output_text += letter
        current = current[1:] + letter
        #letter = 't'
        #max_count = 0.1
        char += 1

    print char
    print output_text

    return

def normalize(array):
    theSum = 1.0*sum(array)
    for i in range(len(array)):
        array[i] /= theSum
    return array
text = process_text("training/4-mod.txt")
chars = sorted(list(set(text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

smooth = 1000
num_chars = 1000
n = 3
t_prob = transition(text, n)
generate(t_prob, num_chars, n, char_to_int, int_to_char, smooth)



