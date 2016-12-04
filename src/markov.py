
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


    print table
    return table


def generate(table, n_out, n):
    word = random.choice(table.keys())
    output_text = word
    print word
    current = word[1:]


    # could change this to choose randomly from all characters
    letter = 't'
    max_count = 0.1
    char = 0

    while char < n_out:

        # Iterate through each key in dictionary
        for key in table.keys():
            if current == key[:n-1]:
                # print "yes"
                # Keep track of most probable next character
                if table[key] > max_count:
                    max_count = table[key]
                    letter = key[-1]

        output_text += letter
        current = current[1:] + letter
        letter = 't'
        max_count = 0.1
        char += 1

    print char
    print output_text

    return


t_prob = transition(process_text("training/4-mod.txt"), 2)
generate(t_prob,1000, 2)



