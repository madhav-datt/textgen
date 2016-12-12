# 
# Explain parameters
#
#

import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import getopt

# Default parameters; only text/wfile/output are required

text = ''
layers = 1
nodes = 512
wfile = ''
dropout = 0.2
slide = 100
numchars = 1000
output = ''

def main(argv):
    hasText = False
    hasWfile = False
    hasOutput = False
    try:
        opts, args = getopt.getopt(argv,"t:w:l:n:d:s:c:o:")
    except getopt.GetoptError:
        print "generate-generic.py -t [text file] -w [weight file] -o [output file] -l [num layers] -n [node numbers per layer, separated with '.'] -d [dropout rate] -s [size of window] -c [num characters to generate]"
        sys.exit(2)
    for opt, arg in opts:
        print opt, arg
        if opt in ("-t"):
            global text
            hasText = True
            text = arg
        elif opt in ("-l"):
            global layers
            layers = int(arg)
        elif opt in ("-n"):
            global nodes
            nodes = int(arg)
        elif opt in ("-w"):
            global wfile
            hasWfile = True
            wfile = arg
        elif opt in ("-d"):
            global dropout
            dropout = float(arg)
        elif opt in ("-s"):
            global slide
            slide = int(arg)
        elif opt in ("-c"):
            global numchars
            numchars = int(arg)
        elif opt in ("-o"):
            global output
            hasOutput = True
            output = arg
    if not hasText or not hasWfile or not hasOutput:
        print "-t, -w, and -o are required parameters"
        sys.exit(2)


if __name__ == "__main__":
   main(sys.argv[1:])

# load ascii text and covert to lowercase
filename = "training/" + text + ".txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
# create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print "Total Characters: ", n_chars
print "Total Vocab: ", n_vocab
# prepare the dataset of input to output pairs encoded as integers
seq_length = slide
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print "Total Patterns: ", n_patterns
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()
if int(layers) == 1:
    model.add(LSTM(int(nodes), input_shape=(X.shape[1], X.shape[2])))#, return_sequences=True))
    model.add(Dropout(float(dropout)))
else:
    model.add(LSTM(int(nodes), input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(float(dropout)))
    for i in range(int(layers)-2):
        model.add(LSTM(int(nodes), return_sequences=True))
        model.add(Dropout(float(dropout)))
    model.add(LSTM(int(nodes)))
    model.add(Dropout(float(dropout)))
model.add(Dense(y.shape[1], activation='softmax'))

# load the network weights
filename = "weights/" + wfile + ".hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0))

# Seed with 100 characters from "A Scandal in Bohemia" (training/1.txt)
seed = "to sherlock holmes she is always the woman. i have seldom heard him mention her under any other name"
pattern = []

# Convert the seed to our initial integer input to the network
for i in range(len(seed)):
    pattern = pattern + [char_to_int[seed[i]]]

with open("output/" + output, 'w') as output_file:
    for i in range(numchars):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        output_file.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
print "\nDone."