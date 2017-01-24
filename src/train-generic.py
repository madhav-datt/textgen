#
# Train a LSTM-RNN to generate text in a certain style. Uses the Keras library with Theano backend; 
# code adapted from example at: http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
# Keras documentation here: https://keras.io/
#
# Takes command line arguments:
# 
# -t textfile
# training/[textfile].txt is the training corpus; mandatory argument
#
# -w weightfile
# weights/[weightfile]-[epochnumber].hdf5 will be the generated weight file for eopch epochnumber; mandatory argument
#
# -l layers
# The number of hidden layers in the LSTM-RNN
#
# -n nodenums
# If 1 layers, nodenums is the number of nodes in that layer
# If >1 layers, nodenums is the number of nodes in each layer, separated by '.', for example 512.256 for 2 layers with 512/256 nodes
#
# -e epochs
# Max number of epochs to run for
#
# -b batchsize
# How often to train the network/adjust weights/propagate errors: every batchsize samples the network trains
#
# -d dropout
# Drops a fraction dropout of the input units to 0 while training, avoiding overfitting
#
# -s slide
# Size of the sliding window of text used to predict the next character; number of nodes in the input layer
#   
# All parameters but -t and -w have default values set.
# 

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import sys, getopt

# Default parameters; only text/wfile are required

text = ''
layers = 1
nodes = "512"
wfile = ''
epochs = 1000
batchsize = 64
dropout = 0.2
slide = 100

def main(argv):
    hasText = False
    hasWfile = False
    try:
        opts, args = getopt.getopt(argv,"t:w:l:n:e:b:d:s:")
    except getopt.GetoptError:
        print "train-generic.py -t [text file] -w [weight file] -l [num layers] -n [node numbers per layer, separated with '.'] -e [epochs] -b [batch size] -d [dropout rate] -s [size of window]"
        sys.exit(2)
    for opt, arg in opts:
        print opt, arg
        if opt in ("-t"):
            global text
            text = arg
            hasText = True
        elif opt in ("-l"):
            global layers
            layers = int(arg)
        elif opt in ("-n"):
            global nodes
            nodes = arg
        elif opt in ("-w"):
            global wfile
            wfile = arg
            hasWfile = True
        elif opt in ("-e"):
            global epochs
            epochs = int(arg)
        elif opt in ("-b"):
            global batchsize
            batchsize = int(arg)
        elif opt in ("-d"):
            global dropout
            dropout = float(arg)
        elif opt in ("-s"):
            global slide
            slide = int(arg)
    if not hasText or not hasWfile:
        print "-t and -w are required parameters"
        sys.exit(2)

if __name__ == "__main__":
   main(sys.argv[1:])

nodes = nodes.split('.')
assert len(nodes) == layers

# load ascii text and covert to lowercase
filename = "training/" + text + ".txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
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
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()

if int(layers) == 1:
    # Just the one layer
    model.add(LSTM(int(nodes[0]), input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(dropout))
else:
    # Multiple layers, each with the number of nodes specified
    model.add(LSTM(int(nodes[0]), input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(dropout))
    for i in range(layers-2):
        model.add(LSTM(int(nodes[i+1]), return_sequences=True))
        model.add(Dropout(dropout))
    model.add(LSTM(int(nodes[layers-1])))
    model.add(Dropout(dropout))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0))

# define the checkpoint
filepath="weights/" + wfile + "-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, nb_epoch=epochs, batch_size=batchsize, callbacks=callbacks_list)
