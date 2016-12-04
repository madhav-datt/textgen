# Larger LSTM Network to Generate Text for Alice in Wonderland
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import sys, getopt

text = ''
layers = 1
nodes = 256
wfile = ''
epochs = 20
batchsize = 64
dropout = 0.2

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"t:w:l:n:e:b:d:")
    except getopt.GetoptError:
        print "error"
        sys.exit(2)
    for opt, arg in opts:
        print opt, arg
        if opt in ("-t"):
            global text
            text = arg
        elif opt in ("-l"):
            global layers
            layers = int(arg)
        elif opt in ("-n"):
            global nodes
            nodes = int(arg)
        elif opt in ("-w"):
            global wfile
            wfile = arg
        elif opt in ("-e"):
            global epochs
            epochs = int(arg)
        elif opt in ("-b"):
            global batchsize
            batchsize = int(arg)
        elif opt in ("-d"):
            global dropout
            dropout = float(arg)

if __name__ == "__main__":
   main(sys.argv[1:])

print text
print wfile
print layers
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
print "Total Characters: ", n_chars
print "Total Vocab: ", n_vocab
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
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
    model.add(LSTM(256))
    model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0))
#model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.05, momentum=0.0, decay=0.0, nesterov=False))
# define the checkpoint
filepath="weights/' + wfile + '-{epoch:02d}.hdf5"
#filepath="weights/multilayer-weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, nb_epoch=int(epochs), batch_size=int(batchsize), callbacks=callbacks_list)
