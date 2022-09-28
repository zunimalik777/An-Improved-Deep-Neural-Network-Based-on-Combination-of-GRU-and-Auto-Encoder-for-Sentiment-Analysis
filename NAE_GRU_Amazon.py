import os
import codecs
import re
import tensorflow
import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from keras.layers import Dense, Dropout, Activation, Merge
from keras.layers import Embedding, Reshape
from keras.layers import autoencoder, LSTM, GRU, NAE-GRU
from keras.callbacks import Model Checkpoint
from keras import optimizers
from keras.layers import Concatenate, Permute, Dot, Input, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import h5py
from sklearn.model_selection import StratifiedShuffleSplit

# Datasets: need to random split the training and testing sets
'''
X.shape = (m, seq_length)
Y.shape = (m, label)
'''
# Initialization
X = []
Y = None
X_train = None
X_test = None
Y_train = None
Y_test = None
X_val = None
Y_val = None
MAX_LENGTH = 160

filename = C:'/Users/ZULQARNAIN/SentimentAnalysis/Corpus/amazon'
 
raw_text = open(filename). read()
raw_text = raw_text.lower()
# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
zz=[[char_to_int[char] for char in raw_text[3320:3520]]]
len(zz)
x_train=np.array(zz)
x_train = x_train.astype('float32')
x_train=x_train/float(n_vocab)   

encoding_dim = 3
input_img = Input(shape=(200,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(200, activation='sigmoid')(encoded)
autoencoder = Model(input=input_img, output=decoded)
encoder = Model(input=input_img, output=encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

filepath="autoencoder-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='binary_crossentropy', verbose=1, save_best_only=False)

autoencoder.summary()
autoencoder.fit(x_train, x_train,
                nb_epoch=100,
                batch_size=32,
                shuffle=True,
                validation_data=(x_train, x_train),callbacks=[checkpoint],verbose=0)

autoencoder.load_weights(filename)
autoencoder.compile(loss='mean_squared_error', optimizer='adam')

def __shuffleSplit__(X, Y):
    global X_train, X_test
    s = StratifiedShuffleSplit(n_splits=1, test_size=0.25, train_size=0.75, random_state=0)
    print (s)
    print ('S.split:',s.split)
    for train_index, test_index in s.split(X, Y):
        print('Training set:', train_index, 'Testing set:', test_index)
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

def __randomSplit__(X, Y):
    X = np.array(X)
    shuffle_index = np.random.permutation(np.arange(len(Y)))
    X_new = X[shuffle_index]
    Y_new = Y[shuffle_index]
    X_train = X_new[0:8259]
    Y_train = Y_new[0:8259]
    X_val = X_new[8260:9595]
    Y_val = Y_new[8260:9595]
    X_test = X_new[9596:10661]
    Y_test = Y_new[9595:10661]
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def __findMax__(filename):
    MAX_LENGTH = 0
    fopen = codecs.open(filename, 'r+', 'utf-8', errors='ignore')
    for eachLine in fopen.readlines():
        tmp = eachLine.strip(' ')
        counts = len(tmp)
        if counts >= MAX_LENGTH:
            MAX_LENGTH = counts
    print('The max length is:', MAX_LENGTH)
    fopen.close()

# Processing dataset
__preProduceFile__(filename)

# Spliting the dataset into train and test
print(len(X))
print(len(Y))
X_train, Y_train, X_val, Y_val, X_test, Y_test = __randomSplit__(X, Y)

# Find max length
# __findMax__(filename)

# Testing initialization
print('X.shape:', len(X))
print('Y.shape:', len(Y))
print('X[100]:', X[100])
print('Y[100]:', Y[100])
print('X tarin:', X_train[7000])
print('Y train:', Y_train[7000])
print('X test:', X_test[0])
print('Y test:', Y_test[0])

# Parameters
'''
When changing the embedding file:
remember to change the EMBEDDING_DIM and filename in fopen
'''
EMBEDDING_DIM = 240
MAX_NUM_WORDS = 220000

'''
embeddig_matrix: pre-trained word vectors
word_index: dictionary of word vectors
Concat: concatenate training and testing sets, in order to build the dictionary of embedding vectors
sequences: parsing results of training and testing sets
data: training sets exchange to embeding vectors matrix
'''


for eachLine in fopen.readlines():
    # First element in each line is the word
    values = eachLine.split()
    word = values[0]
    # Word vectors
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_index[word] = coefs
fopen.close()
print('Found %s word vectors.' % len(embedding_index))



# Vectoize the TRAINING text samples into a 2D integer tensor
X_train = X_train.tolist()
X_test = X_test.tolist()
X_val = X_val.tolist()
Concat = X_train + X_test + X_val
tokenizer = Tokenizer(num_words = MAX_NUM_WORDS)
tokenizer.fit_on_texts(Concat)
sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)
sequences_val = tokenizer.texts_to_sequences(X_val)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# Auto filled with 0
data_train = pad_sequences(sequences_train, maxlen = MAX_LENGTH)
data_test = pad_sequences(sequences_test, maxlen = MAX_LENGTH)
data_val = pad_sequences(sequences_val, maxlen = MAX_LENGTH)

# Prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index)+1) 
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all zeros
        embedding_matrix[i] = embedding_vector

# Load pre-trained word embeddings into an Embedding Layer
embedding_layer = Embedding(num_words, 
                            EMBEDDING_DIM, 
                            weights=[embedding_matrix], 
                            input_length=MAX_LENGTH,
                            trainable=False)
print('Trainning embedding model.')

# Testing embedding
print(embedding_matrix[100])
print(word_index.get('any'))
print(data_train[100])
print(data_test[100])


# Attention
'''
If SINGLE_ATTENTION_VECTOR = true, 
the attention vector is shared across the input_dimensions where the attention is applied.
'''
from keras.layers import multiply

SINGLE_ATTENTION_VECTOR = False
TIME_STEPS = MAX_LENGTH

def __attention3DBlock__(inputs):
    '''
    Input shape = (batch_size, time_steps, input_dim)
    '''
    input_dim = int(inputs.shape[2])
    a = Permute((2,1))(inputs)  # (batch_size, input_dim, time_steps, batch_normalization)
    a = Reshape((input_dim, TIME_STEPS))(a) # (batch_size, input_dim, time_steps, batch_normalization)
    a = Dense(TIME_STEPS, activation='softmax')(a) # (batch_size, input_dim, time_steps, batch_normalization)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a) # (batch_size, input_dim, time_steps, batch_normalization)
    a_probs = Permute((2,1), name='attention_vector')(a) # (batch_size, time_steps, input_dim, batch_normalization)
    output_attention_mul = multiply([inputs, a_probs], name='attenton_mul') # (batch_size, time_steps, input_dim, batch_normalization)

    return output_attention_mul


# Model
from keras import regularizers
import self_attention

# Parameters
'''
n_x: hidden state size of GRU, LSTM, NAE-GRU
regularization: L2 normalization
optimization: AdaGrad/Adam
time_steps = MAX_LENGTH = 120`
'''
batch_size = 32
momentum = 0.9
l2_regularization = 0.001
learning_rate = 0.001
n_x = 32   
epochs = 30
time_steps = MAX_LENGTH

# Build model
print ("Build model...")
sequence_input = Input(shape=(time_steps,), dtype='float32')
print('Sequence input is:', sequence_input) 
embedded_sequences = embedding_layer(sequence_input) 
print('Embedding layer is:', embedded_sequences) 

    param_grid = dict(num_filters=[32, 64, 128],
                      kernel_size=[3, 4, 5],
                      vocab_size=[vocab_size],
                      embedding_dim=[embedding_dim],
                      maxlen=[maxlen])
    model = KerasClassifier(build_fn=create_model,
                            epochs=epochs, batch_size=10,
                            verbose=False)
    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                              cv=4, verbose=1, n_iter=5)

L = (GRU(n_x,
            activation='tanh', 
            dropout=0.2, 
            recurrent_dropout=0.2, 
            return_sequences=True,
            kernel_initializer='he_uniform',
            name='GRU'))(embedded_sequences)
print('GRU is:', L) # (batch_size, time_steps, units=32*2)


L = GRU(n_x, 
        activation='tanh', 
        kernel_regularizer=regularizers.l2(0.001))(L) 
          activation='softmax', 
          kernel_regularizer=regularizers.l2(0.001))(L)
print('Dense layer is:', L)

model = Model(inputs=sequence_input, outputs=L)

# Optimization and compile
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.01)
print('Begin compiling...')
model.compile(loss='categorical_crossentropy', 
              optimizer=opt, 
              metrics=['accuracy'])
model.summary()

L = (LSTM(n_x,
            activation='tanh', 
            dropout=0.2, 
            recurrent_dropout=0.2, 
            return_sequences=True,
            kernel_initializer='he_uniform',
            name='LSTM'))(embedded_sequences)
print('LSTM is:', L) # (batch_size, time_steps, units=32*2)


L = LSTM(n_x, 
        activation='tanh', 
        kernel_regularizer=regularizers.l2(0.001))(L) 
          activation='softmax', 
          kernel_regularizer=regularizers.l2(0.001))(L)
print('Dense layer is:', L)

model = Model(inputs=sequence_input, outputs=L)

# Optimization and compile
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.01)
print('Begin compiling...')
model.compile(loss='categorical_crossentropy', 
              optimizer=opt, 
              metrics=['accuracy'])
model.summary()

L = (NAE_GRU(n_x,
            activation='relu', 
            dropout=0.2, 
            recurrent_dropout=0.2, 
            return_sequences=True,
            kernel_initializer='he_uniform',
            name='GRU'))(embedded_sequences)
print('NAE_GRU is:', L) # (batch_size, time_steps, units=32*2, batch_normalization)


L = NAE_GRU(n_x, 
        activation='relu', 
        kernel_regularizer=regularizers.l2(0.001))(L) 
          activation='softmax', 
          kernel_regularizer=regularizers.l2(0.001))(L)
print('Dense layer is:', L)

model = Model(inputs=sequence_input, outputs=L)

# Optimization and compile
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.01)
print('Begin compiling...')
model.compile(loss='categorical_crossentropy', 
              optimizer=opt, 
              metrics=['accuracy'])
model.summary()

# Begin training
model.fit(data_train, 
          Y_train, 
          batch_size=batch_size, 
          epochs=epochs, 
          verbose=2,
          validation_data=(data_val, Y_val))
score = model.evaluate(data_test, Y_test, batch_size=batch_size)
print ('The evaluation is: ', score)

# Evaluate testing set
test_accuracy = grid.score(X_test, y_test)


# Save model
print ('Saving model...')
model.save('autoencoder','GRU','LSTM','CNN-GRU-amazon')
