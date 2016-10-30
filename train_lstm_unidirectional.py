from __future__ import print_function

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file

import numpy as np
import random
import sys
import os

import gensim

path = "/home/ubuntu/summarization_query_oriented/data/wikipedia/txt/td_qfs.txt"

try: 
    text = open(path).read().lower()
except UnicodeDecodeError:
    import codecs
    text = codecs.open(path, encoding='utf-8').read().lower()

print('corpus length:', len(text))

word_list = gensim.utils.simple_preprocess(text)
text = " ".join(word for word in word_list)

chars = set(text)
words = set(text.split())

print("chars:",type(chars))
print("words",type(words))
print("total number of unique words", len(words))
print("total number of unique chars", len(chars))

word_indices = dict((c, i) for i, c in enumerate(words))
indices_word = dict((i, c) for i, c in enumerate(words))

print("word_indices", type(word_indices), "length:",len(word_indices) )

print("indices_words", type(indices_word), "length", len(indices_word))

batch_size = 128

maxlen = 10
step = 3
print("maxlen:",maxlen,"step:", step)
sentences = []
next_words = []
list_words = []
sentences2=[]
list_words=text.lower().split()

for i in range(0,len(list_words)-maxlen, step):
    sentences2 = ' '.join(list_words[i: i + maxlen])
    sentences.append(sentences2)
    next_words.append((list_words[i + maxlen]))
    
print('nb sequences(length of sentences):', len(sentences))
print("length of next_word",len(next_words))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(words)), dtype=np.bool)
y = np.zeros((len(sentences), len(words)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence.split()):
        #print(i,t,word)
        X[i, t, word_indices[word]] = 1
    y[i, word_indices[next_words[i]]] = 1

    
#build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(400, return_sequences=True, input_shape=(maxlen, len(words))))
model.add(Dropout(0.6))
model.add(LSTM(400, return_sequences=False))
model.add(Dropout(0.6))
model.add(Dense(len(words)))
#model.add(Dense(1000))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

#resizing

nb_batch = X.shape[0] / batch_size
nb_batch_val = np.floor(0.2*nb_batch)
nb_batch_train = nb_batch - nb_batch_val

nb_sample_val = batch_size*nb_batch_val
nb_sample_train = batch_size*nb_batch_train

X_train = X[:nb_sample_train]
y_train = y[:nb_sample_train]

X_val = X[nb_sample_train:nb_sample_train+nb_sample_val]
y_val = y[nb_sample_train:nb_sample_train+nb_sample_val]

from keras.callbacks import ModelCheckpoint, EarlyStopping
from functions.callbacks import LossHistory, ReduceLROnPlateau
model_folder = "/home/ubuntu/summarization_query_oriented/nn_models/language_models/RNN/"
model_name = "tdqfs_lstm_"
model_name = model_name +"epoch_{epoch:02d}_valloss_{val_loss:.2f}.hdf5"
example_name = "tdqfs_lstm.examples"
history = LossHistory()
checkpointer = ModelCheckpoint(filepath=model_folder+model_name, verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=3, min_lr=0.001)

if os.path.isfile('/home/ubuntu/summarization_query_oriented/nn_models/language_models/RNN/last_tdqfs_lstm'):
    model.load_weights('/home/ubuntu/summarization_query_oriented/nn_models/language_models/RNN/last_tdqfs_lstm')

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

# train the model, output generated text after each iteration
for iteration in range(1, 20):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    with open(model_folder+example_name,"a") as fex :
        fex.write('Iteration ' + str(iteration) + "\n\n")
    fex.close()
    model_name = "tdqds_lstm_"
    model_name = model_name +"epoch_{epoch:02d}_valloss_{val_loss:.2f}_iteration_" + str(iteration) +".hdf5"

    checkpointer = ModelCheckpoint(filepath=model_folder+model_name, verbose=1, save_best_only=True)
    model.fit(X_train, y_train, validation_data=(X_val,y_val),callbacks=[history,checkpointer, earlystopper,reduce_lr], batch_size=batch_size, nb_epoch=10)
    model.save_weights('/home/ubuntu/summarization_query_oriented/nn_models/language_models/RNN/last_tdqfs_lstm',overwrite=True)

    for diversity in range(1,6):
        start_index = random.randint(0, len(list_words) - maxlen - 1)

        print()
        print('----- Example:', diversity)
        generated = ''
        sentence = list_words[start_index: start_index + maxlen]
        generated += ' '.join(sentence)
        print('----- Generating with seed: "' , sentence , '"')
        print()
        sys.stdout.write(generated)
        print()

        for i in range(15):
            x = np.zeros((1, maxlen, len(words)))
            for t, word in enumerate(sentence):
                x[0, t, word_indices[word]] = 1.

            preds = model.predict(x, verbose=0)[0]

            next_index = np.argmax(preds)
            next_word = indices_word[next_index]
            generated += " " + next_word
            del sentence[0]
            sentence.append(next_word)
            sys.stdout.write(' ')
            sys.stdout.write(next_word)
            sys.stdout.flush()
        with open(model_folder+example_name,"a") as fex :
            fex.write('Example ' + str(diversity) + " : ")
            fex.write(generated + "\n")
        fex.close()
        
    with open(model_folder+example_name,"a") as fex :
        fex.write("\n\n")
    fex.close()