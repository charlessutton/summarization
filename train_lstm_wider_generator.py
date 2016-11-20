from __future__ import print_function

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file

import numpy as np
import pickle
import random
import sys
import os    
os.environ['THEANO_FLAGS'] = "device=gpu, floatX=float32"
import gensim
from functions.words_chars import vocabulary_from_json_corpus

json_corpus_path = "/home/ubuntu/summarization_query_oriented/data/wikipedia/json/td_qfs_rank_1/"

# building vocabulary of the corpus
words = vocabulary_from_json_corpus(json_corpus_path)
word_indices = dict((c, i) for i, c in enumerate(words))
indices_word = dict((i, c) for i, c in enumerate(words))
print("word_indices", type(word_indices), "length:",len(word_indices))
print("indices_words", type(indices_word), "length", len(indices_word))

# generator
from functions.generators import lang_model_batch_generator

maxlen = 10

batch_nb = 1000
batch_size = 64

train_generator = lang_model_batch_generator(json_corpus_path, word_indices, maxlen=maxlen, 
                                             batch_nb = batch_nb, batch_size = batch_size, val_mode = False)

batch_nb_val = 2000
batch_size_val = 10

val_generator = lang_model_batch_generator(json_corpus_path, word_indices, maxlen=maxlen, 
                                             batch_nb = batch_nb_val, batch_size = batch_size_val, val_mode = True)

#defining model
print('Build model...')
model = Sequential()
model.add(LSTM(400, return_sequences=True, input_shape=(maxlen, len(word_indices))))
model.add(Dropout(0.6))
model.add(LSTM(400, return_sequences=False))
model.add(Dropout(0.6))
model.add(Dense(len(words)))
#model.add(Dense(1000))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
print('Model built...')

# naming 
model_folder = "/home/ubuntu/summarization_query_oriented/nn_models/language_models/RNN/wider_30102016/"
model_name = "tdqfs_lstm_wider_corpus_last.hdf5"
#model_name = model_name +"epoch_{epoch:02d}_valloss_{val_loss:.2f}.hdf5"
#example_name = "tdqfs_lstm_wider_corpus.examples"

from keras.callbacks import ModelCheckpoint, EarlyStopping
from functions.callbacks import LossHistory, ReduceLROnPlateau

history = LossHistory()
checkpointer = ModelCheckpoint(monitor='val_loss', filepath=model_folder+model_name, verbose=1, save_best_only=True, mode='auto')
earlystopper = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=2, min_lr=0.001)

if os.path.isfile('/home/ubuntu/summarization_query_oriented/nn_models/language_models/RNN/wider_30102016/tdqfs_lstm_wider_corpus_last.hdf5'):
    model.load_weights('/home/ubuntu/summarization_query_oriented/nn_models/language_models/RNN/wider_30102016/tdqfs_lstm_wider_corpus_last.hdf5')

model.fit_generator(generator = train_generator, samples_per_epoch = batch_nb * batch_size, nb_epoch = 1000,
                    verbose=1, callbacks=[history, checkpointer, earlystopper, reduce_lr], 
                    validation_data=val_generator, nb_val_samples=batch_nb_val * batch_size_val,
                    nb_worker=1, pickle_safe=False)

with open(model_folder+'history/dump', 'w') as dumpfile:
    pickle.dump([history.losses, history.val_losses], dumpfile)
