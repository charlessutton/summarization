#### Imports
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
from glove import Glove
import gensim
import numpy as np
from functions.generators import bidirectional_batch_generator
from functions.callbacks import LossHistory, ReduceLROnPlateau

#### Parameters
# paths
word_embedder_folder = "/home/ubuntu/summarization_query_oriented/nn_models/word_vectorization_models/glove/"
word_embedder = "rank1_alltxt_size_400_epoch_30.glove"

# path to the text we use for training, take the corpus or sub corpus of the one you have trained glove 
txt_file = "/home/ubuntu/summarization_query_oriented/data/wikipedia/txt/td_qfs_rank_1_all.txt"

# model folder
model_folder = "/home/ubuntu/summarization_query_oriented/nn_models/language_models/RNN/"

# size
train_data_size = 522
val_data_size = 522
batch_size = 256
seq_length = 10

#### Implementation

# glove model
glove = Glove.load(word_embedder_folder+word_embedder)
corpus = glove.dictionary
corpus_size = len(corpus.keys())

# dictionnaries
word_to_int = corpus
int_to_word = dict((corpus[key], key) for key in corpus.keys())

# generator instance (this perform bidirectionnal batchs)
train_gen = bidirectional_batch_generator(txt_file, data_size = train_data_size,glove_model = glove,  batch_size = batch_size, seq_length=10, verbose=True)

val_gen  =  bidirectional_batch_generator(txt_file, data_size=val_data_size ,glove_model = glove,  batch_size = batch_size, seq_length=10, verbose=True, starting_idx=train_data_size)

# define the LSTM model
model = Sequential()
model.add(LSTM(400, input_shape=(seq_length, glove.no_components), return_sequences=False, name="lstm_1"))
model.add(Dropout(0.5))
model.add(Dense(corpus_size, activation='softmax', name="dense_1"))
model.compile(loss='categorical_crossentropy', optimizer='adagrad')

# Implementing callbacks

model_name = "bidirectional_LSTM_glove_"
model_name = model_name +"epoch_{epoch:02d}_valloss_{val_loss:.2f}.hdf5"

history = LossHistory()
checkpointer = ModelCheckpoint(filepath=model_folder+model_name, verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=3, min_lr=0.001)

model.fit_generator(train_gen, samples_per_epoch = 2*(train_data_size-seqlen), nb_epoch = 1000, verbose = 1)
compteur = 0
# training command
while compteur < 2*train_data_size:
    compteur+=1
    x,y = train_gen.next()
    model.train_on_batch(x,y)

    



#callbacks = [history, checkpointer, earlystopper, reduce_lr], validation_data = val_gen, nb_val_samples = 10000,  pickle_safe=True)
                    #class_weight=None, max_q_size=10, nb_worker=1, pickle_safe=True)

# saving the model

