# Changing d2v with lstm language model
#from __future__ import print_function

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file

import numpy as np
import random
import sys
import os    
os.environ['THEANO_FLAGS'] = "device=gpu, floatX=float32"
import gensim
from functions.words_chars import vocabulary_from_json_corpus

json_corpus_path = "/home/ubuntu/summarization_query_oriented/data/wikipedia/json/td_qfs_rank_1/"
# Import 
import gensim
import keras
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import pyrouge
from pyrouge import Rouge155

from hard_coded import data_json_dir, data_txt_dir, lang_model_dir, model_dir, nn_summarizers_dir, summary_system_super_dir, tdqfs_folder
from hard_coded import non_selected_keys,tdqfs_themes
from functions.training_functions import create_triplets_lstm
from functions.training_functions import *
stopwords = stop_words()

# paths to folder 
data_json = data_json_dir
data_txt = data_txt_dir
lang_model_folder = lang_model_dir
nn_summarizers_folder = nn_summarizers_dir
summary_system_super_folder = summary_system_super_dir
themes = tdqfs_themes

#title_file = "/home/ubuntu/summarization_query_oriented/data/DUC/duc2005_topics.sgml"
#titles_folder = "/home/ubuntu/summarization_query_oriented/data/DUC/duc2005_docs/"


# training parameters

patience_limit = 25

## loading a lstm (to be a shifted LSTM next ...)

# building vocabulary of the corpus
words = vocabulary_from_json_corpus(json_corpus_path)
word_indices = dict((c, i) for i, c in enumerate(words))
indices_word = dict((i, c) for i, c in enumerate(words))
print("word_indices", type(word_indices), "length:",len(word_indices))
print("indices_words", type(indices_word), "length", len(indices_word))

maxlen = 10

#defining the lstm model
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
model_folder = "/home/ubuntu/summarization_query_oriented/nn_models/language_models/RNN/wider_18112016/"
model_name = "tdqfs_lstm_wider_corpus_last.hdf5"

if os.path.isfile('/home/ubuntu/summarization_query_oriented/nn_models/language_models/RNN/wider_18112016/tdqfs_lstm_wider_corpus_last.hdf5'):
    model.load_weights('/home/ubuntu/summarization_query_oriented/nn_models/language_models/RNN/wider_18112016/tdqfs_lstm_wider_corpus_last.hdf5')


print('Build model...')
model2 = Sequential()
model2.add(LSTM(400, return_sequences=True, batch_input_shape=(1,maxlen, len(word_indices)),weights=model.layers[0].get_weights(),stateful=True))
model2.add(LSTM(400, return_sequences=False,weights=model.layers[2].get_weights(),stateful=True))
print('Model built...')

## get wikipedia data
article_names, article_weights = relevant_articles(data_json)

## design a fully connected model

fc_model = Sequential()

fc_model.add(Dense(160, input_dim=1600))
fc_model.add(Activation('sigmoid'))
fc_model.add(Dropout(0.5))

fc_model.add(Dense(16))
fc_model.add(Activation('sigmoid'))
fc_model.add(Dropout(0.5))

fc_model.add(Dense(1))
fc_model.add(Activation('sigmoid'))

# compiling the model
fc_model.compile(loss="binary_crossentropy", optimizer='adam')

# training per batch
batch_per_epoch = 1
#batch_per_epoch = 20
batch_size = 32
patience = 0
batch_counter = 8000
rouge_su4_recall_max = 0
rouge_2_recall_max = 0

while patience < patience_limit :
    # train on several batchs
    
    for i in range(batch_per_epoch):
        tic = time.time()
        triplets, labels = create_triplets_lstm(model2, article_names, article_weights, stopwords, word_indices, nb_triplets=batch_size, triplets_per_file=16, neg_ratio=1, str_mode = False, with_txt_vect=True) # HERE IS THE NEW STUFF
        fc_model.train_on_batch(triplets, labels)
        toc = time.time()
        print("batch  "+str(batch_per_epoch),toc-tic)
    batch_counter += 1

    # summarize DUC
    str_time = time.strftime("%Y_%m_%d")
    fc_model_name = "T6_2_"+str_time+"_fc_model_batch_"+str(batch_counter)+"k"
    system_folder = summary_system_super_folder+fc_model_name+"/"
    os.mkdir(system_folder)

    for theme in themes :
        theme_folder = tdqfs_folder + theme + "/"
        theme_doc_folder = theme_folder + theme + "/"
        queries = get_queries(theme_folder+"queries.txt")
        text = merge_articles_tqdfs(theme_doc_folder)
        for i in range(len(queries)):
            query = queries[i]
            summary = lstm_summarize(text,query,model2, fc_model,stopwords, word_indices, limit = 250, with_txt_vect=True)
            summary = " ".join(summary.split()[:250])

            summary_name = theme + "." + str(i+1) + ".txt"
            with open(system_folder + summary_name,'w') as f :
                f.write(summary.decode('ascii',"ignore").encode("utf8", "replace"))
                print 'writing in '+ system_folder + summary_name

    r = Rouge155()
    r.system_dir = system_folder
    r.model_dir = model_dir
    r.model_filename_pattern = '#ID#.u[0-9]q[0-9].txt'
    r.system_filename_pattern = '([a-z]+.[0-9]+).txt'            

    options = "-e " + r._data_dir + " -n 4 -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -a -x"

    output = r.convert_and_evaluate(rouge_args=options)
    output_dict = r.output_to_dict(output)

    rouge_2_recall = np.round( output_dict["rouge_2_recall"], 5 )
    rouge_su4_recall = np.round( output_dict["rouge_su4_recall"], 5 )

    print 50*'$'
    print "rouge_2", rouge_2_recall
    print "rouge_SU4", rouge_su4_recall
    print 50*'$'

    # save rouge results
    output_dict = r.output_to_dict(output)
    with open(system_folder+"ROUGE_RESULTS.json",'w') as f :
        json.dump(output_dict,f)
    #check if the model has improved 
    if rouge_2_recall > rouge_2_recall_max or rouge_su4_recall > rouge_su4_recall_max :
        patience = 0
        if rouge_2_recall > rouge_2_recall_max :
            rouge_2_recall_max = rouge_2_recall
        if rouge_su4_recall > rouge_su4_recall_max :
            rouge_su4_recall_max = rouge_su4_recall    
            #save this new model
        fc_model_name ="fc_model_batch_"+str(batch_counter)+"k_R2_"+str(rouge_2_recall)+"_SU4_"+str(rouge_su4_recall)
        fc_model.save(nn_summarizers_folder + fc_model_name + ".hdf5")  # creates a HDF5 file 'my_model.h5'
        print fc_model_name, "is saved"
    else :
        patience = patience + 1
        print "patience :", patience
    
    
print('early stopped')