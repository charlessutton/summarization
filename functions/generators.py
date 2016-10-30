import numpy as np
import gensim
from functions.training_functions import relevant_articles
import json
from hard_coded import non_selected_keys
import random

def one_hot_vector(word ,corpus):
    """
    given a corpus of (key,value) = (word,index) , this function return a one hot vector corresponding the input word
    """
    size = len(corpus.values())
    y = np.zeros(size)
    y[corpus[word]] = 1.0
    return y


def bidirectional_batch_generator(txt_path, data_size, glove_model, batch_size = 128, seq_length=10, verbose=True, starting_idx = 0):
    """
    Generator of bidirectional batchs to train on a regular LSTM model
    It yields :
    - X a batch representing a single sentence, therefore the size of the batch vary between each batch
    - y a batch of one hot vectors representing the next word to find
    
    """
    corpus = glove_model.dictionary
    corpus_size = len(corpus.keys())
    word_to_int = corpus
    int_to_word = dict((corpus[key], key) for key in corpus.keys())
    with open(txt_path, 'r') as f:
        raw_text = f.read()
    
    if verbose : print("Preprocessing the whole text")

    text = gensim.utils.simple_preprocess(raw_text)[starting_idx:starting_idx+data_size]
    
    if verbose : print("Preprocessing done")
    
    if verbose : print("Nb of samples " + str(2*(len(text)-seq_length-1)))
    dataX = []
    dataY = []
    for i in range(0, len(text)-seq_length-1, 1):

        for direction in [True,False] : 

            if direction :
                seq_in = text[i:i + seq_length]
                seq_out = text[i + seq_length]
                dataX.append([glove_model.word_vectors[word_to_int[word]] for word in seq_in])
                dataY.append(one_hot_vector(seq_out,corpus))
            else : 
                seq_in = text[i+1:i + seq_length+1][::-1]
                seq_out = text[i]
                dataX.append([glove_model.word_vectors[word_to_int[word]] for word in seq_in])
                dataY.append(one_hot_vector(seq_out,corpus))
            #print(len(dataX))
            if len(dataX) >= batch_size :

                X = np.asarray(dataX)
                y = np.asarray(dataY)
                dataX=[]
                dataY=[]
                yield X,y
                
    print('sortie de boucle')
    yield X,y
    
def check_key(key, unwanted_keys):
    """
    This functions checks if the key is not one of the non_selected_keys
    """
    for unwanted_key in unwanted_keys :
        if unwanted_key in key:
            return False
    return True

def lang_model_batch_generator(json_corpus_path, word_indices, batch_nb = 10 , batch_size = 128, maxlen = 10, val_mode = False):
    """
    This batch generator produce samples to train the lstm (shifted mode)
    It selects a json article given its size in the corpus
    
    """
    if val_mode :
        mode = "val_mode"
    else :
        mode = "train_mode"
    
    article_names, article_weights = relevant_articles(json_corpus_path)
    if val_mode : np.random.seed(555) # to always reproduce the same val set
        
    while True:
        if val_mode : np.random.seed(555)# to always reproduce the same val set

        article_list = np.random.choice(article_names, size = batch_nb, replace = True, p=article_weights)

        for article_name in article_list :
            with open(article_name) as f :
                wiki_json = json.load(f)
            f.close()

            text = " "
            for key in wiki_json.keys():
                if check_key(key,non_selected_keys):
                    prepoc_text = gensim.utils.simple_preprocess(wiki_json[key])
                    text += " ".join(word for word in prepoc_text)
                text+= " "
            list_words = text.lower().split()

            if val_mode: np.random.seed(555) # to always reproduce the same val set

            sample_indices = np.random.choice(range(0,len(list_words)-maxlen),replace=True,size=batch_size)

            sentences = []
            next_words = []

            for sample_idx in sample_indices :
                sentence = ' '.join(list_words[sample_idx: sample_idx + maxlen])
                sentences.append(sentence)
                next_words.append((list_words[sample_idx + maxlen]))

                #print(mode, "sentence : ", sentence) 

            X = np.zeros((len(sentences), maxlen, len(word_indices)), dtype=np.bool)
            y = np.zeros((len(sentences), len(word_indices)), dtype=np.bool)
            for i, sentence in enumerate(sentences):
                for t, word in enumerate(sentence.split()):
                    X[i, t, word_indices[word]] = 1
                y[i, word_indices[next_words[i]]] = 1

            yield X,y