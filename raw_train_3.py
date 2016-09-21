# training file , I remove the concept of validation loss

# In this script we perform the training of the fully connected model

# Import 
import bisect
import collections
import copy
import gensim
import json
import keras
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import pyrouge
from pyrouge import Rouge155
import random
import re
import time

# paths to folder 
data_json = "/home/ubuntu/summarization_query_oriented/data/wikipedia/json/patch_0/"
data_txt = "/home/ubuntu/summarization_query_oriented/data/wikipedia/txt/"
lang_model_folder = "/home/ubuntu/summarization_query_oriented/nn_models/language_models/d2v/"
nn_summarizers_folder = "/home/ubuntu/summarization_query_oriented/nn_models/nn_summarizer/"
title_file = "/home/ubuntu/summarization_query_oriented/data/DUC/duc2005_topics.sgml"
titles_folder = "/home/ubuntu/summarization_query_oriented/data/DUC/duc2005_docs/"
model_dir = "/home/ubuntu/summarization_query_oriented/data/DUC/duc2005_summary_model"
valset_dir = "/home/ubuntu/summarization_query_oriented/data/validation_set/"
summary_system_super_folder = "/home/ubuntu/summarization_query_oriented/data/DUC/duc2005_summary_system/"
# training parameters

patience_limit = 25

# validation data 

X_val = np.load(valset_dir + "X_val.npy")
y_val = np.load(valset_dir + "y_val.npy")


# useful functions to put in a separate file next

non_selected_keys = ["title", "external links","further reading","references","see also"]

def has_at_least_one_relevant_key(file_as_dict):
    
    for key in file_as_dict.keys():
        b = True
        for unwanted_key in non_selected_keys:
            if unwanted_key in key.lower() :
                b = False    
        if b :
            return True
    return False
        
def has_irrelevant_content(file_as_dict):
    # remove articles with mathematics of chemics
    for key in file_as_dict.keys():
        if "{\\" in file_as_dict[key]:
            return True        

    # check that there is at least one interesting key
    if not has_at_least_one_relevant_key(file_as_dict):
        return True

    return False


def relevant_articles(article_folder_path, min_size = 10000) : 
    """
    inputs :
        - absolute path of the folder containing all the json articles
        - min_size : retaining only file with at least size = min_size*10^-4 ko
    output : 
        - article_names: nd array of the names of the relevant articles (absolute paths)
        - article_weights : nd array normalized of the weights of each files
    """
    all_names =  [f for f in listdir(article_folder_path)]
    article_names = []
    article_weights = []
    for name in all_names:
        article_weight = os.path.getsize(article_folder_path+name)
        if article_weight > min_size:
            # the size of the article meets the requirement
            
            with open(article_folder_path+name) as f :
                file_as_dict = json.load(f) # get article as dict
            
            if not has_irrelevant_content(file_as_dict):
                article_names.append(article_folder_path+name)
                article_weights.append(article_weight)
    
    article_names = np.asarray(article_names)
    article_weights = (np.asarray(article_weights) + 0.0) / np.sum(article_weights)
        
    return article_names, article_weights
            
def select_key(file_as_dict, patience = 10):
    if patience > 0 :
        assert has_at_least_one_relevant_key(file_as_dict), "the file has no relevant key"

        keys = file_as_dict.keys()
        rand_idx = np.random.randint(0,len(keys))
        selected_key = keys[rand_idx]

        if len(file_as_dict[selected_key].split("."))<=2:
            return select_key(file_as_dict, patience = patience - 1)

        for unwanted_key in non_selected_keys :
            if unwanted_key in selected_key.lower() :
                return select_key(file_as_dict, patience = patience - 1)

        return selected_key
    else : 
        keys = file_as_dict.keys()
        rand_idx = np.random.randint(0,len(keys))
        selected_key = keys[rand_idx]
        return selected_key

def create_triplets(d2v_model, article_names, article_weights, nb_triplets=20, triplets_per_file=5, neg_ratio=0.5, str_mode = False) :
    """
    inputs :    
        - d2v_model : paragraph vector model 
        - article_names : ndarray containing the names of the json files (absolute path !)
        - article_weights: ndarray normalized of the weight of each files 
        - nb_triplets : nb of triplets to generate
        - triplets_per_file : number of triplet built for each selected file
        - neg_ratio : ratio of positives / negative examples. Negative examples are taken inside the article !
        
    output : 
        - triplets : nd_array of triplets of shape (nb_triplets+ , embed_dim)
        - labels : nd_array of labels of shape (nb_triplets+ ,)

    """
    triplets = []
    labels = []
    
    assert nb_triplets>=triplets_per_file, "you should have nb_triplets > triplets_per_file"
    
    # nb of pos / neg triplets per file
    neg_per_file = np.floor(triplets_per_file*neg_ratio) #number of negative triplets to generate given(query + partial summary)
    assert neg_per_file >= 1, "you have to increase your neg_ratio"
    
    nb_files = nb_triplets / triplets_per_file
    selected_files_array = np.random.choice(article_names, size=nb_files, p=article_weights, replace = False)
    
    for full_name in selected_files_array :
        with open(full_name) as f :
            file_as_dict = json.load(f)
        
        counter = 0
        while counter < triplets_per_file :
            
            # select a key for positive examples
            key_pos = select_key(file_as_dict)
            
            triplet = build_triplet(d2v_model, file_as_dict, key_pos, positive = True, str_mode = str_mode)
            label = 1
            
            triplets.append(triplet)
            labels.append(label)
            counter += 1 
            
            if neg_ratio < 1 : 
                
                if np.random.rand() < neg_ratio :
                    
                    triplet = build_triplet(d2v_model, file_as_dict, key_pos, positive = False, str_mode = str_mode)
                    label = 0
                    
                    triplets.append(triplet)
                    labels.append(label)
                    counter += 1 

            else :
                
                for n in range(int(np.floor(neg_ratio))):
                    
                    triplet = build_triplet(d2v_model, file_as_dict, key_pos, positive = False, str_mode = str_mode)
                    label = 0
                    
                    triplets.append(triplet)
                    labels.append(label)
                    counter += 1 

            
    triplets = np.asarray(triplets)[:nb_triplets]
    labels = np.asarray(labels)[:nb_triplets]
    
    return triplets, labels

def build_triplet(d2v_model, file_as_dict, key_pos, positive = True, str_mode = False):

    query_str = key_pos
    query_prep = gensim.utils.simple_preprocess(query_str, deacc=True)
    query_vector = d2v_model.infer_vector(query_prep)
    
    summary_str = file_as_dict[key_pos]
    sentences = summary_str.split(".")
    
    partial_summary = []
    candidates = []
    
    size_partial_summary = np.random.rand()
    
    for sentence in sentences: 
        if np.random.rand() < size_partial_summary :
            partial_summary.append(sentence)
        else :
            candidates.append(sentence)
    
    candidate = ""
    counter_candidate = 0
    while (candidate == "" or partial_summary == "") and counter_candidate < 10:
        counter_candidate += 1
        
        if positive : 
            if len(candidates) > 0:
                random_candidate_index = np.random.randint(0,len(candidates))
                candidate = candidates[random_candidate_index]
            else :
                random_candidate_index = np.random.randint(0,len(partial_summary))
                candidate = partial_summary[random_candidate_index]
                partial_summary[random_candidate_index] = ""


            candidate_prep = gensim.utils.simple_preprocess(candidate, deacc=True)
            candidate_vector = d2v_model.infer_vector(candidate_prep)

        else :

            key_neg = select_key(file_as_dict)
            counter = 0

            while key_neg == key_pos and counter<10 : # the counter is for the preproduction code 
                counter += 1
                key_neg = select_key(file_as_dict)

            summary_str = file_as_dict[key_neg]

            sentences = summary_str.split('.')
            random_candidate_index = np.random.randint(0,len(sentences))
            candidate = sentences[random_candidate_index]
            candidate_prep = gensim.utils.simple_preprocess(candidate, deacc=True)
            candidate_vector = d2v_model.infer_vector(candidate_prep)
        
        partial_summary_str = "".join(partial_summary)
        partial_summary_prep = gensim.utils.simple_preprocess(partial_summary_str, deacc=True)
        partial_summary_vector = d2v_model.infer_vector(partial_summary_prep)
    
    if str_mode :
        return query_str, partial_summary_str, candidate
    else :
        return np.hstack( [query_vector, partial_summary_vector, candidate_vector] )


def doc_title_table(title_file):
    with open(title_file , 'r') as f :
        lines = f.readlines()
        raw_text = "".join(l for l in lines)
        left_idx_num = [ m.end(0) for m in re.finditer(r"<num>",raw_text)]
        right_idx_num = [ m.start(0) for m in re.finditer(r"</num>",raw_text)]

        left_idx_title = [ m.end(0) for m in re.finditer(r"<title>",raw_text)]
        right_idx_title = [ m.start(0) for m in re.finditer(r"</title>",raw_text)]

        docs_title_dict = {}
        for i in range(len(left_idx_num)):
            docs_title_dict[raw_text[left_idx_num[i]+1:right_idx_num[i]-1]] = raw_text[left_idx_title[i]+1:right_idx_title[i]-1]
    return docs_title_dict

def merge_articles(docs_folder):
""" for DUC corpus """ 
    s = ""
    
    for doc in os.listdir(docs_folder):
        try:
            with open(docs_folder + doc ,'r') as f:

                lines = f.readlines()
                raw_doc = "".join(txt for txt in lines)
                left_idx_headline = [ m.end(0) for m in re.finditer(r"<HEADLINE>",raw_doc)]
                right_idx_headline = [ m.start(0) for m in re.finditer(r"</HEADLINE>",raw_doc)]

                left_idx_text = [ m.end(0) for m in re.finditer(r"<TEXT>",raw_doc)]
                right_idx_text = [ m.start(0) for m in re.finditer(r"</TEXT>",raw_doc)]

                raw_headline = raw_doc[left_idx_headline[0]:right_idx_headline[0]]
                raw_text = raw_doc[left_idx_text[0]:right_idx_text[0]]

                left_idx_paragraph_headline = [ m.end(0) for m in re.finditer(r"<P>",raw_headline)]
                right_idx_paragraph_headline = [ m.start(0) for m in re.finditer(r"</P>",raw_headline)]

                left_idx_paragraph_text = [ m.end(0) for m in re.finditer(r"<P>",raw_text)]
                right_idx_paragraph_text = [ m.start(0) for m in re.finditer(r"</P>",raw_text)]

                for i in range(len(left_idx_paragraph_headline)):
                    s += raw_headline[left_idx_paragraph_headline[i]:right_idx_paragraph_headline[i]-2] + "."

                for i in range(len(left_idx_paragraph_text)):
                    s += raw_text[left_idx_paragraph_text[i]:right_idx_paragraph_text[i]-1]
        except:
            pass

    return s

def summarize(text, query, d2v_model, nn_model, limit = 250):

    query_prep = gensim.utils.simple_preprocess(query, deacc=True)
    query_vector = d2v_model.infer_vector(query_prep)
    
    summary  = ""
    summary_vector = d2v_model.infer_vector([""])
    summary_idx = []
    
    sentences = text.split('.')
    sentences = np.asarray(sentences)
    
    remaining_sentences = copy.copy(sentences)
    
    size = 0
    counter = 0
    while size < limit and len(remaining_sentences)>0 :
        counter = counter+1
        scores = []
        for sentence in remaining_sentences :
            
            
            sentence_prep = gensim.utils.simple_preprocess(sentence, deacc=True)
            sentence_vector = d2v_model.infer_vector(sentence_prep)

            nn_input = np.hstack([query_vector, summary_vector, sentence_vector])
            nn_input = np.asarray([nn_input]) # weird but it is important to do it
            score = nn_model.predict(nn_input) 
            scores.append(score)
        #print(scores)
        max_idx_rem = int(np.argmax(scores))
        idx_selected_sentence = np.arange(len(sentences))[sentences == remaining_sentences[max_idx_rem]]
        idx_selected_sentence = int(idx_selected_sentence[0])
        size += len(remaining_sentences[max_idx_rem].split())
        
        remaining_sentences = list(remaining_sentences)
        del remaining_sentences[max_idx_rem]
        bisect.insort_left(summary_idx,idx_selected_sentence)

        summary  = ""

        for idx in summary_idx:
            summary = summary + " " + sentences[idx]

        summary_prep = gensim.utils.simple_preprocess(summary, deacc=True)
        summary_vector = d2v_model.infer_vector(summary_prep)

    return summary

## loading a d2vmodel (to be a shifted LSTM next ...)

# parameters of doc2vec
dm = 0
min_count = 5
window = 10
size = 400
sample = 1e-4
negative = 5
workers = 4
epoch = 20

# Initialize the model ( IMPORTANT )
d2v_model = gensim.models.doc2vec.Doc2Vec(dm=dm,min_count=min_count, window=window, size=size, sample=sample, negative=negative, workers=workers,iter = epoch)

# load model
model_name ="dm_"+str(dm)+"_mc_"+str(min_count)+"_w_"+str(window)+"_size_"+str(size)+"_neg_"+str(negative)+"_ep_"+str(epoch)
try :
    d2v_model = d2v_model.load(lang_model_folder+model_name+".d2v")
except :
    print "try a model in : ", os.listdir(lang_model_folder)
print("model loaded")


## get wikipedia data

article_names, article_weights = relevant_articles(data_json)

# DUC data 
docs_title_dict = doc_title_table(title_file)

## design a fully connected model

fc_model = Sequential()

fc_model.add(Dense(120, input_dim=1200))
fc_model.add(Activation('sigmoid'))
fc_model.add(Dropout(0.5))

fc_model.add(Dense(12))
fc_model.add(Activation('sigmoid'))
fc_model.add(Dropout(0.5))

fc_model.add(Dense(1))
fc_model.add(Activation('sigmoid'))

# compiling the model
fc_model.compile(loss="binary_crossentropy", optimizer='sgd')

# training per batch
batch_per_epoch = 2000
#batch_per_epoch = 20
batch_size = 128
patience = 0
batch_counter = 19
rouge_su4_recall_max = 0
rouge_2_recall_max = 0

while patience < patience_limit :
    # train on several batchs
    for i in range(batch_per_epoch):
        triplets, labels = create_triplets(d2v_model, article_names, article_weights, nb_triplets=batch_size, triplets_per_file=16, neg_ratio=1, str_mode = False)
        fc_model.train_on_batch(triplets, labels)
    
    batch_counter += 1

    # summarize DUC
    str_time = time.strftime("%Y_%m_%d")
    fc_model_name = str_time+"_fc_model_batch_"+str(batch_counter)+"k"
    system_folder = summary_system_super_folder+fc_model_name+"/"
    os.mkdir(system_folder)
    
    for docs_key in docs_title_dict.keys():

        docs_folder = titles_folder+docs_key+"/"
        text = merge_articles(docs_folder)
        query = docs_title_dict[docs_key]
        summary = summarize(text,query,d2v_model, fc_model, limit = 250)

        summary = " ".join(summary.split()[:250])

        with open(system_folder+docs_key,'w') as f :
            f.write(summary)
            print 'writing in '+ system_folder + docs_key

            
    # perform rouge
    r = Rouge155()
    r.system_dir = system_folder
    r.model_dir = model_dir 
    r.system_filename_pattern = 'd(\d+)[a-z]'
    r.model_filename_pattern = 'D#ID#.M.250.[A-Z].[A-Z]'
        
    options =  '-a -d -e ' + r._data_dir + ' -m -n 2 -s -2 4 -u -x -f B'

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
