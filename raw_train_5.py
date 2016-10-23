# Quadruplets 

# training file , I remove the concept of validation loss

# In this script we perform the training of the fully connected model

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
from functions.training_functions import *

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

## loading a d2vmodel (to be a shifted LSTM next ...)

# parameters of doc2vec
dm = 0
min_count = 5
window = 10
size = 400
sample = 1e-4
negative = 5
workers = 4
epoch = 100

# Initialize the model ( IMPORTANT )
d2v_model = gensim.models.doc2vec.Doc2Vec(dm=dm,min_count=min_count, window=window, size=size, sample=sample, negative=negative, workers=workers,iter = epoch)

# load model
model_name ="dm_"+str(dm)+"_mc_"+str(min_count)+"_w_"+str(window)+"_size_"+str(size)+"_neg_"+str(negative)+"_ep_"+str(epoch)+"_wosw"
try :
    d2v_model = d2v_model.load(lang_model_folder+model_name+".d2v")
except :
    print "try a model in : ", os.listdir(lang_model_folder)
print("model loaded")

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
fc_model.compile(loss="binary_crossentropy", optimizer='adagrad')

# training per batch
batch_per_epoch = 50
#batch_per_epoch = 20
batch_size = 128
patience = 0
batch_counter = 6000
rouge_su4_recall_max = 0
rouge_2_recall_max = 0

while patience < patience_limit :
    # train on several batchs
    for i in range(batch_per_epoch):
        triplets, labels = create_triplets(d2v_model, article_names, article_weights, nb_triplets=batch_size, triplets_per_file=16, neg_ratio=1, str_mode = False, with_txt_vect=True) # HERE IS THE NEW STUFF
        fc_model.train_on_batch(triplets, labels)
    
    batch_counter += 1

    # summarize DUC
    str_time = time.strftime("%Y_%m_%d")
    fc_model_name = "T5"+str_time+"_fc_model_batch_"+str(batch_counter)+"k"
    system_folder = summary_system_super_folder+fc_model_name+"/"
    os.mkdir(system_folder)

    for theme in themes :
        theme_folder = tdqfs_folder + theme + "/"
        theme_doc_folder = theme_folder + theme + "/"
        queries = get_queries(theme_folder+"queries.txt")
        text = merge_articles_tqdfs(theme_doc_folder)
        for i in range(len(queries)):
            query = queries[i]
            summary = summarize(text,query,d2v_model, fc_model, limit = 250, with_txt_vect=True)
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