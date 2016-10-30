import bisect
import collections
import copy
import gensim
import json
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import random
import re
import time

from hard_coded import non_selected_keys

def has_at_least_one_relevant_key(file_as_dict):
    """
    check if the article contains at least a key that is not in non_selected_keys
    """
    for key in file_as_dict.keys():
        b = True
        for unwanted_key in non_selected_keys:
            if unwanted_key in key.lower() :
                b = False    
        if b :
            return True
    return False
        
def has_irrelevant_content(file_as_dict):
    """
    check if the article has scientific content (\\display etc..), or if the article doesn't have enough relevant keys
    """
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
    """
    Select a relevant key (at least recursively up to nb of patience iteration), otherwise return the last key selected
    """
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

def create_triplets(d2v_model, article_names, article_weights, nb_triplets=20, triplets_per_file=5, neg_ratio=0.5, str_mode = False, with_txt_vect = False) :
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
            
            triplet = build_triplet(d2v_model, file_as_dict, key_pos, positive = True, str_mode = str_mode, with_txt_vect=with_txt_vect)
            label = 1
            
            triplets.append(triplet)
            labels.append(label)
            counter += 1 
            
            if neg_ratio < 1 : 
                
                if np.random.rand() < neg_ratio :
                    
                    triplet = build_triplet(d2v_model, file_as_dict, key_pos, positive = False, str_mode = str_mode, with_txt_vect=with_txt_vect)
                    label = 0
                    
                    triplets.append(triplet)
                    labels.append(label)
                    counter += 1 

            else :
                
                for n in range(int(np.floor(neg_ratio))):
                    
                    triplet = build_triplet(d2v_model, file_as_dict, key_pos, positive = False, str_mode = str_mode,with_txt_vect=with_txt_vect)
                    label = 0
                    
                    triplets.append(triplet)
                    labels.append(label)
                    counter += 1 

            
    triplets = np.asarray(triplets)[:nb_triplets]
    labels = np.asarray(labels)[:nb_triplets]
    
    return triplets, labels

def build_triplet(d2v_model, file_as_dict, key_pos, positive = True, str_mode = False, remove_stop_words=True, with_txt_vect = False):
    if remove_stop_words : 
        stopwords = stop_words()
    else :
        stopwords = []
        
    if with_txt_vect :
        text_str = ""
        for key in file_as_dict.keys():
            if key not in non_selected_keys :
                text_str += file_as_dict[key]
        t1 = time.time()
        text_prep = gensim.utils.simple_preprocess(text_str, deacc=True)
        t2 = time.time() 
        text_vector = d2v_model.infer_vector(remove_stopwords(text_prep,stopwords))
        t3 = time.time()
        #print "preprocess" + str(t2-t1)
        #print "infer_vector" + str(t3-t2)
            
    query_str = key_pos
    query_prep = gensim.utils.simple_preprocess(query_str, deacc=True)
    query_vector = d2v_model.infer_vector(remove_stopwords(query_prep,stopwords))
    
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
            candidate_vector = d2v_model.infer_vector(remove_stopwords(candidate_prep,stopwords))

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
            candidate_vector = d2v_model.infer_vector(remove_stopwords(candidate_prep,stopwords))
        
        partial_summary_str = "".join(partial_summary)
        partial_summary_prep = gensim.utils.simple_preprocess(partial_summary_str, deacc=True)
        partial_summary_vector = d2v_model.infer_vector(remove_stopwords(partial_summary_prep,stopwords))
    
    if str_mode :
        return query_str, partial_summary_str, candidate
    elif with_txt_vect:
        return np.hstack( [query_vector, partial_summary_vector, candidate_vector, text_vector])
    else :
        return np.hstack( [query_vector, partial_summary_vector, candidate_vector] )


def doc_title_table(title_file):
    """
    parse the title file, for duc dataset !
    """
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
    """
    naively concatenate doc in the doc folder
    Docs are from the duc dataset !
    """

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

def summarize(text, query, d2v_model, nn_model, limit = 250, remove_stop_words = True,with_txt_vect=False):
    """
    Perform summarization on text given query,
    """
    if remove_stop_words : 
        stopwords = stop_words()
    else :
        stopwords = []
    
    if with_txt_vect :
        text_prep = gensim.utils.simple_preprocess(text, deacc=True)
        text_vector = d2v_model.infer_vector(remove_stopwords(text_prep,stopwords))

        
    query_prep = gensim.utils.simple_preprocess(query, deacc=True)
    query_vector = d2v_model.infer_vector(remove_stopwords(query_prep,stopwords))
    
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
            if with_txt_vect :
                nn_input = np.hstack([query_vector, summary_vector, sentence_vector, text_vector])
            else:
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

def get_queries(query_txt_file):
    with open(query_txt_file, 'r') as f :
        queries = f.readlines()
    return queries

def merge_articles_tqdfs(theme_doc_folder):
    """ for tqdfs corpus """ 
    s = ""
    for source in os.listdir(theme_doc_folder):
        try :
            for doc in os.listdir(theme_doc_folder + source):
                with open(theme_doc_folder + source + "/" + doc ,'r') as f:
                    lines = f.readlines()
                    s += "".join(txt for txt in lines)
                s += " "
        except:
            pass
    return s

def merge_articles_tdqfs(theme_doc_folder):
    """ for tqdfs corpus """ 
    s = ""
    for source in os.listdir(theme_doc_folder):
        try :
            for doc in os.listdir(theme_doc_folder + source):
                with open(theme_doc_folder + source + "/" + doc ,'r') as f:
                    lines = f.readlines()
                    s += "".join(txt.decode('utf-8','ignore') for txt in lines)
                s += " "
        except:
            pass
    return s


def stop_words() :
    return ['a', 'able', 'about', 'above', 'abst', 'accordance',
            'according', 'accordingly', 'across', 'act', 'actually',
            'added', 'adj', 'adopted', 'affected', 'affecting', 'affects',
            'after', 'afterwards', 'again', 'against', 'ah', 'all',
            'almost', 'alone', 'along', 'already', 'also', 'although',
            'always', 'am', 'among', 'amongst', 'an', 'and', 'announce',
            'another', 'any', 'anybody', 'anyhow', 'anymore', 'anyone',
            'anything', 'anyway', 'anyways', 'anywhere', 'apparently',
            'approximately', 'are', 'aren', 'arent', 'arise', 'around',
            'as', 'aside', 'ask', 'asking', 'at', 'auth', 'available',
            'away', 'awfully', 'b', 'back', 'be', 'became', 'because',
            'become', 'becomes', 'becoming', 'been', 'before',
            'beforehand', 'begin', 'beginning', 'beginnings', 'begins',
            'behind', 'being', 'believe', 'below', 'beside', 'besides',
            'between', 'beyond', 'biol', 'both', 'brief', 'briefly',
            'but', 'by', 'c', 'ca', 'came', 'can', 'cannot', "can't",
            'certain', 'certainly', 'co', 'com',
            'come', 'comes', 'contain', 'containing', 'contains',
            'could', "couldn't", 'd', 'date', 'did', "didn't", 'different',
            'do', 'does', "doesn't", 'doing', 'done', "don't", 'down',
            'downwards', 'due', 'during', 'e', 'each', 'ed', 'edu',
            'effect', 'eg', 'eight', 'eighty', 'either', 'else',
            'elsewhere', 'end', 'ending', 'enough', 'especially',
            'et', 'et-al', 'etc', 'even', 'ever', 'every', 'everybody',
            'everyone', 'everything', 'everywhere', 'ex', 'except', 'f',
            'far', 'few', 'ff', 'fifth', 'first', 'five', 'fix',
            'followed', 'following', 'follows', 'for', 'former',
            'formerly', 'forth', 'found', 'four', 'from', 'further',
            'furthermore', 'g', 'gave', 'get', 'gets', 'getting', 'give',
            'given', 'gives', 'giving', 'go', 'goes', 'gone', 'got',
            'gotten', 'h', 'had', 'happens', 'hardly', 'has', "hasn't",
            'have', "haven't", 'having', 'he', 'hed', 'hence', 'her',
            'here', 'hereafter', 'hereby', 'herein', 'heres', 'hereupon',
            'hers', 'herself', 'hes', 'hi', 'hid', 'him', 'himself',
            'his', 'hither', 'home', 'how', 'howbeit', 'however',
            'hundred', 'i', 'id', 'ie', 'if', "i'll", 'im', 'immediate',
            'immediately', 'importance', 'important', 'in', 'inc',
            'indeed', 'index', 'information', 'instead', 'into',
            'invention', 'inward', 'is', "isn't", 'it', 'itd', "it'll",
            'its', 'itself', "i've", 'j', 'just', 'k', 'keep', 'keeps',
            'kept', 'keys', 'kg', 'km', 'know', 'known', 'knows', 'l',
            'largely', 'last', 'lately', 'later', 'latter', 'latterly',
            'least', 'less', 'lest', 'let', 'lets', 'like', 'liked',
            'likely', 'line', 'little', "'ll", 'look', 'looking', 'looks',
            'ltd', 'm', 'made', 'mainly', 'make', 'makes', 'many', 'may',
            'maybe', 'me', 'mean', 'means', 'meantime', 'meanwhile',
            'merely', 'mg', 'might', 'million', 'miss', 'ml', 'more',
            'moreover', 'most', 'mostly', 'mr', 'mrs', 'much', 'mug',
            'must', 'my', 'myself', 'n', 'na', 'name', 'namely', 'nay',
            'nd', 'near', 'nearly', 'necessarily', 'necessary', 'need',
            'needs', 'neither', 'never', 'nevertheless', 'new', 'next',
            'nine', 'ninety', 'no', 'nobody', 'non', 'none',
            'nonetheless', 'noone', 'nor', 'normally', 'nos', 'not',
            'noted', 'nothing', 'now', 'nowhere', 'o', 'obtain',
            'obtained', 'obviously', 'of', 'off', 'often', 'oh', 'ok',
            'okay', 'old', 'omitted', 'on', 'once', 'one', 'ones', 'only',
            'onto', 'or', 'ord', 'other', 'others', 'otherwise', 'ought',
            'our', 'ours', 'ourselves', 'out', 'outside', 'over',
            'overall', 'owing', 'own', 'p', 'page', 'pages', 'part',
            'particular', 'particularly', 'past', 'per', 'perhaps',
            'placed', 'please', 'plus', 'poorly', 'possible', 'possibly',
            'potentially', 'pp', 'predominantly', 'present', 'previously',
            'primarily', 'probably', 'promptly', 'proud', 'provides',
            'put', 'q', 'que', 'quickly', 'quite', 'qv', 'r', 'ran',
            'rather', 'rd', 're', 'readily', 'really', 'recent',
            'recently', 'ref', 'refs', 'regarding', 'regardless',
            'regards', 'related', 'relatively', 'research',
            'respectively', 'resulted', 'resulting', 'results', 'right',
            'run', 's', 'said', 'same', 'saw', 'say', 'saying', 'says',
            'sec', 'section', 'see', 'seeing', 'seem', 'seemed',
            'seeming', 'seems', 'seen', 'self', 'selves', 'sent', 'seven',
            'several', 'shall', 'she', 'shed', "she'll", 'shes', 'should',
            "shouldn't", 'show', 'showed', 'shown', 'showns', 'shows',
            'significant', 'significantly', 'similar', 'similarly',
            'since', 'six', 'slightly', 'so', 'some', 'somebody',
            'somehow', 'someone', 'somethan', 'something', 'sometime',
            'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry',
            'specifically', 'specified', 'specify', 'specifying',
            'state', 'states', 'still', 'stop', 'strongly', 'sub',
            'substantially', 'successfully', 'such', 'sufficiently',
            'suggest', 'sup', 'sure', 't', 'take', 'taken', 'taking',
            'tell', 'tends', 'th', 'than', 'thank', 'thanks', 'thanx',
            'that', "that'll", 'thats', "that've", 'the', 'their',
            'theirs', 'them', 'themselves', 'then', 'thence', 'there',
            'thereafter', 'thereby', 'thered', 'therefore', 'therein',
            "there'll", 'thereof', 'therere', 'theres', 'thereto',
            'thereupon', "there've", 'these', 'they', 'theyd', "they'll",
            'theyre', "they've", 'think', 'this', 'those', 'thou',
            'though', 'thoughh', 'thousand', 'throug', 'through',
            'throughout', 'thru', 'thus', 'til', 'tip', 'to', 'together',
            'too', 'took', 'toward', 'towards', 'tried', 'tries', 'truly',
            'try', 'trying', 'ts', 'twice', 'two', 'u', 'un', 'under',
            'unfortunately', 'unless', 'unlike', 'unlikely', 'until',
            'unto', 'up', 'upon', 'ups', 'us', 'use', 'used', 'useful',
            'usefully', 'usefulness', 'uses', 'using', 'usually', 'v',
            'value', 'various', "'ve", 'very', 'via', 'viz', 'vol',
            'vols', 'vs', 'w', 'want', 'wants', 'was', "wasn't", 'way',
            'we', 'wed', 'welcome', "we'll", 'went', 'were', "weren't",
            "we've", 'what', 'whatever', "what'll", 'whats', 'when',
            'whence', 'whenever', 'where', 'whereafter', 'whereas',
            'whereby', 'wherein', 'wheres', 'whereupon', 'wherever',
            'whether', 'which', 'while', 'whim', 'whither', 'who', 'whod',
            'whoever', 'whole', "who'll", 'whom', 'whomever', 'whos',
            'whose', 'why', 'widely', 'willing', 'wish', 'with', 'within',
            'without', "won't", 'words', 'world', 'would', "wouldn't",
            'www', 'x', 'y', 'yes', 'yet', 'you', 'youd', "you'll",
            'your', 'youre', 'yours', 'yourself', 'yourselves', "you've",
            'z', 'zero',
            #punctuation marks:
            '`', "'", '(', ')', ',', '_', ';', ':', '~', '&', '-', '--',
            '$', '^', '*', #maybe more
            #added due to word tokenizer:
            "'s", "'ll", "'t", "'ve", "'m",
            'doesn', 'don', 'hasn', 'haven', 'isn', 'wasn', 'won', 'weren',
            'wouldn', 'didn', 'shouldn', 'couldn']

def remove_stopwords(sentence_prep, stopwords):
    sentence_wo_sw = []
    for word in sentence_prep:
        if word in stopwords:
            pass
        else : 
            sentence_wo_sw.append(word)
    return sentence_wo_sw