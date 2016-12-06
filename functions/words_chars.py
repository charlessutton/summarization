import os
import json
import gensim 

def vocabulary_from_json_corpus(json_corpus_path, chars_mode = False):
    chars = set()
    words = set()
    for json_file in os.listdir(json_corpus_path):
        with open(json_corpus_path+json_file) as f :
            wiki_json = json.load(f)
        f.close()
        for key in wiki_json.keys():

            text = wiki_json[key]
            word_list = gensim.utils.simple_preprocess(text)
            text = " ".join(word for word in word_list)
            chars.update(set(text))
            words.update(set(text.split()))
    
    if chars_mode: 
        return chars
    else:
        return words
    
def doc2bow(text, word_indices):
    """
    take a text as input and return a sparse bow vector as output
    text is a sequence of unicode strings (typically output of gensim.utils.simple_preprocessing) 
    this function ignores words that are not in the dictionary
    """
    res = np.zeros(len(word_indices))
    for word in text :
        try :
            res[word_indices[word]] += 1 
        except KeyError : #case where the word is not in the dictionnary
            pass
    return res