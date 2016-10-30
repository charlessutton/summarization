# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10th 2016

@author: Charles Sutton

This script is written to make json files from the patches of wikipedia we made
"""
# imports 
import sys
sys.path.append('/home/ubuntu/summarization_query_oriented/functions/')
import os

from data_acquisition import rank_to_title, counter
from my_parser.parser import page_to_json
import wikipedia

# folders
json_folder = '/home/ubuntu/summarization_query_oriented/data/wikipedia/json/td_qfs_rank_1/'

titles_of_interest = ["Alzheimer's disease","Asthma","Cancer", "Obesity"]

all_titles = []
for title in titles_of_interest:
    rk_to_ttl = rank_to_title(title)
    all_titles.extend(rk_to_ttl.keys())

all_titles = list(set(all_titles))

compteur = 1470 #if you interrupted the run of a patch put here the nb of file already written
for title in all_titles[compteur:] :
    if compteur%10==0 : print compteur , "/" , len(all_titles) 
    compteur += 1

    try : 
        page_to_json(wikipedia.page(title), json_folder)
        print 'writing ' + title
        pass
    except IOError:
        print 'cannot write ' + title
    except :
        print "Unexpected error in " + title + " :", sys.exc_info()[0]     
        
