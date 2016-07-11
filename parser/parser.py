# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 20:24:16 2016

@author: Charles Sutton

This is the parser used to organise the wikipedia corpus for our query oriented summarization main model
"""

import re
from bisect import bisect_right
import numpy as np
import json

def page_to_dict(page):
    """
    INPUT
    - page : a wikipedia page (see pkg wikipedia)
    OUTPUT : 
    - query_summary , a dictionnary where the key are either
        i. the title
        ii. title and subsection title
        iii. title, subsection title and sub sub section title
    Values are the corresponding text in the wikipedia page
    """ 
    text = page.content
    query_summary = {}
    
    #name of the page not to use for further training
    query_summary["title"] = page.title

    # title level
    title = page.title
    summary = page.summary
    
    query_summary[title] = summary
    
    #preparing sub and sub sub section level    
    sub_titles = re.findall("\n={2} .* ={2}\n",text)
    sub_sub_titles = re.findall("\n={3} .* ={3}\n",text)
    
    sub_titles_idx = []
    for match in re.finditer("\n={2} .* ={2}\n",text):
        sub_titles_idx.append(match.start())
    
    sub_sub_titles_idx = []    
    for match in re.finditer("\n={3} .* ={3}\n",text):
        sub_sub_titles_idx.append(match.start())
    
    sub_titles = np.asarray(sub_titles)
    sub_sub_titles = np.asarray(sub_sub_titles)
    sub_titles_idx = np.asarray(sub_titles_idx)
    sub_sub_titles_idx = np.asarray(sub_sub_titles_idx)

    #subsection level 
    for sub_title in sub_titles :
        query = title + ' ' + sub_title[4:-4]
        summary = page.section(sub_title[4:-4])
        query_summary[query] = summary
        
    #subsubsection level 
    for i in range(len(sub_sub_titles)):
        idx = sub_sub_titles_idx[i]        
        sub_idx = find_le(sub_titles_idx ,idx)
        sub_title = sub_titles[sub_titles_idx == sub_idx][0]
        sub_sub_title = sub_sub_titles[i]
        query = title + ' ' + sub_title[4:-4] + ' ' + sub_sub_title[5:-5]
        summary = page.section(sub_sub_title[5:-5])
        query_summary[query] = summary        
        
    return query_summary

def post_process(query_summary, delete_references = True, delete_ext_links = True):
    """ 
    INPUT : the dict query_summary dictionnary built by page_to_dict
    OUTPUT : the dict where we removed empty values, references and other subsections ...
    """
    
    # remove empty    
    for key in query_summary.keys():
        if not query_summary[key] :
            del query_summary[key]
            
    # remove references 
        #to complete
    
    # remove external links
        
    # remove other unuseful subsection
    return query_summary

def page_to_json(page, folder_path):
    """ 
    INPUT
    - page : a wikipedia page (see package wikipedia), 
    - folder_path : the path to the folder where you want to save the json files
    OUTPOUT 
    - void, a json file named according to the title in wikipedia is written in the folder
    """
    name = page.title
    query_summary = page_to_dict(page)
    query_summary = post_process(query_summary)
    with open(folder_path+name+'.json', 'w') as fp:
        json.dump(query_summary, fp)
    

def find_le(a, x):
    'Find rightmost value less than or equal to x'
    i = bisect_right(a, x)
    if i:
        return a[i-1]
    raise ValueError
