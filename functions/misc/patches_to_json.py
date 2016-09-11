# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 23:28:23 2016

@author: Code

This script is written to make json files from the patches of wikipedia we made
"""

# imports 
import sys
sys.path.append('/Users/Code/Desktop/summarization_query_oriented/parser')
import os
os.chdir('/Users/Code/Desktop/summarization_query_oriented')
import parser
import wikipedia

# folders
patch_no = 0
patches_folder = '/Users/Code/Desktop/data_for_summarization/patches_wikipedia/patch_'
json_folder = '/Users/Code/Desktop/data_for_summarization/json/patch_'+str(patch_no)+'/'

# open a patch
try : 
    with open(patches_folder+str(patch_no),'r') as f:
        titles = f.readlines()
except IOError:
    print 'cannot read', patches_folder+str(patch_no)
except :
    print "Unexpected error:", sys.exc_info()[0]

compteur = 0 #if you interrupted the run of a patch put here the nb of file already written
for title in titles : 
    compteur += 1
    title = title[:-1]
    try : 
        parser.page_to_json(wikipedia.page(title), json_folder)
        print 'writing ' + title
    except IOError:
        print 'cannot write ' + title, json_folder+str(patch_no)
    except :
        print "Unexpected error in " + title + " :", sys.exc_info()[0]     
        