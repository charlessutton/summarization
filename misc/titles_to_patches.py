# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 23:44:19 2016

@author: Code

This script is used to build patches of titles
"""

# import 
import random

# open data
path_to_titles_file = "/Users/Code/Downloads/enwiki-latest-all-titles-in-ns0"
with open(path_to_titles_file,'r') as f:
    lines = f.readlines()


# shuffle the data
random.seed(12345)
random.shuffle(lines)

# split it into patches of 10 k titles

# save into multiple files

path_to_save = "/Users/Code/Desktop/patches_wikipedia/"

for i in range(len(lines) / 100000) :
    with open(path_to_save+"patch_"+str(i),'w') as f :
        f.writelines(lines[100000*i:100000*(i+1)])
#the last
with open(path_to_save+"patch_"+str(126),'w') as f :
    f.writelines(lines[100000*126:])
        