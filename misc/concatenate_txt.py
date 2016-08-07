# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 19:02:38 2016

@author: Code
"""
import json
import os
from os import listdir

my_path = '/Users/Code/Desktop/data_for_summarization/json/patch_0/'
os.chdir(my_path) # here you put the path you want to be
files_name = listdir(my_path)  #gives a list of the names of the files in the directory
#print(files_name)
output_folder_path = '/Users/Code/Desktop/data_for_summarization/txt/'
non_selected_sections = ["title", "external links","further reading","references","see also"]


with open(output_folder_path+"test_20k_500.txt","a") as fout:    
    for file_name in files_name[20000:20500] :
        # check is the article is not "blacklisted" (usually articles that display contents)
        with open(my_path+file_name) as data_file:
            data = json.load(data_file)
            
            write_file = True
            
            for key in data.keys():
                if "{\\" in data[key] : 
                    write_file = False
                    break
                
            if write_file :  
                for key in data.keys():
                    if key.lower() in non_selected_sections  :
                        pass
                    else:                     
                        text = data[key]                    
                        fout.write(text.encode('utf-8'))
fout.close()