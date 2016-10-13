# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 19:02:38 2016

@author: Code
"""
import json
import os
from os import listdir
from hard_coded import non_selected_keys, data_json_dir, data_txt_dir
my_path = data_json_dir
os.chdir(my_path) # here you put the path you want to be
files_name = listdir(my_path)  #gives a list of the names of the files in the directory
#print(files_name)
output_folder_path = data_txt_dir
non_selected_sections = non_selected_keys

output_file_name = "td_qfs_rank_1_all"

with open(output_folder_path+output_file_name+".txt","w") as fout:
    print(output_folder_path+output_file_name+".txt")
    for file_name in files_name :
        # check is the article is not "blacklisted" (usually articles that display contents)
        with open(my_path+file_name) as data_file:

            data = json.load(data_file)
            
            keys_to_write = []
            
            for key in data.keys():
                if "{\\" in data[key] : 
                    pass
                elif key.lower() in non_selected_sections : 
                    pass
                else : 
                    keys_to_write.append(key)
                
            for key in keys_to_write:
                text = data[key]                    
                fout.write(text.encode('utf-8'))
fout.close()