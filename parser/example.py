# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 17:36:27 2016

@author: Code
"""
#Imports
import wikipedia
import os
from parser import page_to_json, page_to_dict

#initialisation 
folder_name = "/Users/Code/Desktop/summarization_query_oriented/json_files/" #folder where you want to save your json file
title = "François Hollande" #put exact title name of the wikipedia page you want to parse

#code
page = wikipedia.page(title)
page_to_json(page,folder_name)