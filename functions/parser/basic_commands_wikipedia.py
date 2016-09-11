# -*- coding: utf-8 -*-
"""
Created on Tue Jul 05 12:40:06 2016

@author: sutton
"""

#==============================================================================
# wikipedia commands
#==============================================================================
import wikipedia

francois_hollande = wikipedia.page("Francois Hollande")

print type(francois_hollande.title)

print francois_hollande.url

print francois_hollande.content

print francois_hollande.section("Personal life")

francois_hollande.sections

#==============================================================================
# regular expression command
#==============================================================================

import re

import nltk

text = francois_hollande.content

re.findall("== ",text)