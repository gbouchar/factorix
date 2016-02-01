# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 13:53:15 2016

This file is for reading in and preparing the FB15K237 dataset
(and potentially other datasets?)

@author: Johannes
"""

import Dictionaries as Dict

#global variable: directory of dataset (choose yourself)
#dir_FB15K237 = '/Users/Johannes/PhD/CFM/FB15K237/'  
dir_FB15K237 = '/home/ucabmwe/Scratch/FB_data/'


# function only or loading the textual mentions of FB15K237
def read_FB15K237_mentions():
    filename = dir_FB15K237 + 'Release/text_emnlp.txt'
    f = open(filename, "rb")
    rawtext = f.read()
    f.close()
    return rawtext


# function for loading the FB15K237 dataset, ready as integer tuples.
# interactions define which types interact with one another
# when loading the train set, the mentions will be included and dictionaries 
# created. When using 'which' as "valid" or "test", these dictionaries must
# be passed to this function, too, in order to translate the valid/test data
# into integer tuples.
def load_FB15K237_FB(which = 'train', String2Int=None, Int2String=None,
                     interactions = None, threshold_mentions=10):
    filename = dir_FB15K237 + 'fb15k-237kb/' + which + '.txt'
    f = open(filename, "rb")
    rawtext = f.read()
    f.close()
    rawtext = str(rawtext)
    
    str_triples = [line.split("\\t")[:3] for line in rawtext.split("\\n")[:-1] ]
    
    if which == 'train':
        # load textual mentions, add to str_triples
        rawtext_mentions = str(read_FB15K237_mentions())
        counter = 0
        for iline,line in enumerate( rawtext_mentions.split('\\r\\n') ):
            if len(line) == 0:
                continue
            splitted = line.split('\\t')
            if len(splitted) != 4:
                continue
            if int(splitted[3]) < threshold_mentions:
                continue
            counter += 1
            str_triples += [ [splitted[i] for i in [0,1,2]] ]
        print("Using ", counter, " textual mentions for training.")
    
        # now define dictionaries
        String2Int, Int2String = Dict.DefineGlobalDict(str_triples, interactions)
    
    
    # if not train, then dictionaries String2Int and Int2String must be provided
    int_tuples = Dict.convert_String_Tuples_to_Integer_Tuples(str_triples,
                                                                  String2Int, 
                                                                  interactions)
    return int_tuples, String2Int, Int2String
    
    
    
    
    
    
# demo of usage. Note: global variable 'dir_FB15K237' needs to be set before
# to match the directory where the FB15K237 data is stored.
if 0:
    interactions = [(3,(0,1)), (4,(1,2))]
    thresh_mentions = 5
    train, String2Int, Int2String = load_FB15K237_FB('train', None, None,                      
                                                     interactions, thresh_mentions)

    val, String2Int, Int2String = load_FB15K237_FB('valid', String2Int, 
                                                   Int2String, interactions)

    test, String2Int, Int2String = load_FB15K237_FB('test', String2Int, 
                                                    Int2String, interactions)
   
    
