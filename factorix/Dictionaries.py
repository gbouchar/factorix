# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:49:41 2016

@author: Johannes
@author: Guillaume
@author: Sebastian
"""
from collections import defaultdict
import numpy as np
import re

class Indexer:
    """
    Mapping from strings to integers and vice versa.

    New words get new indices, starting at 0
    >>> index = Indexer()
    >>> index('cat')
    0

    The next new word gets the next index
    >>> index('the')
    1

    They keep their indices when called again
    >>> index('cat')
    0

    The index remembers the inverse mapping
    >>> index.inv(0)
    'cat'

    Batch calls:
    >>> index.ints('the','cat')
    [1, 0]

    >>> index.strings(1,0)
    ['the', 'cat']

    Freeze the dictionary
    >>> index.freeze()
    >>> index('dog')
    Traceback (most recent call last):
     ...
    KeyError: 'dog not indexed yet and indexer is frozen'
    """

    def __init__(self):
        self.current = 0
        self.index = {}
        self.index_to_string = {}
        self.frozen = False

    def freeze(self, frozen = True):
        self.frozen = frozen

    def string_to_int(self, string):
        if string in self.index:
            return self.index[string]
        else:
            if not self.frozen:
                result = self.current
                self.index[string] = result
                self.index_to_string[result] = string
                self.current += 1
                return result
            else:
                raise KeyError('{} not indexed yet and indexer is frozen'.format(string))

    def int_to_string(self, int):
        return self.index_to_string[int]

    def inv(self, string):
        return self.int_to_string(string)

    def __call__(self, string):
        return self.string_to_int(string)

    def ints(self, *strings):
        return [self.string_to_int(string) for string in strings]

    def strings(self, *ints):
        return [self.int_to_string(i) for i in ints]


class Vocabulary(object):
    def __init__(self, first_words=[]):
        self.word2idx = dict()
        self.idx2word = []
        self.frozen = False
        for w in first_words:
            self.get_idx(w)
        self.unknown_word_idx = None
        for w in first_words:
            self.get_idx(w)

    def freeze(self, frozen=True, unknown_word="<UNK>"):
        if not self.frozen:
            self.unknown_word_idx = self.get_idx(unknown_word)
        self.frozen = frozen

    def __str__(self):
        return 'Vocab ' + str(self.word2idx)

    def size(self):
        return len(self.word2idx)

    def get_idx(self, word):
        if word not in self.word2idx:
            if self.frozen:
                return self.unknown_word_idx
            else:
                # the token did not exist, add it to the vocabulary
                self.word2idx[word] = len(self.idx2word)
                self.idx2word.append(word)  # important: should come after the previous line to keep indices right
        return self.word2idx[word]

    def str2tok(self, sentence, update=True):
        if update:
            return [self.get_idx(w) for w in sentence_split(sentence)]
        else:
            return [self.word2idx[w] for w in sentence_split(sentence)]

    def get_word(self, idx):
        return self.idx2word[int(idx)]

    def toks2str(self, ut):
        return ' '.join([self.idx2word[t] for t in ut]).replace(' ?', '?').replace(' .', '.')

    def array2str(self, a):
        f = np.vectorize(self.get_word)
        return f(a)


# ##################################################################################


def sentence_split(s):
    s = re.sub(' +', ' ', s.replace('?', ' ?').replace('.', ' .'))
    return s.split(' ')



"""
1) GlobalDict One dictionary for EVERY possible entry in any of the types/domains
"""



def convert_String_Tuples_to_Integer_Tuples(string_tuples, String2Int, interactions):
    int_representations = []
    for string_tuple in string_tuples:
        unary = []    
        for i_s, s in enumerate(string_tuple):
            s = str(i_s) + ">>" + s
            unary += [ String2Int[s] ] 
        
        binary = []
        for interaction in interactions:
            i1,i2 = interaction[1]
            type_indicator = str(i1) + "," +  str(i2)
            s = string_tuple[i1] + "::::" + string_tuple[i2]
            s = type_indicator + ">>" + s
            binary += [ String2Int[s] ] 
        int_representations += [unary + binary]
    return int_representations
        


#interactions = [(3,(0,1)), (4,(1,2))]: interactions between types.
#format example: (3,(0,1)). 3: column where this interaction will be located
#(0,1) types that interact in this column.
def DefineGlobalDict(string_tuples, interactions = None):
    String2Int = defaultdict(int)
    Int2String = {}
    
    dictStrings = set([])
    
    L = len(string_tuples)
    for i_st, str_tuple in enumerate(string_tuples):
        if not i_st % 1000:
            print ("Defining Global Dict.. ", i_st, "/", L)
        for i_type, s in enumerate(str_tuple):
            # collect entities for each type, with no type interaction.
            string = str(i_type) + ">>" + s
            dictStrings.add(string)
            
            # now allow for type interaction for every
        for interaction in interactions:
            interaction = interaction[1]    #first element is column
            type_indicator = str(interaction[0]) + "," +  str(interaction[1])
            s = str_tuple[interaction[0]] + "::::" + str_tuple[interaction[1]]
            string = type_indicator + ">>" + s
            #if string not in dictStrings:
            #    dictStrings += [string]
            dictStrings.add(string)
            
                
    for i_string, string in enumerate(list(dictStrings)):
        String2Int[string] = i_string + 1  #start indexing at 1; 0 reserved for not in dict.
        Int2String[i_string + 1] = string
    return String2Int, Int2String
    
    


# demo of usage
if 0:
    interactions = [(3,(0,1)), (4,(1,2))]
    string_tuples = [["ent1", "rel1", "ent2"],
                     ["ent3", "rel2", "ent4"],
                     ["ent5", "rel3", "ent6"],
                     ["ent7", "rel4", "ent8"],
                     ["ent9", "rel5", "ent10"],
                     ["ent11", "rel6", "ent12"]]
    t1 = string_tuples[0]


    String2Int, Int2String = DefineGlobalDict(string_tuples, interactions)
    
    convert_String_Tuples_to_Integer_Tuples([t1], String2Int, interactions)    
    