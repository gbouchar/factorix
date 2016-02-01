# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 10:24:01 2016

@author: Johannes
"""

import numpy as np
import Dictionaries as Dict

import learn_factorization as lf
import tensorflow as tf








# object_domain: the domain that contains the object
# make batch where everything is constant, only the object varies from a gold_tuple.
def get_ranking_batch(gold_tuple, interactions, object_domain, 
                      String2Int, Int2String, verbose = False):

    gold_obj_index = gold_tuple[object_domain]    
    
    #get identities of all the _objects_ in the dictionary
    object_list = [k for k in String2Int.keys() \
                    if len(k.split(",")) == 1 and \
                    k.split(">>")[0] == str(object_domain) ]
    
     
    #define batch, starting at replication of the gold tuple
    batch = np.tile(gold_tuple, [len(object_list), 1])
    
    #identify the columns to actually change in batch
    interation_of_interest = [i for i in interactions if (object_domain in i[1])]
    
    # string of the old object: if 0 or not zero
    if gold_tuple[object_domain]:
        old_object_string = Int2String[ gold_tuple[object_domain] ]
    else:
        old_object_string = "0"
        print("No object string!")
    
    
    #go through batch, object by object
    for i_obj, obj_string in enumerate(object_list):         
        new_object_index = String2Int[obj_string]

        if verbose:
            print ('--------')
            print ("old oject index", gold_tuple[object_domain]    )
            print ("old oject string", Int2String[ gold_tuple[object_domain]    ])
            print ("new_object_index", new_object_index)
            print ("new_object_string", Int2String[new_object_index])

        # if the new object index is the gold index, remember its position in batch
        if new_object_index == gold_obj_index:
            gold_index = i_obj

        # set the entry in the object column to a new object index
        batch[i_obj, object_domain] = new_object_index

        # loop over types that interact with the object
        for column, interaction in interation_of_interest:   
            #find out the string of entities interacting here
            old_int = gold_tuple[column]
            
            if not old_int: #if is 0, i.e. unknown
                new_index = 0
            else:
                old_string = Int2String[ old_int ]
            
                
                # replace the old object string with the new object string
                new_string = old_string.replace(old_object_string.split(">>")[1], obj_string.split(">>")[1])            
                # get index corresponding to new string (0 if not seen yet)
                new_index = String2Int[new_string]                
            
            # set entry in batch at current interaction type to this value
            batch[i_obj, column] = new_index
            
            
    return batch, gold_index
        
    
    
    
    




# possible metrics: "RR", "HITS@x" where x is a positive integer, eg. "HITS@10"
def compute_metric(predicted_scores, gold_index, metric = "RR"):
    ranking_permutation = np.argsort(-predicted_scores)
    predicted_rank = np.where(ranking_permutation == gold_index)[0][0]
    
    if metric == "RR":
        value = 1.0/ (predicted_rank + 1.0) #indexing starts at 0, hence add 1
    elif "HITS" in metric:
        R = int(metric.strip("HITS@"))
        value = int(predicted_rank + 1 < R)
    else:
        raise(NameError, "Wrong name for metric")
     
    return value
 
   
# possible metrics: "RR", "HITS@x" where x is a positive integer, eg. "HITS@10"
def compute_filtered_metric(predicted_scores, gold_index, other_gold_indices,
                            metric = "RR"):
    
    #eliminate other entries from ranking, but not the gold index of interest
    elim = other_gold_indices.tolist()
    elim.remove(gold_index)    
    left_over= [i for i in range(predicted_scores.shape[0]) if not (i in elim)]

    new_gold_index = left_over.index(gold_index)

    # now compute gold rank only among the left over entries of the batch
    filtered_scores = predicted_scores[left_over]
    rank_perm = np.argsort(-filtered_scores)
    g_rank = np.where(rank_perm == new_gold_index)[0][0]
    
    if metric == "RR":
        filtered_value = 1.0/ (g_rank + 1.0)   #indexing starts at 0, hence add 1
    elif "HITS@" in metric:
        R = int(metric.strip("HITS@"))
        filtered_value = int(g_rank + 1.0 <= R)
        #print "g_rank:", g_rank
    else:
        raise(NameError, "Wrong name for metric")
      
    return filtered_value
    
    
    
# Testing ranking batch creator
if 0:    
    string_tuples = [["ent1", "rel1", "ent2"],
                     ["ent3", "rel2", "ent4"],
                     ["ent5", "rel3", "ent6"],
                     ["ent7", "rel4", "ent8"],
                     ["ent9", "rel5", "ent10"],
                     ["ent11", "rel6", "ent12"]]

    # the entry in the tuple, and the interaction going on there.
    interactions = []#[(3,(0,1)), (4,(1,2))]
    
    String2Int, Int2String = Dict.DefineGlobalDict(string_tuples, interactions)
    
    
    
    object_domain=2
    #gold_tuple = (1,2,3,4,5)
    gold_tuple = (String2Int["0>>ent3"], String2Int["1>>rel2"], String2Int["2>>ent4"]) 
    
    batch, gold_index = get_ranking_batch(gold_tuple, interactions, object_domain, 
                      String2Int, Int2String)
    
    
    
    
    
    
    
# Testing metrics
if 0:
    predicted_scores = np.random.uniform(0.0, 1.0, [10])
    gold_index = 2
    other_gold_indices = np.array([0,2,3])
    
    RR = compute_metric(predicted_scores, gold_index, "RR")
    H6 = compute_metric(predicted_scores, gold_index, "HITS@6")
    fRR = compute_filtered_metric(predicted_scores, gold_index, 
                                  other_gold_indices, "RR")
    fH6 = compute_filtered_metric(predicted_scores, gold_index, 
                                  other_gold_indices, "HITS@6")
                                  
    print (predicted_scores)
    print ("RR:", fRR)
    for i in range(11)[1:]:
        m = compute_filtered_metric(predicted_scores, gold_index, 
                                  other_gold_indices, "HITS@"+str(i))
        print( "HITS@"+str(i), m)
        
    
    
    
    
