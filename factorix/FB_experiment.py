# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 11:16:39 2016

@author: Johannes
"""

print("Before evetything -- FB_experiment.py")

import naga.factorix.learn_factorization as lf
import dataset_reader as dr
import numpy as np
from naga.factorix.scoring import sparse_multilinear_dot_product, \
multilinear_square_product, generalised_multilinear_dot_product


import sys
import evaluation as eva
import tensorflow as tf



# run ranking evaluation and compute MRR and HITS@10
def ranking_evaluation(pos_test_tuples, scores, interactions, object_domain, 
                      String2Int, Int2String, params, dot_product = None):
    
    if dot_product == None:
        dot_product = sparse_multilinear_dot_product
    
    #just to get size of ranking batch
    batch, gold_index = eva.get_ranking_batch(pos_test_tuples[0], 
                                              interactions, object_domain, 
                                              String2Int, Int2String)
                                      
    inputs_eval = tf.placeholder("int32", [len(batch), 3+len(interactions)])
    pred_eval =  dot_product(params, inputs_eval)

    RRs, H10s = [],[]
    with tf.Session() as sess:  
        sess.run(tf.initialize_all_variables())
        for i_tup, tup in enumerate(pos_test_tuples):
            batch, gold_index = eva.get_ranking_batch(tup, interactions, object_domain, 
                                  String2Int, Int2String)
            
            print ("Evaluation..object ", i_tup, 'of', len(pos_test_tuples))
            feed_eval = {inputs_eval: batch}
            batch_scores = sigmoid( sess.run([pred_eval], feed_dict=feed_eval)[0] )
            
            RR = eva.compute_metric(batch_scores, gold_index, metric = "RR")
            H10 = eva.compute_metric(batch_scores, gold_index, metric = "HITS@10")
            print ("RR, H10=", RR, H10)
            RRs += [RR]
            H10s += [H10]

    return sum(RRs)/float(len(RRs)), sum(H10s)/float(len(H10s))



 

def sigmoid(x):
    ones = np.ones(np.shape(x))
    return ones/(ones + np.exp(-x))


def main():   
    print("Staring main...")        
    #define default parameters    
    interactions = [(3,(0,1)), (4,(1,2)), (5,(2,0))]
    thresh_mentions = 6
    neg = 2          #negative samples per positive
    rank = 5          #embedding dimensionality
    #n_train = 287044
    n_iter = 20
    #choose a dot product to use. # can be 
            #sparse_multilinear_dot_product
            #multilinear_square_product
            #generalised_multilinear_dot_product
    dot_product = generalised_multilinear_dot_product
    minibatch_size=1000
    learning_rate = 0.001
    l2 = 0.0
    eval_file_name = 'output.txt'
    

    # use input arguments to define parameters (i.e. not using default above)
    for x in sys.argv:
        if len( x.split("__") ) > 1:
            arg = x.split("__")[1]
        if "interactions" in x:
            interactions = arg
            print("Interactions not defined yet")
            #TODO
        elif "n_iter" in x:
            n_iter = int(x.split("__")[1])
        elif "thresh_mentions" in x:
            thresh_mentions = int(arg)
        elif "rank" in x:
            rank =  int(arg)
        elif "n_train" in x:
            n_train = int(arg)
        elif "neg" in x:
            neg = int(arg)
        elif "minibatch" in x:
            minibatch_size = int(arg)
        elif "learning_rate" in x:
            learning_rate = float(arg)
        elif "L2" in x:
            l2 = float(arg)
        elif "eval_file_name" in x:
            eval_file_name = str(arg)

    
    print("Done parsing input arguments.")    
    # load data & dictionaries
    train, String2Int, Int2String = dr.load_FB15K237_FB('train', None, None,                      
                                                     interactions, thresh_mentions)  
    valid, String2Int, Int2String = dr.load_FB15K237_FB('valid', String2Int, 
                                                   Int2String, interactions)    
    test, String2Int, Int2String = dr.load_FB15K237_FB('test', String2Int, 
                                                    Int2String, interactions)
       
    # in case not the full train data should be used
    #train = train[:n_train]
    n_train = len(train)
    values = [1.0 for x in range(n_train)]

       
    # sample some random negative validation tuples
    n_test = len(valid)
    valid_negative = np.random.randint(0, 1000 ,[n_test,3+len(interactions)])
    print("Done loading data.")
    
    # initialise embeddings, reserve one extra entry (the first) for unknown
    n_emb = len(Int2String.keys()) + 1
    emb0 = np.random.normal(size=(n_emb, rank)) * 0.1
    
    # initialise the norm scalers
    norm_scalers = np.random.normal(size = [3+len(interactions)]) * 0.1
    
    # set other factorization inputs
    optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
   

    scoring = lf.generalised_multilinear_dot_product_scorer
    
    # factorize the train tuples, obtain all model parameters
    # print("Starting factorisation...")
    params = lf.factorize_tuples((train, values), rank, minibatch_size=minibatch_size,
                                 emb0=emb0, n_iter=n_iter, 
                                 negative_prop = neg, 
                                 loss_type = "logistic", 
                                 tf_optim=optim, scoring=scoring,
                                 norm_scalers = norm_scalers)
    
    #define two placeholders to feed in train and valid data for evaluation
    inputs_train = tf.placeholder("int32", [n_train, len(train[0])])
    inputs_valid = tf.placeholder("int32", [len(valid), len(train[0])])

    # define prediction ops for train and valid set    
    pred_train =  dot_product(params, inputs_train)
    pred_valid =  dot_product(params, inputs_valid)
    
    # define the data feeders
    feed1 = {inputs_train: train}
    feed2 = {inputs_valid: valid_negative}
    feed3 = {inputs_valid: valid}    
    

    print("Start evaluation...")
    # obtain predictions for train, valid and negative validation set
    with tf.Session() as sess:  
        sess.run(tf.initialize_all_variables())
        print("Generating predictions... training data")
        prediction_values = sigmoid( sess.run([pred_train], feed_dict=feed1)[0] )
        print("Generating predictions... validation data (negative)")
        prediction_values2 = sigmoid( sess.run([pred_valid], feed_dict=feed2)[0] )
        print("Generating predictions... validation data (positive)")
        prediction_values3 = sigmoid( sess.run([pred_valid], feed_dict=feed3)[0] )
        

    # evaluation: avg prediction among the different tuple sets
    m1 = np.mean(prediction_values)  
    m2 = np.mean(prediction_values2)  
    m3 = np.mean(prediction_values3)  
    print ("Avg Pos Train Prediction", m1)
    print ("Avg Random Neg Prediction", m2)
    print ("Avg Pos Test Prediction",  m3)
    
    # do batch ranking evaluation, compute MRR and HITS@10 on valid set.
    h = 20
    #h=len(valid)
    #params[0][0,:] = np.zeros([params[0].shape[1]])
    MRR, H10 = ranking_evaluation(valid[:h], prediction_values3[:h], 
                         interactions, 2, String2Int, Int2String, params, dot_product)
    print ("MRR:", MRR)
    print ("HITS@10:", H10)
    
    print("Writing results to file " + eval_file_name)
    with open(eval_file_name, 'w') as f:
        for x in sys.argv:
            f.write(x+"\n")
        f.write("---------\n")
        f.write("MRR: "+ str(MRR)+"\n")
        f.write("HITS@10: "+ str(H10)+"\n")
        f.write("Mean train score:" + str(m1)+"\n")
        f.write("Mean val score (neg):" + str(m2)+"\n")
        f.write("Mean val score (pos):" + str(m3)+"\n")
    print("EOF.")




print("in FB_experiment.py -- before main")  
main()
