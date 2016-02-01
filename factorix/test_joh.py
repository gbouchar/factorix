# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 11:16:39 2016

@author: Johannes
"""

import naga.factorix.learn_factorization as lf
import dataset_reader as dr
import numpy as np
from matplotlib.pyplot import *
from scipy.sparse.linalg import svds
from naga.factorix.scoring import sparse_multilinear_dot_product, \
multilinear_square_product, generalised_multilinear_dot_product



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
            
            print "Evaluation..object ", i_tup, 'of', len(pos_test_tuples)
            feed_eval = {inputs_eval: batch}
            batch_scores = sigmoid( sess.run([pred_eval], feed_dict=feed_eval)[0] )
            
            RR = eva.compute_metric(batch_scores, gold_index, metric = "RR")
            H10 = eva.compute_metric(batch_scores, gold_index, metric = "HITS@10")
            print "RR, H10=", RR, H10
            RRs += [RR]
            H10s += [H10]

    return sum(RRs)/float(len(RRs)), sum(H10s)/float(len(H10s))



 

def sigmoid(x):
    ones = np.ones(np.shape(x))
    return ones/(ones + np.exp(-x))


#def main():

if 1:   

    interactions = [(3,(0,1)), (4,(1,2))]
    #interactions = []
    thresh_mentions = 5
    neg = 2          #negative samples per positive
    rank = 5          #embedding dimensionality
    n_train = 287044
    n_iter = 20

    #choose a dot product to use. # can be 
            #sparse_multilinear_dot_product
            #multilinear_square_product
            #generalised_multilinear_dot_product
    dot_product = generalised_multilinear_dot_product
    
    
    
    # load data & dictionaries
    train, String2Int, Int2String = dr.load_FB15K237_FB('train', None, None,                      
                                                     interactions, thresh_mentions)  
    valid, String2Int, Int2String = dr.load_FB15K237_FB('valid', String2Int, 
                                                   Int2String, interactions)    
    test, String2Int, Int2String = dr.load_FB15K237_FB('test', String2Int, 
                                                    Int2String, interactions)
       
    # in case not the full train data should be used
    train = train[:n_train]
    values = [1.0 for x in range(n_train)]

       
    # sample some random negative validation tuples
    n_test = len(valid)
    valid_negative = np.random.randint(0, 1000 ,[n_test,3+len(interactions)])
    
    
    # initialise embeddings, reserve one extra entry (the first) for unknown
    n_emb = len(Int2String.keys()) + 1
    emb0 = np.random.normal(size=(n_emb, rank)) * 0.1
    
    # initialise the norm scalers
    norm_scalers = np.random.normal(size = [3+len(interactions)]) * 0.1
    
    # set other factorization inputs
    optim = tf.train.AdamOptimizer(learning_rate=0.001)
    scoring = lf.generalised_multilinear_dot_product_scorer
    minibatch_size=1000
    
    # factorize the train tuples, obtain all model parameters
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
    print "Avg Pos Train Prediction", np.mean(prediction_values)  
    print "Avg Random Neg Prediction", np.mean(prediction_values2)  
    print "Avg Pos Test Prediction",  np.mean(prediction_values3)  
    
    # do batch ranking evaluation, compute MRR and HITS@10 on valid set.
    h=100    #h=len(valid)
    #params[0][0,:] = np.zeros([params[0].shape[1]])
    MRR, H10 = ranking_evaluation(valid[:h], prediction_values3[:h], 
                         interactions, 2, String2Int, Int2String, params, dot_product)
    print "MRR:", MRR
    print "HITS@10:", H10
    print "EOF"
  
#main()