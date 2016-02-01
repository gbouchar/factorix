import random
import numpy as np
import tensorflow as tf
import naga.factorix as fx

# from read_arit import read_arit_dialogs, dialog2txt
# from naga.members.guillaume.NeuralPredictor import NeuralPredictor, IndependentSlicer, accuracy


def sparse_multilinear_dot_product(emb, tuples, l2=0):
    """
    Compute the dot product of real vectors at selected embeddings
    Note that this model is called Cannonical Parafac (CP), and corresponds to the "distmult" model in some scientific
    publications on relational database factorization.
    :param emb: embedding matrix of size [n_emb, rank] containing float numbers
    :param tuples: tuple matrix of size [n_t, arity] containing integers
    :param l2: optional l2 regularization strength that is added to the score. If it is different from 0, the function
    returns a pair (pred, l2norm) where pred is the sample prediction, but l2norm is the l2 norm of the selected
    embeddings
    :return: the multilinear dot product between selected embeddings S[i] = sum_j prod_k E[I[i,k],j]

    >>> emb = [[1., 1, 0, 3], [0, 1, 0, 1], [-1, 1, 1, 5]]
    >>> idx = tf.Variable([[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]])
    >>> g = sparse_multilinear_dot_product(emb, idx)
    >>> print(fx.tf_eval(g))
    [  4.   4.  15.  15.   6.   6.]
    """
    emb_sel = tf.gather(emb, tuples)
    pred = tf.reduce_sum(tf.reduce_prod(emb_sel, 1), 1)
    if l2 == 0:  # unregularized prediction ==> returns only the predictions
        return pred
    else:  # l2 regularization of the selected embeddings
        reg = l2 * tf.reduce_sum(tf.square(emb_sel))
        return pred, reg


def multilinear_square_product(emb, tuples, l2=0):
    """
     Compute the square-product of real vectors at selected embeddings.
     This is the sum over all dimensions of the square of summed embedding vectors.
    :param emb: embedding matrix of size [n_emb, rank] containing float numbers
    :param tuples: tuple matrix of size [n_t, arity] containing integers
    :param l2: optional l2 regularization strength that is added to the score. If it is different from 0, the function
    returns a pair (pred, l2norm) where pred is the sample prediction, but l2norm is the l2 norm of the selected
    embeddings
    :return: the multilinear square product between selected embeddings
    S[i] = sum_k ( sum_j  E[I[i,k],j] )^2

    >>> emb = [[12., 0, 0], [0, 1, 0], [-1, 1, 1]]
    >>> idx = tf.Variable([[1,0,0],[1,1,0]])
    >>> g = multilinear_square_product(emb, idx)
    >>> print(fx.tf_eval(g))
    [ 577.  148.]
    """
    emb_sel = tf.gather(emb, tuples)
    pred = tf.reduce_sum(tf.square(tf.reduce_sum(emb_sel, 1)), 1)

    if l2 == 0:  # unregularized prediction ==> returns only the predictions
        return pred
    else:  # l2 regularization of the selected embeddings
        reg = l2 * tf.reduce_sum(tf.square(emb_sel))
        return pred, reg

    
# def generalised_multilinear_dot_product( params, tuples, domain_offsets=[-1,-1,-1]):
def generalised_multilinear_dot_product(params, tuples, l2=0):
    """
     Compute the generalised linear product of real vectors at selected embeddings.
     This is the sum over all dimensions of the square of summed embedding vectors,
     minus a weighted version of the norms of each embedding used.
    :param emb: embedding matrix of size [n_emb, rank] containing float numbers
    :param tuples: tuple matrix of size [n_t, arity] containing integers
    :param l2: optional l2 regularization strength that is added to the score. If it is different from 0, the function
    returns a pair (pred, l2norm) where pred is the sample prediction, but l2norm is the l2 norm of the selected
    embeddings
    :return: the multilinear square product between selected embeddings
    S[i] = sum_k ( sum_j  E[I[i,k],j] )^2

    In the case of domain_offsets = [-1,-1,-1], the multilinear dot prod is recovered.
    #109 - 5 - 34 - 17 = 53
    #(8^2 + 5^2 + 0^2) - 5 - 10 - 17 = 73 - 32 = 41
    >>> domain_offsets = [-1,-2,-1]
    >>> emb = [[4., 1, 0], [2, 1, 0], [-1, 1, 1]]
    >>> params = (emb, domain_offsets)
    >>> idx = tf.Variable([[1,0,0], [1,1,0]])
    >>> g = generalised_multilinear_dot_product(params, idx)
    >>> print(fx.tf_eval(g))
    [ 53.  41.]
    """
    emb, domain_offsets = params
    emb_sel = tf.gather(emb, tuples)
    emb_sum = tf.reduce_sum(emb_sel, 1)
    squares =  tf.square( emb_sum )
    square_score = tf.reduce_sum( squares, 1 )

    squared_norms = tf.reduce_sum( tf.square(emb_sel), 2)

    # norms = tf.mul(norms, domain_offsets)
    # return square_score  #[ 109.   73.]
    # return  tf.reduce_sum( tf.mul(squared_norms, domain_offsets) , 1) #[-56. -32.]
    pred = square_score + tf.reduce_sum(tf.mul(squared_norms, domain_offsets), 1)
    if l2 == 0:  # unregularized prediction ==> returns only the predictions
        return pred
    else:  # l2 regularization of the selected embeddings
        reg = l2 * tf.reduce_sum(tf.square(emb_sel))
        return pred, reg

