import random
import numpy as np
import tensorflow as tf

from naga.shared.tf_addons import tf_eval

# from read_arit import read_arit_dialogs, dialog2txt
# from naga.members.guillaume.NeuralPredictor import NeuralPredictor, IndependentSlicer, accuracy


def multilinear_tuple_scorer(tuples_var, rank=None, n_emb=None, emb0=None):
    emb0 = emb0 if emb0 is not None else np.random.normal(size=(n_emb, rank))
    embeddings = tf.Variable(tf.cast(emb0, 'float32'), 'embeddings')
    return multilinear(embeddings, tuples_var), (embeddings,)


def generalised_multilinear_dot_product_scorer(tuples_var, rank=None, n_emb=None,
                                               emb0=None, norm_scalers = None):
    emb0 = emb0 if emb0 is not None else np.random.normal(size=(n_emb, rank))
    norm_scalers = norm_scalers if norm_scalers is not None \
        else np.random.normal( size=(len(tuples_var[0]) ) )

    embeddings = tf.Variable(tf.cast(emb0, 'float32'), 'embeddings')
    n_scalers = tf.Variable(tf.cast(norm_scalers, 'float32'), 'norm_scalers')
    return generalised_multilinear_dot_product( (embeddings, n_scalers), tuples_var, l2=norm_scalers), \
           (embeddings, n_scalers)


def multilinear(emb, tuples, l2=0):
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
    >>> g = multilinear(emb, idx)
    >>> print(tf_eval(g))
    [  4.   4.  15.  15.   6.   6.]
    """
    emb_sel = tf.gather(emb, tuples)
    
    pred = tf.reduce_sum(tf.reduce_prod(emb_sel, 1), 1)
    if l2 == 0:  # unregularized prediction ==> returns only the predictions
        return pred
    else:  # l2 regularization of the selected embeddings
        reg = l2 * tf.reduce_sum(tf.square(emb_sel))
        return pred, reg


def multilinear_grad(emb: tf.Tensor, tuples: tf.Tensor, score=False, slot_dim=0) -> tf.Tensor:
    tuple_shape = [d.value for d in tuples.get_shape()]
    # if len(tuple_shape) > 2:
    #     n = np.prod(tuple_shape[:-1])
    #     tuples = tf.reshape(tuples, (n, -1))
    # n = tuples.get_shape()[0].value
    order = tuples.get_shape()[2].value
    rank = emb.get_shape()[1].value
    if order == 2:
        if slot_dim == 0:
            if score:
                emb_sel = tf.gather(emb, tuples)
                grad_score = tf.reshape(emb_sel[:, :, 1, :], tuple_shape[:-1] + [rank])
                prod = tf.reduce_prod(emb_sel, 2)
                preds = tf.reshape(tf.reduce_sum(prod, 2), tuple_shape[:-1])
                return grad_score, preds
    raise NotImplementedError('Todo')
                # grad_score0 = tf.reverse(emb_sel, [False, True, False])  # reverse the row and column embeddings
    #         prod = tf.reduce_prod(emb_sel, 1)
    #         preds = tf.reshape(tf.reduce_sum(prod, 1), tuple_shape[:-1])
    #
    #     preds = tf.reshape(tf.reduce_sum(prod, 1), tuple_shape[:-1])
    # else:  # derivative of a product
    #     prod = tf.reduce_prod(emb_sel, 1)
    #     grad_score0 = tf.tile(tf.reshape(prod, (n, 1, rank)), (1, order, 1)) / emb_sel
    # grad_score = tf.reshape(grad_score0, tuple_shape + [rank])
    # if score:
    #         prod = tf.reduce_prod(emb_sel, 1)
    #     preds = tf.reshape(tf.reduce_sum(prod, 1), tuple_shape[:-1])
    #     return grad_score, preds
    # else:
    #     return grad_score


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
    >>> print(tf_eval(g))
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
    >>> print(tf_eval(g))
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


#
# def multilinear_tuple_scorer(tuples_var, rank=None, n_emb=None, emb0=None):
#     emb0 = emb0 if emb0 is not None else np.random.normal(size=(n_emb, rank))
#     embeddings = tf.Variable(tf.cast(emb0, 'float32'), 'embeddings')
#     return multilinear(embeddings, tuples_var), (embeddings,)
#
#
# def generalised_multilinear_dot_product_scorer(tuples_var, rank=None, n_emb=None,
#                                                emb0=None, norm_scalers = None):
#     emb0 = emb0 if emb0 is not None else np.random.normal(size=(n_emb, rank))
#     norm_scalers = norm_scalers if norm_scalers is not None \
#         else np.random.normal( size=(len(tuples_var[0]) ) )
#
#     embeddings = tf.Variable(tf.cast(emb0, 'float32'), 'embeddings')
#     n_scalers = tf.Variable(tf.cast(norm_scalers, 'float32'), 'norm_scalers')
#     return generalised_multilinear_dot_product( (embeddings, n_scalers), tuples_var, l2=norm_scalers), \
#            (embeddings, n_scalers)
# =======