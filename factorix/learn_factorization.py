import numpy as np
from collections import namedtuple
import warnings

import tensorflow as tf

from factorix.losses import loss_func, get_loss_type
from factorix.dataset_reader import mat2tuples
from factorix.toy_examples import toy_factorization_problem, svd_factorize_matrix
from factorix.samplers import positive_and_negative_tuple_sampler, simple_tuple_generator
from factorix.scoring import multilinear_tuple_scorer



def factorize_tuples(tuples, rank=2, arity=None, minibatch_size=100, n_iter=1000, eval_freq=100,
                     loss_types=('quadratic',),
                     negative_prop=0.0, n_emb=None,
                     minibatch_generator=None, verbose=True,
                     scoring=None, negative_sample=False, tf_optim=None, emb0=None, n_ent = None,
                     bigram = False, dictionaries = None):

    """
    Factorize a knowledge base using a TensorFlow model
    :param tuples: list of tuples representing a knowledge base. Types can be used
    :param scoring: TensorFlow operator accepting 2 tensors as input and one as output ():
        - input 1 is the embedding tensor. It has size [n_ent, rank] and contains float values
        - input 2 is a tensor containing tuples of integer from a minibatch. It has size [minibatch_size, order]
        and values range from 0 to n_ent-1
        - output is a list of continuous values
    :param tf_optim: TensorFlow model performing the optimization
    :return: embeddings

    Note about sparse_hermitian_product:
    To recover the predictions, you must average the real and imaginary part because it is learn using this formula.
    See the sparse_hermitian_product function with the 'real' default option. This is simpler in the real case when we
    use the multilinear function: we would replace hermitian_dot by dot(embeddings, embeddings.transpose())

    # demo of a rectangular matrix factorization with square loss:
    >>> y_mat = toy_factorization_problem(n=7, m=6, rk=4, noise=1)
    >>> emb0 = svd_factorize_matrix(y_mat, rank=4)  # exact svd solution
    >>> u2 = factorize_tuples(mat2tuples(y_mat), 4, emb0=emb0, n_iter=500, verbose=False)[0]
    >>> x_mat_est2 = np.dot(u2[:7], u2[7:].T)  # the initial matrix
    >>> np.linalg.norm(emb0[:7].dot(emb0[7:].T)-x_mat_est2) < 1e-3
    True


    # demo of a symmetric square matrix factorization with square loss:
    # >>> n = 5
    # >>> mat = random_symmetric_real_matrix(n, rank=4)
    # >>> eig_sol = eig_approx(mat, rank=2)
    # >>> embeddings = factorize_tuples(mat2tuples(mat, common_types=True), rank=2, verbose=False, n_iter=100)
    # >>> h = hermitian_dot(embeddings, embeddings)
    # >>> sgd_sol = 0.5 * (h[0] + h[1])
    # >>> np.linalg.norm(eig_sol - sgd_sol)<1e3
    # True
    """

    if isinstance(tuples, tuple) and len(tuples) == 2:
        raise BaseException
        warnings.warn('Providing tuples as (inputs, outputs) is deprecated. '
                      'Use [(input_1, output_1), ..., (input_n, output_n)] instead'
                      'You should provide them as zip(inputs, outputs)')
        tuples = [x for x in zip(tuples[0], tuples[1])]

    if scoring is None:
        if n_emb is None:
            n_emb = np.max([np.max(x) for x, y in tuples]) + 1

    # the TensorFlow optimizer
    tf_optim = tf_optim if tf_optim is not None else tf.train.AdamOptimizer(learning_rate=0.1)

    if isinstance(tuples, list):
        inputs, outputs, minibatch_generator = simple_tuple_generator(tuples, minibatch_size, n_iter, eval_freq,
                                                                  negative_prop, n_ent, bigram, dictionaries)

    # the scoring function is usually a dot product between embeddings
    if scoring is None:
        preds, params = multilinear_tuple_scorer(inputs, rank=rank, n_emb=n_emb, emb0=emb0)
        #preds, params = multilinear_tuple_scorer(inputs, rank=rank, n_emb=n_emb, emb0=emb0)
    
    # elif scoring == generalised_multilinear_dot_product_scorer:  # commented because it can be done externally
    #     preds, params = scoring(inputs, rank=rank, n_emb=n_emb, emb0=emb0,
    #                             norm_scalers=norm_scalers)
    else:
        preds, params = scoring(inputs, rank=rank, n_emb=n_emb, emb0=emb0)

    # Minimize the loss
    loss_ops = {}
    train_ops = {}
    for m in loss_types:
        loss_ops[m] = tf.reduce_mean(loss_func(preds, outputs, m))
        train_ops[m] = tf_optim.minimize(loss_ops[m])

    # Launch the graph.
    with tf.Session() as sess:  # we close the session at the end of the training
        sess.run(tf.initialize_all_variables())
        for epoch, eval_step, (minibatch_inputs, minibatch_outputs, minibatch_type) in minibatch_generator():
            feed = {inputs: minibatch_inputs, outputs: minibatch_outputs}
            if eval_step:
                train_losses = {}
                for m in loss_types:
                    _, train_losses[m] = sess.run([train_ops[m], loss_ops[m]], feed_dict=feed)
                if verbose:
                    print(epoch, train_losses)
            else:
                sess.run(train_ops[minibatch_type], feed_dict=feed)
        final_params = sess.run(params)
    return final_params


if __name__ == '__main__':
    # test_tuples_factorization_rectangular_matrix(verbose=True, hermitian=False)
    # test_tuples_factorization_rectangular_matrix(verbose=True, hermitian=True)
    # test_learn_factorization(verbose=True)
    import factorix.test.test_learn_factorization as t
    t.test_tuples_factorization_rectangular_matrix(verbose=True, hermitian=False)
    # t.test_tuples_factorization_rectangular_matrix(verbose=True, hermitian=True)
    t.test_learn_factorization(verbose=True)
    t.test_sparse_factorization(verbose=True)
