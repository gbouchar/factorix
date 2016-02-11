import random
import numpy as np
import tensorflow as tf
from factorix.scoring import sparse_multilinear_dot_product, \
multilinear_square_product, generalised_multilinear_dot_product
from factorix.losses import loss_func, get_loss_type
import warnings

# from read_arit import read_arit_dialogs, dialog2txt
# from naga.members.guillaume.NeuralPredictor import NeuralPredictor, IndependentSlicer, accuracy


def multilinear_tuple_scorer(tuples_var, rank=None, n_emb=None, emb0=None):
    emb0 = emb0 if emb0 is not None else np.random.normal(size=(n_emb, rank))
    embeddings = tf.Variable(tf.cast(emb0, 'float32'), 'embeddings')
    return sparse_multilinear_dot_product(embeddings, tuples_var), (embeddings,)


def generalised_multilinear_dot_product_scorer(tuples_var, rank=None, n_emb=None,
                                               emb0=None, norm_scalers = None):
    emb0 = emb0 if emb0 is not None else np.random.normal(size=(n_emb, rank))
    norm_scalers = norm_scalers if norm_scalers is not None \
        else np.random.normal( size=(len(tuples_var[0]) ) )

    embeddings = tf.Variable(tf.cast(emb0, 'float32'), 'embeddings')
    n_scalers = tf.Variable(tf.cast(norm_scalers, 'float32'), 'norm_scalers')
    return generalised_multilinear_dot_product( (embeddings, n_scalers), tuples_var, l2=norm_scalers), \
           (embeddings, n_scalers)


def factorize_tuples(tuples, rank=2, arity=None, minibatch_size=100, n_iter=1000, eval_freq=100,
                     loss_types=('quadratic',),
                     negative_prop=0.0, n_emb=None,
                     minibatch_generator=None, verbose=True,
                     scoring=None, negative_sample=False, tf_optim=None, emb0=None):

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
    use the sparse_multilinear_dot_product function: we would replace hermitian_dot by dot(emb, emb.transpose())

    # demo of a rectangular matrix factorization with square loss:
    >>> y_mat = toy_factorization_problem(n=7, m=6, rk=4, noise=1)
    >>> x_mat_est1, emb0 = svd_factorize_matrix(y_mat, rank=4)  # exact svd solution
    >>> u2 = factorize_tuples(mat2tuples(y_mat), 4, emb0=emb0, n_iter=500, verbose=False)[0]
    >>> x_mat_est2 = np.dot(u2[:7], u2[7:].T)  # the initial matrix
    >>> np.linalg.norm(x_mat_est1-x_mat_est2) < 1e-3
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

        warnings.warn('Providing tuples as (inputs, outputs) is deprecated. '
                      'Use [(input_1, output_1), ..., (input_n, output_n)] instead'
                      'You should provide them as zip(inputs, outputs)')
        tuples = [x for x in zip(tuples[0], tuples[1])]

    if scoring is None:
        if n_emb is None:
            n_emb = np.max([np.max(x) for x, y in tuples]) + 1

    # the TensorFlow optimizer
    tf_optim = tf_optim if tf_optim is not None else tf.train.AdamOptimizer(learning_rate=0.1)

    inputs, outputs, minibatch_generator = simple_tuple_generator(tuples, minibatch_size, n_iter, eval_freq,
                                                                  negative_prop)

    # the scoring function is usually a dot product between embeddings
    if scoring is None:
        preds, params = multilinear_tuple_scorer(inputs, rank=rank, n_emb=n_emb, emb0=emb0)
    # elif scoring == generalised_multilinear_dot_product_scorer:  # commented because it can be done externally
    #     preds, params = scoring(inputs, rank=rank, n_emb=n_emb, emb0=emb0,
    #                             norm_scalers=norm_scalers)
    else:
        preds, params = scoring(inputs)

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


def simple_tuple_generator(tuples, minibatch_size, n_iter, eval_freq, negative_prop):
    # the generator of minibatches
    loss_type = get_loss_type(tuples[0][1])  # takes the first tuple as a type example
    minibatch_generator, n_emb, arity,  minibatch_size = \
        tuples_minibatch_generator(tuples, minibatch_size=minibatch_size, n_iter=n_iter, eval_freq=eval_freq,
                                       negative_prop=negative_prop, loss_type=loss_type)

    # model: inputs, outputs and parameters
    inputs = tf.placeholder("int32", [(1+negative_prop) * minibatch_size, arity])
    outputs = tf.placeholder("float32", [(1+negative_prop) * minibatch_size])
    # inputs = tf.placeholder("int32", [minibatch_size, arity])
    # outputs = tf.placeholder("float32", [minibatch_size])
    return inputs, outputs, minibatch_generator


def tuples_minibatch_generator(tuples, minibatch_size=100, n_iter=1000, eval_freq=50, negative_prop = 0.0, loss_type=0):
    train_inputs = np.array([x for x, y in tuples])
    train_outputs = np.array([y for x, y in tuples])
    arity = train_inputs.shape[1]
    n_ent = np.max(train_inputs) + 1
    n_t = train_inputs.shape[0]
    minibatch_size = min(minibatch_size, n_t)  # for small datasets
    minibatch_type = loss_type

    if negative_prop > 0:  # Pre-allocation, avoid memory alloc at each batch generation
        new_train_inputs = np.empty( (minibatch_size * (negative_prop + 1), train_inputs.shape[1]))
        new_train_outputs = np.empty( minibatch_size * (negative_prop + 1))

    def next_minibatch():  # a generator function
        # nonlocal train_inputs, train_outputs, minibatch_size, n_iter, eval_freq, n_t
        # global train_inputs, train_outputs, minibatch_size, eval_freq, n_t
        epoch = 0
        while epoch < n_iter:
            minibatch_indices, n_rem = create_minibatch_indices(n_t, minibatch_size)
            for ids in minibatch_indices:
                if negative_prop > 0:  # Negative generation

                    this_batch_size = len(ids)  # Manage shorter batches (sometimes last ones are shorter)

                    #Pre copying everyting as many times as necessary
                    new_train_inputs[:(this_batch_size*(negative_prop+1)),:] = np.tile(train_inputs[ids,:],(negative_prop + 1,1))
                    new_train_outputs[:(this_batch_size*(negative_prop+1))] = np.tile(train_outputs[ids], negative_prop + 1)
                    #Pre-sample everything, faster
                    rdm_entities = np.random.randint(0, n_ent, this_batch_size * negative_prop)
                    rdm_choices = np.random.random(this_batch_size * negative_prop)

                    for i in range(this_batch_size):
                    #for i in range(np.floor(this_batch_size/float(negative_prop))):
                        
                        for j in range(negative_prop):
                            cur_idx = i * negative_prop + j

                            if rdm_choices[cur_idx] < 0.5:
                                new_train_inputs[this_batch_size + cur_idx,0] = rdm_entities[cur_idx]
                            else:
                                # Here train_inputs.shape[-1]-1 makes negative generation generic for matrices and
                                # tensors: the altered index is either
                                # the first or last dimension to alter the input indexes
                                new_train_inputs[this_batch_size + cur_idx, train_inputs.shape[-1]-1 ] = rdm_entities[cur_idx]

                            new_train_outputs[this_batch_size + cur_idx] = 0.0

                    minibatch_inputs = new_train_inputs[:this_batch_size * (negative_prop +1),:] #truncate arrays in case of shorter batch
                    minibatch_outputs = new_train_outputs[:this_batch_size * (negative_prop +1)]
                else:  # No negative generation
                    minibatch_inputs = train_inputs[ids, :]
                    minibatch_outputs = train_outputs[ids]

                eval_step = epoch % eval_freq == 0 or epoch == n_iter - 1
                yield epoch, eval_step, (minibatch_inputs, minibatch_outputs, minibatch_type)
                epoch += 1

    return next_minibatch, n_ent, arity, minibatch_size

    # if negative_sampling:
    #     neg_inputs, gold_neg = sample_negatives(n_neg, n_ent, arity, supporting_examples=sel_train_inputs,
    # method='uniform')
    #     preds_neg = scoring(emb, neg_inputs)
    #     loss_op_neg = tf.reduce_mean(loss_func(preds_neg, gold_neg, loss_type))
    #     compensation_loss = tf.reduce_mean(loss_func(preds, tf.constant(0, 'float32', [minibatch_size]), loss_type))
    #     loss_op = loss_op_pos + neg_weight * (loss_op_neg - compensation_loss)
    # else:


# def sample_negatives(n, n_ent, order, supporting_examples=None, method='uniform'):
#     gold_neg = tf.constant(0, 'float32', [n_neg])
#     if method == 'uniform':
#         logproba = - order * np.log(float(n_ent))
#         return (tf.random_uniform([n, order], 0, n_ent, 'int32'), gold_neg, logproba),
#     if method == 'uniform_per_dim':
#         sampling_params = tf.random_uniform([n, order], 0, )
#         dim = tf.random_uniform(0, order, dtype='int32')
#         logproba = - order * np.log(float(n_ent))
#         return (tf.random_uniform([n, order], 0, n_ent, 'int32'), gold_neg, logproba, params),
#     else:
#         raise ValueError

def minibatch_sampler(arrays, minibatch_size):
    """A generator that returns random slices of the input data

    The data is a list of arrays that can be indexed in their first dimension (i.e. elt.shape[0] is the same for every
    element elt in the list)

    Args:
        arrays: list of arrays with the same first dimension
        minibatch_size: size of the random slices that are generated

    Returns:
        A generator that gives a list of subarrays (slices) where each element elt has elt.shape[0]=minibatch_size

    """
    def iterator():
        minibatch_indices, n_rem = create_minibatch_indices(arrays[0].shape[0], minibatch_size)
        for ind in minibatch_indices:
            mb = [elt[ind] for elt in arrays]
            yield mb
    return list(iterator())


def minibatch_sampler_old(arrays, minibatch_size, n_iter=1, eval_freq=100):
    """A generator that returns random slices of the input data

    The data is a list of arrays that can be indexed in their first dimension (i.e. elt.shape[0] is the same for every
    element elt in the list)

    Args:
        arrays: list of arrays with the same first dimension
        minibatch_size: size of the random slices that are generated
        n_iter: number of epochs
        eval_freq: evaluation frequency

    Returns:
        A generator that gives a triples (epoch, is_evaluation_step, list_of_subarrays)
        Where:
            - epoch is an integer representing the epoch,
            - is_evaluation_step is a boolean indicating if the evaluation should be done
            - list_of_subarrays is a list of sub-arrays where each element elt has elt.shape[0]=minibatch_size

    """
    eval_step = 0
    for epoch in range(n_iter):
        minibatch_indices, n_rem = create_minibatch_indices(arrays[0].shape[0], minibatch_size)
        for ind in minibatch_indices:
            mb = [a[ind] for a in arrays]
            is_eval_step = (eval_step % eval_freq) == 0
            yield epoch, is_eval_step, tuple(mb)
            eval_step += 1


def create_minibatch_indices(n, minibatch_size):
    """
    :param n: total number of indices from which to pick from
    :param minibatch_size: size of the minibatches (must be lower than n)
    :return: (list of random indices, number of random duplicate indices in the last minibatch to complete it)
    """
    all_indices = np.random.permutation(n)
    n_steps = (n - 1) // minibatch_size + 1
    n_rem = n_steps * minibatch_size - n
    if n_rem > 0:
        inds_to_add = np.random.randint(0, n_rem, size=n_rem)
        all_indices = np.concatenate((all_indices, inds_to_add))
    return np.split(all_indices, n_steps), n_rem


def toy_factorization_problem(n=7, m=6, rk=4, noise=1, square=False, prop_zeros=.9):
    """
    Create a random matrix which is the sum of a low-rank matrix and a entry-wise centered Gaussian noise
    :param n: number of rows
    :param m: number of columns
    :param rk: size of the embeddings
    :param noise: noise level
    :param square: square matrix problem (m is not used, the row embeddings are equal to the column embeddings)
    :return: random low-rank matrix with noise added
    """
    u0_mat = np.random.randn(n, rk)
    v0_mat = np.random.randn(m, rk)
    y_mat = np.random.randn(n, m) * noise + np.dot(u0_mat, v0_mat.transpose())
    if prop_zeros > 0.:
        indices = np.random.randint(0, n * m, int(n * m * prop_zeros))
        y_mat = y_mat.reshape((-1))
        y_mat[indices] = 0.0
        y_mat = y_mat.reshape((n, m))
    return y_mat


def svd_factorize_matrix(y_mat, rank):
    from scipy.sparse.linalg import svds
    u1_mat, d1_vec, v1_matt = svds(y_mat, rank)
    d1_diag_matrix = np.zeros((rank, rank))
    for i in range(rank):
        d1_diag_matrix[i, i] = np.sqrt(d1_vec[i])
    u = np.dot(u1_mat, d1_diag_matrix)
    v = np.dot(v1_matt.T, d1_diag_matrix)
    return np.dot(u, v.T), np.concatenate([u, v], axis=0)


def mat2tuples(y_mat, sparse=False, common_types=False):
    # conversion to tuples
    n, m = y_mat.shape
    if common_types:
        offset = 0
    else:
        offset = n
    if sparse:
        tuples = [([i, offset + j], y_mat[i, j]) for i in range(n) for j in range(m) if abs(y_mat[i, j]) > 1e-8]
    else:
        tuples = [([i, offset + j], y_mat[i, j]) for i in range(n) for j in range(m)]
    return tuples


def test_tuples_factorization_rectangular_matrix(oracle_init=False, verbose=False, hermitian=False):
    """
    In this test, we compare the solution of the factorization given by an exact SVD and the solution given by
    the factorize_tuple function because with fully-observed matrix data and quadratic loss the solutions should match
    exactly.

    Args:
        verbose:
        hermitian:

    Returns:

    """
    """
    :param verbose: True for demo mode where explanations are printed in the standard output. Otherwise a test is run.
    :param oracle_init: Do we initialize at the exact solution?
    :return: Nothing
    """
    from naga.factorix.hermitian import hermitian_tuple_scorer, hermitian_dot
    y_mat = toy_factorization_problem(n=7, m=6, rk=4, noise=1)
    n, m = y_mat.shape
    rk = 4  # embedding size for the model

    x_mat_est1, emb0 = svd_factorize_matrix(y_mat, rank=4)  # exact svd solution

    emb0 = np.random.normal(size=(n + m, rk)) * 0.1

    x_mat_init = np.dot(emb0[:n], emb0[n:].T)  # the initial matrix

    if verbose:
        print('\n\n\nWe obtained a first exact solution by Singular Value Decomposition')
        print('The difference between the observation matrix and the estimated solution is:')
        print(np.linalg.norm(x_mat_est1-y_mat))
        print()

    # conversion to tuples
    indices = [[i, n + j] for i in range(n) for j in range(m)]
    values = [y_mat[i, j] for i in range(n) for j in range(m)]

    if not hermitian:  # the dot product of real vectors
        u2 = factorize_tuples((indices, values), rk, emb0=emb0, n_iter=500)[0]
        x_mat_est2 = np.dot(u2[:n], u2[n:].T)  # the initial matrix
        coefs = None
    else:  # uses the complex numbers to score
        scoring = lambda inputs: hermitian_tuple_scorer(inputs, rank=rk, n_emb=n + m, emb0=emb0,
                                                        symmetry_coef=(1.0, 1.0),
                                                        learn_symmetry_coef=True)
        # optimization (we specify optional parameters, but the values by default could work as well)
        u2, coefs = factorize_tuples((indices, values), rk, emb0=emb0, n_iter=500, scoring=scoring)
        # recover the matrix based on the hermitian dot product of the embeddings
        x_mat_est2_cplx = hermitian_dot(u2[:n, :], u2[n:, :].T)  # fx.clpx2real(fx.hermitian_dot(u2[:n], u2[n:]))
        x_mat_est2 = x_mat_est2_cplx[0] * coefs[0] + x_mat_est2_cplx[1] * coefs[1]

    if verbose:
        print(x_mat_est2.shape, x_mat_init)
        print('We computed an estimator by minimizing the square loss on the tuples extracted from the matrix')
        print('The difference between the initial solution and the estimated solution is:')
        print(np.linalg.norm(x_mat_est2-x_mat_init))
        print('The difference between the observations and the estimated solution is:')
        print(np.linalg.norm(y_mat-x_mat_est2))
        print('The difference between the exact solution and the estimated solution is:')
        print(np.linalg.norm(x_mat_est1-x_mat_est2))
        if hermitian:
            print('Symmetry coefficients: ', coefs)
    else:
        assert(np.linalg.norm(x_mat_est1-x_mat_est2) < 1e-3)


def test_learn_factorization(verbose=False):
    y_mat = toy_factorization_problem(n=7, rk=4, noise=1, square=True)
    x_mat_est1, emb0 = svd_factorize_matrix(y_mat, rank=4)  # exact svd solution
    u2 = factorize_tuples(mat2tuples(y_mat), 4, emb0=None, n_iter=500, verbose=verbose, eval_freq=100)[0]
    x_mat_est2 = np.dot(u2[:7], u2[7:].T)
    if verbose:
        print(np.linalg.norm(x_mat_est1 - x_mat_est2))
    else:
        assert(np.linalg.norm(x_mat_est1 - x_mat_est2) < 1e-3)


def test_sparse_factorization(verbose=False):
    n = 7
    y_mat = toy_factorization_problem(n=n, rk=4, noise=1, square=True, prop_zeros=0.5)
    tuples_dense = mat2tuples(y_mat, sparse=False)
    tuples_sparse = mat2tuples(y_mat, sparse=True)
    if verbose:
        print(y_mat)
        print(tuples_dense)
        print(tuples_sparse)
    x_mat_est1, emb0 = svd_factorize_matrix(y_mat, rank=4)  # exact svd solution

    tuples_dense = tuples_round_robin_sampler(tuples_dense, batch_size=n * n)
    u_dense = factorize_tuples(tuples_dense, 4, emb0=None, n_iter=500, verbose=verbose, eval_freq=100)[0]
    rr_sampler = tuples_round_robin_sampler(tuples_sparse, batch_size=n * n, prop_negatives=1)
    u_sparse = factorize_tuples(rr_sampler, 4, emb0=None, n_iter=500, verbose=verbose, eval_freq=100)[0]
    x_mat_est_dense = np.dot(u_dense[:7], u_dense[7:].T)
    x_mat_est_sparse = np.dot(u_sparse[:7], u_sparse[7:].T)
    if verbose:
        print(np.linalg.norm(x_mat_est1 - x_mat_est_dense))
        print(np.linalg.norm(x_mat_est1 - x_mat_est_sparse))
        print(np.linalg.norm(x_mat_est_dense - x_mat_est_sparse))
    else:
        assert(np.linalg.norm(x_mat_est1 - x_mat_est_dense) < 1e-3)
        assert(np.linalg.norm(x_mat_est1 - x_mat_est_sparse) < 1e-3)

if __name__ == '__main__':
    # test_tuples_factorization_rectangular_matrix(verbose=True, hermitian=False)
    # test_tuples_factorization_rectangular_matrix(verbose=True, hermitian=True)
    # test_learn_factorization(verbose=True)
    test_sparse_factorization(verbose=True)
