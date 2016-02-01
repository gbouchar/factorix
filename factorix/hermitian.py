import numpy as np
import tensorflow as tf
import naga.factorix as fx


def hermitian_tuple_scorer(tuples_var, rank=None, n_emb=None, emb0=None, symmetry_coef=(1.0, 1.0),
                           learn_symmetry_coef=True):
    """
    The Hermitian Scorer can learn embeddings for non-symmetric relations
    :param tuples_var: TensorFlow variable that encodes the tuples as inputs
    :param rank: size of the embeddings, including real and imaginary parts. The complex rank is half of it.
    not needed if emb0 is given
    :param n_emb: number of embeddings (not needed if initial embeddings are given)
    :param emb0: initial embeddings (optional)
    :param symmetry_coef: symmetry coefficient that equals np.inf for symmetric matrices, -np.inf for anti-symmetric
    matrices and a real scalar for other cases.
    :param learn_symmetry_coef: False if the symmetry coefficient is not learned [True by default]
    :return: a pair (scoring TensorFlow graph, parameters). The parameters have the form
     ([n_emd*rank] float matrix, symmetry coef)

    >>> emb = [[1., 1, 0, 3], [0, 1, 0, 1], [-1, 1, 1, 5]]
    >>> tuples_var = tf.Variable([[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]])
    >>> (g, params) = hermitian_tuple_scorer(tuples_var, emb0=emb, symmetry_coef=(1.0, 0.0))
    >>> print(fx.tf_eval(g))  # symmetric form
    [  4.   4.  15.  15.   6.   6.]
    >>> (g, params) = hermitian_tuple_scorer(tuples_var, emb0=emb, symmetry_coef=(0.0, 1.0))
    >>> print(fx.tf_eval(g))  # skewed (anti-symmetric) form
    [-2.  2.  3. -3.  4. -4.]
    >>> (g, params) = hermitian_tuple_scorer(tuples_var, emb0=emb, symmetry_coef=(1.0, 1.0))
    >>> print(fx.tf_eval(g))  # combination of the previous two forms
    [  2.   6.  18.  12.  10.   2.]
    >>> (g, params) = hermitian_tuple_scorer(tuples_var, emb0=emb, symmetry_coef=(0.9, 0.1))
    >>> print(fx.tf_eval(g))  # close to symmetric
    [  3.39999986   3.79999995  13.80000019  13.19999981   5.79999971
       4.99999952]
    """
    emb0 = emb0 if emb0 is not None else np.random.normal(size=(n_emb, rank))
    embeddings = tf.Variable(tf.cast(emb0, 'float32'), 'embeddings')
    symmetry_coef = tf.Variable(symmetry_coef, name='symmetry_coef', trainable=learn_symmetry_coef)
    params = (embeddings, symmetry_coef)
    return sparse_hermitian_scoring(params, tuples_var), params


def sparse_hermitian_product(emb, tuples):
    """
    Compute the Hermitian inner product between selected complex embeddings
    This corresponds to the usual dot product applied on the conjugate of the first vector: <conj(x), y>
    where conj is the complex conjugate (obtained by inverting the imaginary part)
    We consider that the embedding dimension is twice the rank, where the first part is in emb[:,:rk] and
    the imaginary part is in emb[:,rk:].
    It computes
     S[i] = <conj(E[I[i,1]], E[I[i,2]]>
    Usage:
    S = sparse_hermitian_product(E, I):
    :param emb: embedding matrix of size [n_emb, 2 * r] containing float numbers where r is the complex rank
    :param tuples: tuple matrix of size [n_t, 2] containing integers that correspond to the indices of the embeddings
    :return: a pair containing the real and imaginary parts of the Hermitian dot products
    """
    rk = emb.get_shape()[1].value // 2
    emb_re = emb[:, :rk]
    emb_im = emb[:, rk:]
    emb_sel_a_re = tf.gather(emb_re, tuples[:, 0])
    emb_sel_a_im = tf.gather(emb_im, tuples[:, 0])
    emb_sel_b_re = tf.gather(emb_re, tuples[:, 1])
    emb_sel_b_im = tf.gather(emb_im, tuples[:, 1])
    pred_re = tf.reduce_sum(tf.mul(emb_sel_a_re, emb_sel_b_re) + tf.mul(emb_sel_a_im, emb_sel_b_im), 1)
    pred_im = tf.reduce_sum(tf.mul(emb_sel_a_re, emb_sel_b_im) - tf.mul(emb_sel_a_im, emb_sel_b_re), 1)
    return pred_re, pred_im


def sparse_hermitian_scoring(params, tuples):
    """
    TensorFlow operator that scores tuples by the dot product of their complex embeddings.

    It is the same a the sparse_multilinear_dot_product function, but uses complex embeddings instead. The complex
    embeddings are of size 2 * R where R is the complex
    dimension. They are encoded such that the first columns correspond to the real part, and the last R correspond to
    the imaginary part. The result of this function is a length-N vector with values:

        S[i] = alpha_0 * Re(sum_j <E[I[i,1],j], E[I[i,2],j]>) + alpha_1 * Im(sum_j <E[I[i,1],j], E[I[i,2],j]>))

    Where:
        - I is the tuple tensor of integers with shape (T, 2)
        - E is the N * 2R tensor of complex embeddings (R first columns: real part, the last R columsn: imaginary part)
        - alpha_0 and alpha_1 are the symmetry coefficients

    :param params: tuple (emb, symm_coef) containing:
        - emb: a real tensor of size [N, 2*R] containing the N rank-R embeddings by row (real part in the rank first R
        columns, imaginary part in the last R columns)
        - the 2-tuple (s0, s1) of symmetry coefficients (or complex-to-real projection coefficients) that are used to
        transform the complex result of the dot product into a real number, as used by most statistical models (e.g.
        mean of a Gaussian or Poisson distributions, natural parameter of a Bernouilli distribution). The conversion
        from complexto real is a simple weighted sum: results = s0 * Re(<e_i, e_j>) + s1 * Im(<e_i, e_j>
    :param tuples: tuple matrix of size [T, 2] containing T pairs of integers corresponding to the indices of the
        embeddings.
    :return: Hermitian dot products of selected embeddings
    >>> emb = (tf.Variable([[1., 1, 0, 3], [0, 1, 0, 1], [-1, 1, 1, 5]]), (0.0, 1.0))
    >>> idx = tf.Variable([[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]])
    >>> g = sparse_hermitian_scoring(emb, idx)
    >>> print(fx.tf_eval(g))
    [-2.  2.  3. -3.  4. -4.]
    """
    emb, symmetry = params
    pred_re, pred_im = sparse_hermitian_product(emb, tuples)
    return symmetry[0] * pred_re + symmetry[1] * pred_im


def hermitian_dot(u, v):
    """
    Hermitian dot product between multiple embeddings given by rows.
    :param u: first matrix of n embeddings
    :param v: second matrix of m embeddings
    :param alpha: weight of the real part in the response
    :return: a pair of n * m matrix of Hermitian inner products between all vector combinations:
        - Re(<u_i, v_j>) for the first output
        - Im(<u_i, v_j>) for the second output
    >>> emb = np.array([[1., 1, 0, 3], [0, 1, 0, 1], [-1, 1, 1, 5]])
    >>> print(hermitian_dot(emb, emb.T))
    (array([[ 11.,   4.,  15.],
           [  4.,   2.,   6.],
           [ 15.,   6.,  28.]]), array([[ 0., -2.,  3.],
           [ 2.,  0.,  4.],
           [-3., -4.,  0.]]))
    """
    print(u.shape, v.shape)
    rk = u.shape[1] // 2
    u_re = u[:, :rk]
    u_im = u[:, rk:]
    v_re = v[:rk, :]
    v_im = v[rk:, :]
    return np.dot(u_re, v_re) + np.dot(u_im, v_im), np.dot(u_re, v_im) - np.dot(u_im, v_re)


def sparse_dot_product0(emb, tuples, use_matmul=True, output_type='real'):
    """
    Compute the dot product of complex vectors.
    It uses complex vectors but tensorflow does not optimize in the complex space (or there is a bug in the gradient
    propagation with complex numbers...)
    :param emb: embeddings
    :param tuples: indices at which we compute dot products
    :return: scores (dot products)
    """
    n_t = tuples.get_shape()[0].value
    rk = emb.get_shape()[1].value
    emb_sel_a = tf.gather(emb, tuples[:, 0])
    emb_sel_b = tf.gather(emb, tuples[:, 1])
    if use_matmul:
        pred_cplx = tf.squeeze(tf.batch_matmul(
                tf.reshape(emb_sel_a, [n_t, rk, 1]),
                tf.reshape(emb_sel_b, [n_t, rk, 1]), adj_x=True))
    else:
        pred_cplx = tf.reduce_sum(tf.mul(tf.conj(emb_sel_a), emb_sel_b), 1)
    if output_type == 'complex':
        return pred_cplx
    elif output_type == 'real':
        return tf.real(pred_cplx) + tf.imag(pred_cplx)
    elif output_type == 'real':
        return tf.abs(pred_cplx)
    elif output_type == 'angle':
        raise NotImplementedError('No argument or inverse-tanh function for complex number in Tensorflow')
    else:
        raise NotImplementedError()


def test_hermitian_tuple_scorer():
    emb = [[1., 1, 0, 3], [0, 1, 0, 1], [-1, 1, 1, 5]]
    tuples_var = tf.Variable([[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]])
    (g, params) = hermitian_tuple_scorer(tuples_var, emb0=emb, symmetry_coef=(1.0, 0.0))
    print(fx.tf_eval(g))
    (g, params) = hermitian_tuple_scorer(tuples_var, emb0=emb, symmetry_coef=(0.0, 1.0))
    print(fx.tf_eval(g))
    (g, params) = hermitian_tuple_scorer(tuples_var, emb0=emb, symmetry_coef=(0.5, 0.5))
    print(fx.tf_eval(g))
    (g, params) = hermitian_tuple_scorer(tuples_var, emb0=emb, symmetry_coef=(0.9, 0.1))
    print(fx.tf_eval(g))  # close to symmetric
    (g, params) = hermitian_tuple_scorer(tuples_var, emb0=emb, symmetry_coef=(0.1, 0.9))
    print(fx.tf_eval(g))  # close to anti-symmetric


def test_tuples_factorization_rectangular_matrix(demo=False):
    """
    In this test, we compare the solution of the factorization given by an exact SVD and the solution given by
    the factorize_tuple function because with fully-observed matrix data and quadratic loss the solutions should match
    exactly.
    :param demo: True for demo mode where explanations are printed in the standard output. Otherwise a test is run.
    :return: Nothing
    """
    from naga.factorix.learn_factorization import factorize_tuples
    from scipy.sparse.linalg import svds
    # Create initial data
    n = 7  # number of rows
    m = 6  # number of column
    rk0 = 4  # size of the embeddings
    rk = 4  # embedding size for the model
    noise = 1  # noise level
    oracle_init = False  # do we initialize at the exact solution?
    u0_mat = np.random.randn(n, rk0)
    v0_mat = np.random.randn(m, rk0)
    y_mat = np.random.randn(n, m) * noise + np.dot(u0_mat, v0_mat.transpose())

    # svd solution
    u1_mat, d1_vec, v1_matt = svds(y_mat, rk)
    v1_mat = v1_matt.transpose()
    d1_diag_matrix = np.zeros((rk, rk))
    for i in range(rk):
        d1_diag_matrix[i, i] = np.sqrt(d1_vec[i])
    x_mat_est1 = np.dot(np.dot(u1_mat, np.square(d1_diag_matrix)), v1_matt)

    if demo:
        print('We obtained a first exact solution by Singular Value Decomposition')
        print('The difference between the observation matrix and the estimated solution is:')
        print(np.linalg.norm(x_mat_est1-y_mat))
        print()

    # #### sgd solution ####

    # conversion to tuples
    indices = [[i, n + j] for i in range(n) for j in range(m)]
    values = [y_mat[i, j] for i in range(n) for j in range(m)]

    # initialization
    if oracle_init:
        emb0_u_re = np.dot(u1_mat[:, :2], d1_diag_matrix[:2, :2])
        emb0_u_im = np.dot(u1_mat[:, 2:], d1_diag_matrix[2:, 2:])
        emb0_v_re = np.dot(v1_mat[:, :2], d1_diag_matrix[:2, :2])
        emb0_v_im = np.dot(v1_mat[:, 2:], d1_diag_matrix[2:, 2:])
        emb0_u = 1.0 * (np.concatenate([emb0_u_im, -emb0_u_re], axis=1) + np.concatenate([emb0_u_re, emb0_u_im], axis=1))
        emb0_v = np.concatenate([emb0_v_re, emb0_v_im], axis=1)
        emb0 = np.concatenate([emb0_u, emb0_v], axis=0)
    else:  # random initialization
        emb0 = np.random.normal(size=(n + m, rk)) * 0.1
    # x_mat_init = fx.hermitian_dot(emb0[:n], emb0[n:])
    x_mat_init = np.dot(emb0[:n], emb0[n:].T)

    # choose an optimizer (Adam deems to be the most reliable)
    # optim = tf.train.GradientDescentOptimizer(learning_rate=1.)
    optim = tf.train.AdamOptimizer(learning_rate=0.1)
    # optim = tf.train.RMSPropOptimizer(learning_rate=1., decay=0.1)
    # optim = tf.train.AdagradOptimizer(learning_rate=.1)
    # optim = tf.train.FtrlOptimizer(1.0, -0.5, l2_regularization_strength=0.0, initial_accumulator_value=1e-8)

    # choose a scoring function (for rectangular matrices, the standard or the complex dot products work the same)
    scoring = lambda inputs: hermitian_tuple_scorer(inputs, rank=rk, n_emb=n + m, emb0=emb0, symmetry_coef=(1.0, 1.0),
                                                    learn_symmetry_coef=True)
    # scoring = None

    # optimization (we specify optional parameters, but the values by default could work as well)
    u2, coefs = factorize_tuples((indices, values), rk, emb0=emb0, n_iter=300, tf_optim=optim, scoring=scoring)
    # recover the matrix based on the hermitian dot product of the embeddings
    x_mat_est2_cplx = hermitian_dot(u2[:n, :], u2[n:, :].T)  # fx.clpx2real(fx.hermitian_dot(u2[:n], u2[n:]))
    x_mat_est2 = x_mat_est2_cplx[0] * coefs[0] + x_mat_est2_cplx[1] * coefs[1]
    if demo:
        print(x_mat_est2.shape, x_mat_init)
        print('We computed an estimator by minimizing the square loss on the tuples extracted from the matrix')
        print('The difference between the initial solution and the estimated solution is:')
        print(np.linalg.norm(x_mat_est2-x_mat_init))
        print('The difference between the observations and the estimated solution is:')
        print(np.linalg.norm(y_mat-x_mat_est2))
        print('The difference between the exact solution and the estimated solution is:')
        print(np.linalg.norm(x_mat_est1-x_mat_est2))
        print('Symmetry coefficients: ', coefs)
    assert(np.linalg.norm(x_mat_est1-x_mat_est2) < 1e-3)

if __name__ == '__main__':
    test_hermitian_tuple_scorer()
