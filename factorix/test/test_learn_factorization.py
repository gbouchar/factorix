import numpy as np

from hermitian import hermitian_tuple_scorer, hermitian_dot
from toy_examples import toy_factorization_problem, svd_factorize_matrix
from learn_factorization import factorize_tuples
from dataset_reader import mat2tuples
from samplers import tuple_sampler


def test_tuples_factorization_rectangular_matrix(verbose=False, hermitian=False):
    """
    In this test, we compare the solution of the factorization given by an exact SVD and the solution given by
    the factorize_tuple function because with fully-observed matrix data and quadratic loss the solutions should match
    exactly.

    Args:
        verbose: True for demo mode where explanations are printed in the standard output. Otherwise a test is run.
        hermitian: Do we use the Hermitian product? (Hermitian product is asymmetric)

    Returns:
        Nothing

    """
    y_mat = toy_factorization_problem(n=7, m=6, rk=4, noise=1)
    n, m = y_mat.shape
    rk = 4  # embedding size for the model

    u, v = svd_factorize_matrix(y_mat, rank=4, return_embeddings=True)  # exact svd solution
    emb0 = np.concatenate([u, v])
    x_mat_est1 = u.dot(v.T)

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
    u, v = svd_factorize_matrix(y_mat, rank=4, return_embeddings=True)  # exact svd solution
    emb0 = np.concatenate([u, v])
    x_mat_est1 = u.dot(v.T)
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
    u, v = svd_factorize_matrix(y_mat, rank=4, return_embeddings=True)  # exact svd solution
    # emb0 = np.concatenate([u, v])
    x_mat_est1 = u.dot(v.T)

    u_dense = factorize_tuples(tuples_dense, 4, emb0=None, n_iter=500, verbose=verbose, eval_freq=100)[0]
    # rr_sampler = all_tensor_indices(tuples_sparse, batch_size=n * n, prop_negatives=1)
    # sampler = tuple_sampler(tuples_sparse, minibatch_size=10, prop_negatives=0.5, idx_ranges=[(0, n), (n, 2 * n)])
    # u_sparse = factorize_tuples(sampler, 4, emb0=None, n_iter=500, verbose=verbose, eval_freq=100)[0]
    x_mat_est_dense = np.dot(u_dense[:7], u_dense[7:].T)
    # x_mat_est_sparse = np.dot(u_sparse[:7], u_sparse[7:].T)
    if verbose:
        print(np.linalg.norm(x_mat_est1 - x_mat_est_dense))
        # print(np.linalg.norm(x_mat_est1 - x_mat_est_sparse))
        # print(np.linalg.norm(x_mat_est_dense - x_mat_est_sparse))
    else:
        assert(np.linalg.norm(x_mat_est1 - x_mat_est_dense) < 1e-3)
        # assert(np.linalg.norm(x_mat_est1 - x_mat_est_sparse) < 1e-3)

if __name__=="__main__":
    test_tuples_factorization_rectangular_matrix(verbose=True)
    test_learn_factorization(verbose=True)
    test_sparse_factorization(verbose=True)
