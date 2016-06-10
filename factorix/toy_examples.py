import numpy as np

np.set_printoptions(precision=3)


def toy_factorization_problem(n=7, m=6, rk=2, noise=1, square=False, prop_zeros=0):
    """
    Create a random matrix which is the sum of a low-rank matrix and a entry-wise centered Gaussian noise

    Args:
        n:number of rows
        m:number of rows
        rk:size of the embeddings
        noise:noise level
        square:square matrix problem (m is not used, the row embeddings are equal to the column embeddings)
        prop_zeros:random low-rank matrix with noise added

    Returns:
        random matrix (2D array)

    Examples:
        >>> np.random.seed(1)
        >>> toy_factorization_problem(5, 4)
        array([[ 3.677,  0.294,  1.414,  1.402],
               [ 2.34 ,  1.085,  1.482,  0.349],
               [ 5.884, -0.331,  3.245,  2.402],
               [ 3.428, -0.667,  2.128, -0.478],
               [ 0.309, -0.02 , -0.481,  0.398]])
        >>> toy_factorization_problem(5, 4, prop_zeros=0.5)
        array([[ 0.   ,  0.   , -1.87 ,  0.974],
               [-1.325,  0.   ,  1.042,  0.   ],
               [ 1.443, -0.896,  0.   ,  2.669],
               [ 1.799,  0.   , -1.28 , -0.889],
               [ 1.479,  0.   ,  0.   ,  0.   ]])
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


def svd_factorize_matrix(y_mat, rank, return_embeddings=False):
    """
    exact approximation of a matrix using square loss an fully observed entries
    Args:
        y_mat: input matrix to approximate
        rank: rank of the approximation
        return_embeddings: boolean. If True, it returns the embeddings instead of the approximate matrix

    Returns:
        approximate matrix of the specified rank

    Example:
        >>> np.random.seed(1)
        >>> mat = toy_factorization_problem(5, 4)
        >>> svd_factorize_matrix(mat, 2)
        array([[ 3.492,  0.148,  1.681,  1.545],
               [ 2.356, -0.032,  1.273,  0.648],
               [ 6.038,  0.099,  3.074,  2.198],
               [ 3.338, -0.508,  2.295, -0.472],
               [ 0.09 ,  0.148, -0.11 ,  0.473]])
    """
    from scipy.sparse.linalg import svds
    u1_mat, d1_vec, v1_matt = svds(y_mat, rank)
    d1_diag_matrix = np.zeros((rank, rank))
    for i in range(rank):
        d1_diag_matrix[i, i] = np.sqrt(d1_vec[i])
    u = np.dot(u1_mat, d1_diag_matrix)
    v = np.dot(v1_matt.T, d1_diag_matrix)
    if return_embeddings:
        return u, v
    else:
        return np.dot(u, v.T)


np.random.seed(1)
mat = toy_factorization_problem(5, 4)
svd_factorize_matrix(mat, 2)

