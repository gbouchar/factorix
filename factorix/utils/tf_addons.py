import numpy as np
import tensorflow as tf


def tf_eval(var, feed_dict=None):
    """ Opens a TensorFlow session, initialize the variables, evaluate the expression, close the session and return
    the result
    Args:
        var: TensorFlow expression
        feed_dict: optional dictionary of placeholder values to pass to the function

    Returns:
        result of the expression

    """
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        # for k, v in feed_dict.items():
        #     print(k, v)
        res = sess.run(var, feed_dict=feed_dict)
    return res


def tf_show(var: tf.Variable, name=None, summarize=1000):
    """
    Useful function to print the value of the current variable during evaluation
    Args:
        var: variable to show
        name: name to display
        summarize: number of values to display

    Returns:
        the same variable but wrapped with a Print module

    """
    name = name or var.name
    shape = tuple([d.value for d in var.get_shape()])
    return tf.Print(var, [var], message=name + str(shape), summarize=summarize)


def tf_debug_gradient(x, y, verbose=True):
    """
    Print the theoretical and numeric gradients, and the absolute difference between the two
    Args:
        x (tf.Variable): input variable
        y (tf.Variable): output variable
        verbose: switch display of information

    Returns:
        the theoretical and numeric gradient
    """
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        if verbose:
            print(y.eval())
        gt, gn = tf.test.compute_gradient(
                x, [d.value for d in x.get_shape()], y, [d.value for d in y.get_shape()], delta=1e-2)
        if verbose:
            print(np.concatenate((gt, gn, np.round(np.abs(gt-gn),2)), len(gt.shape) - 1))
            print(y.eval())
    return gt, gn


def batch_same_matmul(batched_vectors, mat, transpose_b=False):
    dims = [a.value for a in batched_vectors.get_shape()]
    if transpose_b:
        final_dim = mat.get_shape()[0].value
    else:
        final_dim = mat.get_shape()[-1].value
    unfolded_length = np.prod(dims[:-1])
    return tf.reshape(tf.matmul(tf.reshape(batched_vectors, (unfolded_length, dims[-1])), mat, transpose_b=transpose_b),
                      dims[:-1] + [final_dim])


def repeat_tensor(tensor, n_repeats, start_dim=0):
    if isinstance(n_repeats, int):
        n_repeats = (n_repeats,)
    dims = tuple(a.value for a in tensor.get_shape())
    tmp = tf.reshape(tensor, dims[:start_dim] + (1,) * len(n_repeats) + dims[start_dim:])
    return tf.tile(tmp, (1,) * start_dim + n_repeats + (1,) * (len(dims) - start_dim))


def softmax_loss(scores, indices):
    """

    Args:
        scores:
        indices:

    Returns:

    Examples
        >>> mat = tf.Variable([[[1, 2, 3, 3.1], [5, 6, 7, 7.1], [8, 9, 10, 10.1]], [[9, 8, 7, 6.9], [6, 5, 4, 3.9], [3, 2, 1, 0.9]]], name='mat')
        >>> idx = tf.Variable([[0, 1, 2], [0, 1, 3]], dtype=np.int64)
        >>> tf_eval(softmax_loss(mat, idx))
        array([[ 2.95873141,  1.95873165,  0.95873165],
               [ 0.48592091,  1.48592091,  2.58592081]], dtype=float32)
    """
    n_dims = len(scores.get_shape())
    return -select_entries(scores, indices) + reduce_log_sum_exp(scores, n_dims - 1)


def softmax(mat, reduction_indices=None, name=None):
    """

    Args:
        mat: original matrix
        reduction_indices: dimension in which the softmax applied

    Returns:
        normalized log-probabilities. Eg. for reduction_indices=1, it gives log(exp(x_ij)/(sum_k exp(x_ik)))

    Examples:
        >>> tf_eval(softmax(tf.Variable([1.0, 2.0, 3.0]), 0))
        array([ 0.09003058,  0.24472849,  0.665241  ], dtype=float32)
        >>> mat = tf.Variable([[[1, 2, 3, 3.1], [5, 6, 7, 7.1], [8, 9, 10, 10.1]], [[9, 8, 7, 6.9], [6, 5, 4, 3.9], [3, 2, 1, 0.9]]], name='mat')
        >>> a = tf_eval(softmax(mat, 1))
        >>> np.sum(a, 1)
        array([[ 0.99999994,  0.99999994,  0.99999994,  0.99999994],
               [ 1.00000048,  1.00000048,  1.        ,  1.        ]], dtype=float32)
        >>> a = tf_eval(softmax(mat, 0))
        >>> np.sum(a, 0)
        array([[ 0.9999997 ,  0.99999994,  1.00000012,  0.99999994],
               [ 1.00000024,  1.00000024,  1.        ,  1.00000012],
               [ 0.99999952,  0.99999976,  1.00000036,  0.99999994]], dtype=float32)
    """
    return tf.exp(log_softmax(mat, reduction_indices=reduction_indices), name=name)

def log_softmax(mat, reduction_indices=None, name=None):
    """

    Args:
        mat: original matrix
        reduction_indices: dimension in which the softmax applied

    Returns:
        normalized log-probabilities. Eg. for reduction_indices=1, it gives log(exp(x_ij)/(sum_k exp(x_ik)))

    Examples:
        >>> mat = tf.Variable([[[1, 2, 3, 3.1], [5, 6, 7, 7.1], [8, 9, 10, 10.1]], [[9, 8, 7, 6.9], [6, 5, 4, 3.9], [3, 2, 1, 0.9]]], name='mat')
        >>> a = tf_eval(tf.exp(log_softmax(mat, 1)))
        >>> np.sum(a, 1)
        array([[ 0.99999994,  0.99999994,  0.99999994,  0.99999994],
               [ 1.00000048,  1.00000048,  1.        ,  1.        ]], dtype=float32)
    """
    if reduction_indices is None:
        reduction_indices = [0] * len(mat.get_shape())
    n_rep = mat.get_shape()[reduction_indices].value
    lse = reduce_log_sum_exp(mat, reduction_indices=reduction_indices)
    probas = mat - repeat_tensor(lse, n_rep, reduction_indices)
    return tf.identity(probas, name=name)


def reduce_log_sum_exp(mat, reduction_indices=None, safe=True):
    """
    Perform numerically stable log-sum-exp operations
    Args:
        mat: the matrix from which the log-sum-exp is computed
        reduction_indices: the axis along which computations are made
        safe: whether use the numerically stable version (safe) or not (naive application of exp, reduce_sum and log)

    Returns:
        a tensor output with log-sum-exp values:
        If reduction_indices is 1, then:
        output[i] = log(sum_j(exp(mat[i, j])))

    """
    dims = tuple(a.value for a in mat.get_shape())
    if not safe:
        return tf.log(tf.reduce_sum(tf.exp(mat), reduction_indices=reduction_indices))
    else:
        maxi = tf.reduce_max(mat, reduction_indices=reduction_indices)
        maxi_rep = repeat_tensor(maxi, dims[reduction_indices], reduction_indices)
        return tf.log(tf.reduce_sum(tf.exp(mat - maxi_rep), reduction_indices=reduction_indices)) + maxi


# def tf_eval(op):
#     with tf.Session() as sess:
#         sess.run(tf.initialize_all_variables())
#         res = sess.run(op)
#     print(op.name, ':', res)
#     return res


def select_entries(tensor, idx):
    """
    Select entries in a tensor
    This is similar to the gather operator, but it selects one value at a time

    Args:
        tensor: 3d tensor from which values are extracted
        idx: 2d array of indices corresponding to the last dimension of the tensor

    Returns:
        the operator output which selects the right entries: output[i,j] = tensor[i, j, idx[i,j]]

    Example
        >>> mat = tf.Variable([[[1, 2, 3, 3.1], [5, 6, 7, 7.1], [8, 9, 10, 10.1]], [[9, 8, 7, 6.9], [6, 5, 4, 3.9], [3, 2, 1, 0.9]]], name='mat')
        >>> idx = tf.Variable(tf.cast([[0, 0, 3], [0, 1, 0]], np.int64), name='indices')
        >>> tf_eval(select_entries(mat, idx))
        array([[  1.        ,   5.        ,  10.10000038],
               [  9.        ,   5.        ,   3.        ]], dtype=float32)
    """
    mat_dims = tuple(a.value for a in tensor.get_shape())
    k = mat_dims[-1]
    idx_dims = tuple(a.value for a in idx.get_shape())
    if mat_dims[:-1] != idx_dims:
        raise ValueError("Value tensor has size {0} does not begin as the index tensor which has size {1}".format(
                mat_dims, idx_dims
        ))
    mat_reshaped = tf.reshape(tensor, (np.prod(mat_dims),))
    shifts1 = np.dot(np.ones((idx_dims[0], 1)), np.reshape(np.arange(0, idx_dims[1]), (1, -1))) * k
    shifts2 = np.dot(np.reshape(np.arange(0, idx_dims[0]), (-1, 1)), np.ones((1, idx_dims[1]))) * k * mat_dims[-2]
    # print(mat_dims, np.prod(mat_dims))
    # print(shifts1, shifts2)
    # print(mat_reshaped.get_shape())
    idx_reshaped = idx + tf.constant(shifts1 + shifts2, np.int64)
    return tf.gather(mat_reshaped, idx_reshaped)

# def main():
#     pass
#
# if __name__ == "__main__":
#     main()