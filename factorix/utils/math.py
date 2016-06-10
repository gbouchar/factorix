import numpy as np
from scipy.misc import logsumexp


def unique_rows(a):
    """Unique rows of a matrix
    Args:
        a: 2D array

    Returns:
        an array with the same number of columns but where rows are unique

    >>> unique_rows([[1,2], [1,2], [1,4], [4,1], [2, 1], [1,2], [4,1]])
    array([[1, 2],
           [2, 1],
           [4, 1],
           [1, 4]])
    """
    return np.vstack({tuple(row) for row in a})


def c(vec):
    """Complement function for probabilities in the log-space: robustly computes 1-P(A) in the log-space
    Args:
        vec: vector of negative numbers representing log-probabilities of an event.

    Returns: the log-probabilities of (1-P(A)) were log(P(A)) are given in the vec numpy array

    Examples:
        >>> c(-1e-200)
        -460.51701859880916

        # >>> np.log(1 - np.exp(-1e-200)) raises a `RuntimeWarning: divide by zero` error
    """
    # return np.log1p(-np.exp(vec))  # Not robust to -1e-200
    if np.max(np.array(vec)) > 0:
        print('vec', vec)
    return np.log(-np.expm1(vec))


def log_softmax(vec):
    """Robust version of the log of softmax values

    Args:
        vec: vector of log-odds

    Returns:
        A vector whose exponential sums to one with lgo of softmax values log(exp(x_k)/sum_i (exp(x_i)))

    Examples:
        >>> print(log_softmax(np.array([1.0, 1.0, 1.0, 1.0])))
        [-1.38629436 -1.38629436 -1.38629436 -1.38629436]
        >>> print(log_softmax(np.array([-1.0, -1.0, -1.0, -1.0])))
        [-1.38629436 -1.38629436 -1.38629436 -1.38629436]
        >>> print(log_softmax(np.array([1.0, 0.0, -1.0, 1.1])))
        [-0.9587315 -1.9587315 -2.9587315 -0.8587315]

    """
    return vec - logsumexp(vec)


def softmax(vec):
    """Robust version of the softmax

    Args:
        vec: vector of log-odds

    Returns:
        A vector summing to one with softmax values exp(x_k)/sum_i (exp(x_i))

    Examples:
        >>> print(softmax(np.array([1.0, 1.0, 1.0, 1.0])))
        [ 0.25  0.25  0.25  0.25]
        >>> print(softmax(np.array([-1.0, -1.0, -1.0, -1.0])))
        [ 0.25  0.25  0.25  0.25]
        >>> print(softmax(np.array([1.0, 0.0, -1.0, 1.1])))
        [ 0.38337889  0.14103721  0.05188469  0.4236992 ]
    """
    return np.exp(log_softmax(vec))


def repeat_equally(a, n):
    """ Expands an array to a given number of row by repeating rows nearly equally

    Args:
        a: array (or the number of rows we want to repeat equally)
        n: number of rows on the output array

    Returns:
        array with n rows and the same shape in the other dimension
        if a is an integer, return the number of 'approximately equal' repetitions

    Examples:
        >>> repeat_equally([[1,2],[3,4],[5,6]],7)
        array([[1, 2],
               [1, 2],
               [3, 4],
               [3, 4],
               [5, 6],
               [5, 6],
               [5, 6]])
    """

    if isinstance(a, int):
        return np.concatenate([np.ones(a - n % a) * (n//a), np.ones(n % a) * (n//a + 1)])
    else:
        a = np.array(a)
        k = a.shape[0]
        n_rep = np.concatenate([np.ones(k - n % k, dtype=int) * (n//k),
                                np.ones(n % k, dtype=int) * (n//k + 1)]).tolist()
        return np.repeat(a, n_rep, axis=0)

import numpy as np
from scipy.misc import logsumexp


def unique_rows(a):
    """Unique rows of a matrix
    Args:
        a: 2D array

    Returns:
        an array with the same number of columns but where rows are unique

    >>> unique_rows([[1,2], [1,2], [1,4], [4,1], [2, 1], [1,2], [4,1]])
    array([[1, 2],
           [2, 1],
           [4, 1],
           [1, 4]])
    """
    return np.vstack({tuple(row) for row in a})


def c(vec):
    """Complement function for probabilities in the log-space: robustly computes 1-P(A) in the log-space
    Args:
        vec: vector of negative numbers representing log-probabilities of an event.

    Returns: the log-probabilities of (1-P(A)) were log(P(A)) are given in the vec numpy array

    Examples:
        >>> c(-1e-200)
        -460.51701859880916

        # >>> np.log(1 - np.exp(-1e-200)) raises a `RuntimeWarning: divide by zero` error
    """
    # return np.log1p(-np.exp(vec))  # Not robust to -1e-200
    if np.max(np.array(vec)) > 0:
        print('vec', vec)
    return np.log(-np.expm1(vec))


def log_softmax(vec):
    """Robust version of the log of softmax values

    Args:
        vec: vector of log-odds

    Returns:
        A vector whose exponential sums to one with lgo of softmax values log(exp(x_k)/sum_i (exp(x_i)))

    Examples:
        >>> print(log_softmax(np.array([1.0, 1.0, 1.0, 1.0])))
        [-1.38629436 -1.38629436 -1.38629436 -1.38629436]
        >>> print(log_softmax(np.array([-1.0, -1.0, -1.0, -1.0])))
        [-1.38629436 -1.38629436 -1.38629436 -1.38629436]
        >>> print(log_softmax(np.array([1.0, 0.0, -1.0, 1.1])))
        [-0.9587315 -1.9587315 -2.9587315 -0.8587315]

    """
    return vec - logsumexp(vec)


def softmax(vec):
    """Robust version of the softmax

    Args:
        vec: vector of log-odds

    Returns:
        A vector summing to one with softmax values exp(x_k)/sum_i (exp(x_i))

    Examples:
        >>> print(softmax(np.array([1.0, 1.0, 1.0, 1.0])))
        [ 0.25  0.25  0.25  0.25]
        >>> print(softmax(np.array([-1.0, -1.0, -1.0, -1.0])))
        [ 0.25  0.25  0.25  0.25]
        >>> print(softmax(np.array([1.0, 0.0, -1.0, 1.1])))
        [ 0.38337889  0.14103721  0.05188469  0.4236992 ]
    """
    return np.exp(log_softmax(vec))


def repeat_equally(a, n):
    """ Expands an array to a given number of row by repeating rows nearly equally

    Args:
        a: array (or the number of rows we want to repeat equally)
        n: number of rows on the output array

    Returns:
        array with n rows and the same shape in the other dimension
        if a is an integer, return the number of 'approximately equal' repetitions

    Examples:
        >>> repeat_equally([[1,2],[3,4],[5,6]],7)
        array([[1, 2],
               [1, 2],
               [3, 4],
               [3, 4],
               [5, 6],
               [5, 6],
               [5, 6]])
    """

    if isinstance(a, int):
        return np.concatenate([np.ones(a - n % a) * (n//a), np.ones(n % a) * (n//a + 1)])
    else:
        a = np.array(a)
        k = a.shape[0]
        n_rep = np.concatenate([np.ones(k - n % k, dtype=int) * (n//k),
                                np.ones(n % k, dtype=int) * (n//k + 1)]).tolist()
        return np.repeat(a, n_rep, axis=0)

