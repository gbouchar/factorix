
from collections import defaultdict
import tensorflow as tf
from naga.shared.tf_addons import tf_eval


def bin_tuples_by_type(tuples):
    """
    Creates a dictionary of output types containing tuples of the corresponding type.

    Args:
        tuples: list of (input, output) pairs. The output is used to decide the type.

    Returns:
        list of (input, output) pairs. The output is used to decide the type.

    Raises:
        ValueError: If there is an unsupported output type (e.g. a string).

    Examples:
        >>> d = bin_tuples_by_type([((1, 2), 1.0), ((2, 3), 1), ((3, 4, 5), True), ((6, 1), True)])
        >>> d['quadratic']
        [((1, 2), 1.0)]
        >>> d['logistic']
        [((3, 4, 5), True), ((6, 1), True)]
        >>> d['poisson']
        [((2, 3), 1)]
    """
    binned_tuples = defaultdict(list)
    for t_in, t_out in tuples:
        binned_tuples[get_loss_type(t_out)].append((t_in, t_out))
    return dict(binned_tuples)


def get_loss_type(x):
    """
    Type of loss for a example of data:
    - 'quadratic' for float
    - 'logistic' for bool
    - 'poisson' for int

    Args:
        x: the example data

    Returns:
        A string corresponding to the loss type
    """
    if isinstance(x, float):
        return 'quadratic'
    elif isinstance(x, bool):
        return 'logistic'
    elif isinstance(x, int):
        return 'poisson'
    else:
        raise ValueError('Invalid type ' + x.__class__())


def loss_func(pred, gold, type_of_loss, emb_norm=0):
    """
    Quadratic loss function
    :param pred: prediction
    :param gold: ground truth
    :return: tensor the same size as the pred tensor with corresponding element-wise losses
    >>> print(tf_eval(loss_func(tf.Variable(1.0), tf.Variable(2.0), 'quadratic')))
    0.5
    >>> print(tf_eval(loss_func(tf.Variable(0.0), tf.Variable(0.), 'logistic')))
    0.693147
    >>> print(tf_eval(loss_func(tf.Variable(0.0), tf.Variable(1.), 'logistic')))
    0.693147
    >>> print(tf_eval(loss_func(tf.Variable(1.0), tf.Variable(1.), 'logistic')))
    0.313262
    """
    if isinstance(pred, tuple):  # when a second argument is given, it is added to the loss (e.g. regularization)
        if len(pred) == 2:
            emb_norm = pred[1]
            pred = pred[0]
        else:
            raise ValueError('When the predictions are a tuple, they must have length 2')

    if type_of_loss == 'quadratic':
        loss = loss_func_quadratic(pred, gold)
    elif type_of_loss == 'logistic':
        loss = loss_func_logistic(pred, gold)
    elif type_of_loss == 'softmax':
        loss = loss_func_softmax(pred, gold)
    else:
        raise ValueError
    return loss + emb_norm


def loss_func_quadratic(pred, gold):
    """
    Quadratic loss function
    :param pred: prediction
    :param gold: ground truth
    :return: tensor the same size as the pred tensor with corresponding element-wise losses

    >>> print(tf_eval(loss_func_quadratic(tf.Variable(1.0), tf.Variable(1.0))))
    0.0

    """
    return 0.5 * tf.square(pred - gold)


def loss_func_logistic(pred, gold):
    return tf.nn.softplus(pred * (1 - 2 * gold))


def softmax_loss(preds, golds):
    loss = tf.reduce_mean(loss_func_softmax(preds, golds))
    return loss


def loss_func_softmax(pred, gold):
    """softmax function with integers as the second argument (instead of zero-one encoding matrix)

    Args:
        pred: log-odds where the last dimension is the number of labels
        gold: integer array the same size as pred but the last dimension which is 1

    Returns:
        the softmax values applied to the predictions

    """
    pred = tf.reshape(pred, [-1, pred.get_shape()[-1].value])
    gold = tf.reshape(gold, [pred.get_shape()[0].value])
    n = pred.get_shape()[0].value
    voc_size = pred.get_shape()[1].value
    rg = tf.range(0, n)
    inds = tf.transpose(tf.pack([rg, tf.cast(gold, 'int32')]))
    vals = tf.ones([n])
    # gold_mat = tf.SparseTensor( , [n, voc_size])
    gold_mat = tf.sparse_to_dense(inds, [n, voc_size], vals)
    return tf.nn.softmax_cross_entropy_with_logits(pred, gold_mat)


def test_loss_func_softmax():
    """Test the softmax computation
    Returns:
        Nothing
    >>> test_loss_func_softmax()
    [ 2.00384593  2.19245744  0.31507221  1.76036525]
    [ 0.20384566  0.59245753  1.31507206  0.26036522]
    """
    pred = tf.Variable([[1.1, 2.1, 3.9], [-1.0, -0.5, -2.1], [1.1, 2.1, -3.9], [-1.0, 0.5, -2.1]])
    gold = tf.Variable([1, 2, 1, 0])
    print(tf_eval(loss_func_softmax(pred, gold)))
    gold = tf.Variable([2, 1, 0, 1])
    print(tf_eval(loss_func_softmax(pred, gold)))


if __name__ == '__main__':
    test_loss_func_softmax()