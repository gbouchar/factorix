import numpy as np
import tensorflow as tf

np.set_printoptions(precision=3)  # only useful to have more compact doctests


def learn(loss_op, sampler, optimizer=None, hooks=None, max_epochs=500, variables=None):
    """

    Args:
        loss_op: TensorFlow operator that computes the loss to minimize
        sampler: sampler that generate dictionary inputs
        optimizer: TensorFlow optimization object
        hooks: functions that are called during training
        max_epochs: maximal number of epochs through the data

    Returns:

    Examples:
        Linear Regression
        -----------------

    >>> it, (x, y) = feed_dict_sampler([([[1.0, 2]], [2.0]), ([[4, 5]], [6.5]), ([[7, 8]], [11])])
    >>> w = tf.Variable(np.zeros((2, 1), dtype=np.float32))
    >>> optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    >>> l = tf.nn.l2_loss(tf.matmul(x, w) - y)
    >>> simple_hook = lambda s, e, it, l, c: it and ((it % 500) == 0) and print("%d) loss=%f" % (it, c))
    >>> w_opt = learn(l, it, optimizer, hooks=[simple_hook], max_epochs=500, variables=[w])
    500) loss=0.000265
    1000) loss=0.000028
    1500) loss=0.000000
    >>> w_opt
    [array([[ 1. ],
           [ 0.5]], dtype=float32)]

    We show how to factorize a rectangular matrix with square loss:
    >>> np.random.seed(1)
    >>> n, m, rank = 7, 6, 3
    >>> mat = np.random.randn(n, rank).dot(np.random.randn(rank, m))
    >>> tuples = [([i, n + j], mat[i, j]) for i in range(n) for j in range(m)]
    >>> tuple_iterable = data_to_batches(tuples, minibatch_size=n * m)
    >>> sampler, (x, y) = feed_dict_sampler(tuple_iterable, types=[np.int64, np.float32])
    >>> emb_var = tf.Variable(tf.cast(np.random.randn(n + m, rank), 'float32'))
    >>> offset = tf.Variable(1.0, 'float32')
    >>> negative_reward_op = tf.reduce_mean(tf.square(tf.reduce_sum(tf.reduce_prod(tf.gather(emb_var, x), 1), 1) + offset - y))
    >>> embeddings, offset_val = learn(negative_reward_op, sampler,  max_epochs=200, variables=[emb_var, offset])
    50) loss=0.005049
    100) loss=0.000035
    150) loss=0.000000
    200) loss=0.000000
    >>> mat_est = embeddings[:n, :].dot(embeddings[n:, :].T)
    >>> np.linalg.norm(mat_est - mat)  # we should have recovered the low-rank matrix
    0.00021819871358843225
    """
    # default values
    if variables is None:
        variables = tf.trainable_variables()
    if optimizer is None:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    if hooks is None:
        hooks = [lambda s, e, it, l, c: it and (((it) % 50) == 0) and print("%d) loss=%f" % (it, c))]

    # the main operator
    minimization_op = optimizer.minimize(loss_op)
    # Launch the graph.
    with tf.Session() as session:  # we close the s at the end of the training
        session.run(tf.initialize_all_variables())
        epoch = 1
        iteration = 1
        while epoch <= max_epochs:
            for feed_dict in sampler:
                _, current_loss = session.run([minimization_op, loss_op], feed_dict=feed_dict)
                for hook in hooks:
                    hook(session, epoch, iteration, loss_op, current_loss)
                iteration += 1
            for hook in hooks:  # calling post-e hooks
                hook(session, epoch, None, loss_op, 0)
            epoch += 1
        final_params = session.run(variables)
    return final_params


def data_to_batches(data, minibatch_size):
    """

    Args:
        tuples: data
        minibatch_size:

    Returns:
        an iterator over (input, output) where input is a 2D array with 3

    Examples
        >>> t = [((1, 2), 1), ((2, 3), 2), ((3, 1), 3), ((4, 2), 4), ((5, 4), 5), ((5, 1), 6), ((3, 5), 7)]
        >>> [x.shape for x, y in data_to_batches(t, 3)]
        [(3, 2), (3, 2), (3, 2)]
        >>> [y.shape for x, y in data_to_batches(t, 3)]
        [(3,), (3,), (3,)]
    """
    n = len(data)
    arity = len(data[0])

    # create arrays only once
    inputs = [np.array([t[i] for t in data]) for i in range(arity)]

    def new_generator():  # every time the iterator is called, sample new indices
        minibatch_indices, n_rem = create_minibatch_indices(n, minibatch_size)
        for ids in minibatch_indices:
            next_batch = []
            for arr in inputs:
                if len(arr.shape) == 1:  # make sure we can index the array correctly
                    next_batch.append(arr[ids])
                else:
                    next_batch.append(arr[ids, :])
            yield next_batch

    return AutoReset(new_generator)  # create an object with a "__iter__" method


class AutoReset(object):
    """ Enables an iterator to be automatically reset when it is called after a StopIteration exception
    """
    def __init__(self, iterator, *args):
        """
        Initializer
        Args:
            iterator: function that creates the iterator
            *args: arguments to pass to the iterator

        Returns:
            An iterator that can be called again

        """
        self.iterator = iterator
        self.args = args

    def __iter__(self):
        return self.iterator(*self.args)


def create_minibatch_indices(n, minibatch_size):
    """
    :param n: total number of indices from which to pick from
    :param minibatch_size: size of the minibatches (must be lower than n)
    :return: (list of random indices, number of random duplicate indices in the last minibatch to complete it)
    """
    all_indices = np.random.permutation(n)  # shuffle order randomly
    n_steps = (n - 1) // minibatch_size + 1  # how many batches fit per epoch
    n_rem = n_steps * minibatch_size - n  # remainder
    if n_rem > 0:
        inds_to_add = np.random.randint(0, n_rem, size=n_rem)
        all_indices = np.concatenate((all_indices, inds_to_add))
    return np.split(all_indices, n_steps), n_rem


def feed_dict_sampler(iterable, names=None, types=None, placeholders=None):
    """
    Sampler that generate a dictionary where keys are TensorFlow placeholders and values are the iterable values
    Args:
        iterator: an iterable of multiple values (one per entry to feed)
        names: name of the placeholders that are generated
        placeholders: list of placeholders

    Returns: 2 outputs
        1. An iterable generating a dictionary feed_dict to feed a TensorFlow model (using session.run(op, feed_dict))
        2. An list of TensorFlow placeholders that can be used to create models

    Examples:
        # just sums arrays given as input
        >>> it, (x, y) = feed_dict_sampler([([1, 2], [3]), ([4, 5], [6]), ([7, 8], [9])])
        >>> sess = tf.Session()
        >>> [sess.run(tf.reduce_sum(x) + y, feed_dict=f) for f in it]
        [array([ 6.], dtype=float32), array([ 15.], dtype=float32), array([ 24.], dtype=float32)]
        >>> [sess.run(tf.reduce_sum(x) + y, feed_dict=f) for f in it]
        [array([ 6.], dtype=float32), array([ 15.], dtype=float32), array([ 24.], dtype=float32)]
        >>> sess.close()

        # learns a linear regression
        >>> it, (x, y) = feed_dict_sampler([([[1.0, 2]], [2.0]), ([[4, 5]], [6.5]), ([[7, 8]], [11])])
        >>> w = tf.Variable(np.zeros((2, 1), dtype=np.float32))
        >>> optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
        >>> loss = tf.nn.l2_loss(tf.matmul(x, w) - y)
        >>> min_loss = optimizer.minimize(loss)
        >>> sess = tf.Session()
        >>> sess.run(tf.initialize_all_variables())
        >>> losses = [[sess.run([min_loss, loss], feed_dict=f)[1] for f in it] for t in range(1000)]
        >>> np.hstack([np.array(losses[i-1]) for i in [1,10,100,1000]])
        array([  2.000e+00,   1.568e+01,   3.423e+01,   2.882e-02,   8.828e-06,
                 1.218e-01,   7.667e-03,   8.059e-04,   1.189e-03,   0.000e+00,
                 4.547e-13,   7.276e-12], dtype=float32)
        >>> sess.run(w)
        array([[ 1. ],
               [ 0.5]], dtype=float32)
        >>> sess.close()

        # >>> [[f for f in it] for t in range(100)]
    """
    first_items = next(iter(iterable))  # if there is a fixed size input, the first item is enough to infer shape
    if placeholders is None:  # initialize the placeholders in the first iteration
        if names is None:
            names = ['Placeholder%d' % i for i in range(len(first_items))]
        if types is None:
            types = [None] * len(first_items)
        placeholders = []
        for a, (s, t) in zip(first_items, zip(names, types)):
            if isinstance(a, str):  # just in case the input is a string, there is a TensorFlow type for that
                placeholders.append(tf.placeholder(tf.string, 1, name=s))
            else:  # most of the time, an array-like is provided
                if not isinstance(a, np.ndarray):
                    if t is None:
                        a = np.array(a, dtype=np.float32)  # just in case the input is not an array
                        t = a.dtype.name
                    else:
                        a = np.array(a, dtype=t)  # just in case the input is not an array
                else:
                    if t is None:
                        t = a.dtype.name
                placeholders.append(tf.placeholder(t, a.shape, name=s))

    dict_iterable = AutoReset(lambda: map(lambda items: dict(zip(placeholders, items)), iter(iterable)))
    return dict_iterable, placeholders


if __name__ == '__main__':
    # test_learning_factorization(verbose=True)
    n, m, rank = 7, 6, 3
    mat = np.random.randn(n, rank).dot(np.random.randn(rank, m))
    tuples = [([i, n + j], mat[i, j]) for i in range(n) for j in range(m)]
    tuple_iterable = data_to_batches(tuples, minibatch_size=n * m)
    sampler, (x, y) = feed_dict_sampler(tuple_iterable, types=[np.int64, np.float32])
    emb_var = tf.Variable(tf.cast(np.random.randn(n + m, rank), 'float32'))
    loss_op = tf.reduce_mean(tf.square(tf.reduce_sum(tf.reduce_prod(tf.gather(emb_var, x), 1), 1) - y))
    emb, = learn(loss_op, sampler,  max_epochs=200)
    mat_est = emb[:n, :].dot(emb[n:, :].T)
    print(np.linalg.norm(mat_est - mat))  # we should have recovered the low-rank matrix

