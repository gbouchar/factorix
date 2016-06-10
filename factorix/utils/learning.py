import numpy as np
from numbers import Number
import tensorflow as tf
import collections
import time

from factorix.utils.dictionaries import _EOD_TOK

np.set_printoptions(precision=3)  # only useful to have more compact doctests


def learn(loss_op, sampler=None, optimizer=None, hooks=None, max_epochs=500, config=None, variables=None):
    """
    A generic function that minimizes the loss of a TensorFlow model, including hooks during learning

    Args:
        loss_op: TensorFlow operator that computes the loss to minimize
        sampler: sampler that generate dictionary inputs
        optimizer: TensorFlow optimization object [the Adam optimizer from TensorFlow]
        hooks: functions that are called during training [a simple display of the loss]
        max_epochs: maximal number of epochs through the data [500]
        variables: list of TensorFlow variables that are returned at the end of the learning [all trainable variables]

    Returns:
        list of values of the variables at the end of the learning

    Examples:
        Linear Regression
        -----------------
        Simple example to learn the parameters of a linear regression model

        >>> it, (x, y) = feed_dict_sampler([([[1.0, 2]], [2.0]), ([[4, 5]], [6.5]), ([[7, 8]], [11])])
        >>> w = tf.Variable(np.zeros((2, 1), dtype=np.float32))
        >>> l = tf.nn.l2_loss(tf.matmul(x, w) - y)
        >>> simple_hook = lambda it, e, xy, f: it and ((it % 500) == 0) and print("%d) loss=%f" % (it, f[0]))
        >>> w_opt = learn(l, it, hooks=[simple_hook], max_epochs=500, variables=[w])
        500) loss=0.000265
        1000) loss=0.000028
        1500) loss=0.000000
        >>> w_opt
        [array([[ 1. ],
               [ 0.5]], dtype=float32)]

        Matrix factorization
        -----------------

        We show how to factorize a rectangular matrix with square loss:
        >>> np.random.seed(1)
        >>> n, m, rank = 7, 6, 3
        >>> mat = np.random.randn(n, rank).dot(np.random.randn(rank, m))
        >>> tuples = [([i, n + j], mat[i, j]) for i in range(n) for j in range(m)]
        >>> tuple_iterable = data_to_batches(tuples, minibatch_size=n * m)
        >>> sampler, (x, y) = feed_dict_sampler(tuple_iterable, types=[np.int64, np.float32])
        >>> emb_var = tf.Variable(tf.cast(np.random.randn(n + m, rank), 'float32'))
        >>> negative_reward_op = tf.reduce_mean(tf.square(tf.reduce_sum(tf.reduce_prod(tf.gather(emb_var, x), 1), 1) - y))
        >>> emb, = learn(negative_reward_op, sampler,  max_epochs=200, variables=[emb_var])
        50) loss=0.005049
        100) loss=0.000035
        150) loss=0.000000
        200) loss=0.000000
        >>> mat_est = emb[:n, :].dot(emb[n:, :].T)
        >>> np.linalg.norm(mat_est - mat)  # we should have recovered the low-rank matrix
        0.00021819871358843225
    """
    # default values
    if variables is None:
        variables = tf.trainable_variables()
    if optimizer is None:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    if hooks is None:
        hooks = [lambda it, e, xy, loss_value: it and ((it % 50) == 0) and print("%d) loss=%f" % (it, loss_value[0]))]
    hooks = [func_to_iteration_hook(f) for f in hooks]  # if some functions are given, create a Hook object from it
    if config is None:
        config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)

    # the main operator
    minimization_op = optimizer.minimize(loss_op)
    # launch the graph
    return iterate_in_session([[loss_op, minimization_op]], sampler, max_epochs, hooks, config, variables)


def iterate_in_session(updates, feeder, max_epochs, hooks, config, variables, pre_feed=None):
    with tf.Session(config=config) as sess:
        sess.run(tf.initialize_all_variables())
        for hook in hooks:
            hook.session = sess

        for epoch in range(1, max_epochs + 1):
            if feeder is not None:
                for buck_idx, infeed in feeder:
                    if pre_feed:
                        infeed = pre_feed[buck_idx](infeed)
                    res = sess.run(updates[buck_idx], feed_dict=infeed)
                    [hook.end_of_iteration(infeed, res) for hook in hooks]
            else:
                for update in updates:
                    res = sess.run(update)
                    [hook.end_of_iteration(None, res) for hook in hooks]
            [hook.end_of_epoch() for hook in hooks]
        [hook.end_of_training() for hook in hooks]

        if variables:
            final_params = sess.run(variables)
        else:
            final_params = None
    return final_params


# from naga.shared.encoder_decoder import Seq2Seq

from tensorflow.python.training.optimizer import Optimizer
from sys import stderr


class Hook(object):
    def __init__(self, session=None):
        self.__session = session

    @property
    def session(self):
        return self.__session

    @session.setter
    def session(self, sess):
        self.__session = sess

    def end_of_iteration(self, batch, res):
        pass

    def end_of_epoch(self):
        pass

    def end_of_training(self):
        pass


def func_to_iteration_hook(f):
    if isinstance(f, collections.Callable):
        return IterationHook(f)
    elif issubclass(f.__class__, Hook):
        return f
    raise ValueError('The input must be a Hook or a function')


class IterationHook(Hook):
    def __init__(self, func):
        super().__init__()
        self.epoch = 0
        self.iteration = 0
        self.func = func

    def end_of_iteration(self, batch, res):
        self.iteration += 1
        self.func(self.iteration, batch, res)

    def end_of_epoch(self):
        self.epoch += 1


class OperatorHook(Hook):
    def __init__(self, operator: tf.Variable, feed_dict=None, when=('end_of_iteration',), store_results=None):
        super().__init__()
        self.operator = operator
        self.results = store_results and []
        self.feed_dict = feed_dict or {}
        for w in when:
            if w == 'end_of_iteration':
                self.end_of_iteration = self._end_of_iteration
            elif w == 'end_of_epoch':
                self.end_of_epoch = self._end_of_epoch
            elif w == 'end_of_training':
                self.end_of_training = self._end_of_training
            else:
                raise ValueError("Invalid value for 'when' argument. Should contain 'end_of_iteration', 'end_of_epoch' "
                                 "or 'end_of_training'")

    def _end_of_iteration(self, batch, res):
        res = self.session.run(self.operator, self.feed_dict)
        if self.results is not None:
            self.results.append(res)

    def _end_of_epoch(self):
        res = self.session.run(self.operator, self.feed_dict)
        if self.results is not None:
            self.results.append(res)

    def _end_of_training(self):
        res = self.session.run(self.operator, self.feed_dict)
        if self.results is not None:
            self.results.append(res)


class EpochHook(Hook):
    def __init__(self, batchsize: int):
        super().__init__()
        self.train_start = time.time()
        self.batchsize = batchsize
        self.epoch = 0
        self.tot_samples = 0

    def end_of_iteration(self, batch, res):
        self.tot_samples += self.batchsize

    def end_of_epoch(self):
        self.epoch += 1
        print('Epoch {} ({:.1f} samples(s)/s)'.format(self.epoch, self.tot_samples / (time.time() - self.train_start)),
              file=stderr)


class TrainLossHook(Hook):
    def __init__(self, model, buck_cnt, batch, n_data: int):
        super().__init__()
        self.n_data = n_data
        self.model = model
        self.buck_cnt = buck_cnt
        self.batch = batch

    def end_of_epoch(self):
        train_loss = 0
        for buck_idx in range(len(self.buck_cnt)):
            outs = self.session.run((self.model.losses[buck_idx],),
                                    self.batch(buck_idx, whole_bucket=True))
            train_loss += outs[0] * self.buck_cnt[buck_idx]
        train_loss /= self.n_data
        print('Training loss: {:.8f}'.format(train_loss), file=stderr)




class AccuracyHook(Hook):
    def __init__(self, model, dev_data, dec_ids, ):
        super().__init__()
        self.train_start = time.time()
        self.model = model
        self.dev_data = dev_data
        self.dec_ids = dec_ids

    def end_of_epoch(self):
        dev_qs, Y, W, infeed = self.dev_data
        # Remove <EOI/>.
        # Y = Y[0:-1, :]
        # W = W[0:-1, :]
        outs = self.session.run([self.model.losses[-1], ] + self.model.outputs[-1], infeed)
        preds = np.vstack(o.argmax(1) for o in outs[1:])
        dev_loss = outs[0]
        print('Dev loss: {:.8f}'.format(dev_loss), file=stderr)

        dev_corr = 0
        for col in range(0, preds.shape[1]):
            q = dev_qs[col]
            p_toks = []
            for row in range(0, preds.shape[0]):
                p_toks.append(self.dec_ids.key_by_id(preds[row, col]))
            try:
                eod = p_toks.index(_EOD_TOK)
            except ValueError:
                eod = len(p_toks)
            p_eq = ' '.join(str(t) for t in q.unmask(p_toks[:eod]))
            try:
                p_ans = eval(p_eq)
            except (SyntaxError, TypeError):
                p_ans = 'N/A'
            if isinstance(p_ans, Number) and q.is_correct(p_ans):
                dev_corr += 1
        dev_acc = dev_corr / len(dev_qs) * 100
        print('Dev. accuracy: {:.2f}%'.format(dev_acc), file=stderr)


def bucket_feeder(batch, buck_prob, n_data, batchsize):
    from math import ceil

    def bucket_feeder0():
        for _ in range(1, ceil(n_data / batchsize) + 1):
            buck_idx = buck_prob.random()
            yield buck_idx, batch(buck_idx)

    return AutoReset(bucket_feeder0)


def learn_py_pontus(model, batch, optimizer=None, max_epochs=500, batchsize=128,
                    output=None, verbose=False, n_data=None, buck_cnt=None, buck_prob=None, dev_data=None,
                    dec_ids=None):
    """
    A generic function that Pontus could use to minimizes the loss of a TensorFlow model

    Args:
        model: Seq2Seq object with fields 'enc_inputs', 'dec_inputs', 'tgt_ws', 'updates', 'losses', 'outputs'
        batch (collection.Callable): the batch function that take the bucket index as argument and is called iteratively
        optimizer (Optimizer): a TensorFlow optimization object
        max_epochs (int): the maximal number of iterations
        batchsize (int): integer. the size of every batch
        output (int): the output file (a file id)
        verbose (bool):
        n_data (int):
        buck_cnt (numpy.ndarray):
        buck_prob: object with random method
        dev_data (tuple): 4-tuple
        dec_ids: object with key_by_id method

    Returns:

    """

    # As with a ton of other libraries, TensorFlow is far too optimistic in
    #   how many threads it uses for parallelism.  Lowering the number of
    #   threads leads to significant performance gains.
    config = tf.ConfigProto(
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1,
    )

    if optimizer is None:
        try:
            updates = [[u] for u in model.updates]  # in some objects such as Seq2Seq, the model contains an update
        except AttributeError:  # if there is not update, use Adam
            optimizer = optimizer or tf.train.AdamOptimizer(learning_rate=1e-3)
            updates = [[optimizer.minimize(l)] for l in model.losses]
    else:  # if there is an optimizer use it on all the losses
        updates = [[optimizer.minimize(l)] for l in model.losses]

    feeder = bucket_feeder(batch, buck_prob, n_data, batchsize)

    hooks = []
    if verbose:
        hooks.append(EpochHook(batchsize))
        hooks.append(TrainLossHook(model, buck_cnt, batch, n_data))
        hooks.append(AccuracyHook(model, dev_data, dec_ids))

    iterate_in_session(updates, feeder, max_epochs, hooks, config, variables=None)


def data_to_batches(data, minibatch_size, dtypes=None, shuffling=True):
    """
    Create an iterator over vectorized version of the data (they are converted into numpy arrays)
    Args:
        data: list of items that will be concatenated
        minibatch_size: size of the minibatches
        dtypes: list of types for each of the arrays. Should be types compatibles with numpy
        shuffling (bool): should we randomly permute the data or not?

    Returns:
        an iterator over (input, output) where input is a 2D array with 3
        This iterator can be started again forever (unlike usual generators)

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
    if dtypes:
        inputs = []
        for i, t in zip(range(arity), dtypes):
            # [print(np.array(t[i]).shape, t[i]) for t in data]
            a = np.array([t[i] for t in data], dtype=t)
            inputs.append(a)
    else:
        inputs = [np.array([t[i] for t in data]) for i in range(arity)]

    def new_generator():  # every time the iterator is called, sample new indices
        minibatch_indices, n_rem = create_minibatch_indices(n, minibatch_size, shuffling=shuffling)
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


def create_minibatch_indices(n, minibatch_size, shuffling=True):
    """
    :param n: total number of indices from which to pick from
    :param minibatch_size: size of the minibatches (must be lower than n)
    :return: (list of random indices, number of random duplicate indices in the last minibatch to complete it)
    """
    if shuffling:
        all_indices = np.random.permutation(n)  # shuffle order randomly
    else:
        all_indices = np.arange(n)
    n_steps = (n - 1) // minibatch_size + 1  # how many batches fit per epoch
    n_rem = n_steps * minibatch_size - n  # remainder
    if n_rem > 0:
        inds_to_add = np.random.randint(0, n_rem, size=n_rem)
        all_indices = np.concatenate((all_indices, inds_to_add))
    return np.split(all_indices, n_steps), n_rem


from typing import Tuple, Generator


def feed_dict_sampler(iterable, names=None, types=None, placeholders=None):
    """
    Sampler that generate a dictionary where keys are TensorFlow placeholders and values are the iterable values
    Args:
        iterator: an iterable of multiple values (one per entry to feed)
        names: name of the placeholders that are generated
        placeholders: list of placeholders

    Returns: 2 outputs
        1. An iterable generating a dictionary feed_dict to feed a TensorFlow model (using _session.run(op, feed_dict))
        2. An list of TensorFlow placeholders that can be used to create models

    Examples:
        # just sums arrays given as input
        >>> it, (x, y) = feed_dict_sampler([([1, 2], [3]), ([4, 5], [6]), ([7, 8], [9])])
        >>> sess = tf.Session()
        >>> [sess.run(tf.reduce_sum(x) + y, feed_dict=f[1]) for f in it]
        [array([ 6.], dtype=float32), array([ 15.], dtype=float32), array([ 24.], dtype=float32)]
        >>> [sess.run(tf.reduce_sum(x) + y, feed_dict=f[1]) for f in it]
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
        >>> losses = [[sess.run([min_loss, loss], feed_dict=f[1])[1] for f in it] for t in range(1000)]
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

        dict_iterable = placeholder_feeder(placeholders, iterable)
    return dict_iterable, placeholders


def placeholder_feeder(placeholders, iterable):
    food_func = lambda items: (0, dict(zip(placeholders, items)))
    return AutoReset(lambda: map(food_func, iter(iterable)))


def min_cross_entropy(log_p, p_sampler=None, log_q=None, sampler=None, optimizer=None, hooks=None, max_epochs=500, avg_coef=0.999,
                      config=None, variables=None):
    """
    maximize entropy or minimize the cross-entropy between two distribution by first sampling and doing gradient steps

    Args:
        log_p: TensorFlow operator that computes the log-probability of a sample
        sampler: sampler that generate dictionary inputs (in None, it assumes log_p first samples according to p)
        optimizer: TensorFlow optimization object [the Adam optimizer from TensorFlow]
        hooks: functions that are called during training [a simple display of the loss]
        max_epochs: maximal number of epochs through the data [500]
        variables: list of TensorFlow variables that are returned at the end of the learning [all trainable variables]

    Returns:
        list of values of the variables at the end of the learning


    Examples:
    """
    # default values
    if variables is None:
        variables = tf.trainable_variables()
    if optimizer is None:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    if hooks is None:
        hooks = [lambda it, e, xy, f: it and ((it % 50) == 0) and print("%d) loss=%f" % (it, f[0]))]
    hooks = [func_to_iteration_hook(f) for f in hooks]  # if some functions are given, create a Hook object from it
    if config is None:
        config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
    loss_op = 0.5 * (log_p - log_q + 1) ** 2  # the magical formula (you can check it expected derivative under p)
    # the main operator
    minimization_op = optimizer.minimize(loss_op)
    # launch the graph
    return iterate_in_session([[loss_op, minimization_op]], [(0, None)], max_epochs, hooks, config, variables,
                              pre_feed=sampler)

if __name__ == '__main__':
    if True:
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
    else:
        m = 10  # minibatch size
        d = 2
        # sampler = lambda tau: np.linalg.norm(tau - x) + np.random.rand(m, d), types=[np.float32])
        # log_p = tf_gaussian.log_proba(input, mu, 1.0)
        # sampler =

