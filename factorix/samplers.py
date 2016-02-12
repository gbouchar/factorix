from itertools import cycle, islice
import numpy as np
import tensorflow as tf
from collections import namedtuple

from factorix.losses import get_loss_type
#from naga.shared.math import unique_rows, repeat_equally
from naga.shared.math import unique_rows, repeat_equally

np.set_printoptions(precision=3)

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


def positive_minibatch_generator(tuples, minibatch_size):
    """

    Args:
        tuples: tuples
        minibatch_size:

    Returns:
        an iterator over (input, output) where input is a 2D array with 3

    Examples
        >>> t = [((1, 2), 1), ((2, 3), 2), ((3, 1), 3), ((4, 2), 4), ((5, 4), 5), ((5, 1), 6), ((3, 5), 7)]
        >>> [x.shape for x, y in positive_minibatch_generator(t, 3)]
        [(3, 2), (3, 2), (3, 2)]
        >>> [y.shape for x, y in positive_minibatch_generator(t, 3)]
        [(3,), (3,), (3,)]
    """
    n = len(tuples)
    minibatch_indices, n_rem = create_minibatch_indices(n, minibatch_size)
    inputs = np.array([x for x, y in tuples])
    outputs = np.array([y for x, y in tuples])
    for ids in minibatch_indices:
        yield inputs[ids, :], outputs[ids]


#def generate_negatives(input_pos, n, idx_ranges: list((int, str))):
def generate_negatives(input_pos, n, idx_ranges):
    """
    Sample negative tuples by adding noise to a list of positive tuples
    Args:
        input_pos: list of positive tuples to generate from
        n: number of negative tuples to sample
        idx_ranges: range of the indices for each of the modes of the tensor

    Returns:
        an array with n rows and as many columns as the positive tuples given as input

    Examples:
        >>> np.random.seed(1)
        >>> generate_negatives(np.array([[1, 8, 3], [1, 9, 2], [2, 8, 1]]), 7, [(0, 8), (8, 100), (0, 8)])[0]
        array([[ 1, 20,  3],
               [ 1, 80,  3],
               [ 1, 17,  2],
               [ 1, 83,  2],
               [ 2, 13,  1],
               [ 2, 87,  1],
               [ 2, 72,  1]])
        >>> generate_negatives(np.array([[1, 8, 3], [1, 9, 2], [2, 8, 1]]), 7, [(0, 8), (8, 100), (0, 8)])[0]
        array([[1, 8, 3],
               [4, 8, 3],
               [7, 9, 2],
               [5, 9, 2],
               [4, 8, 1],
               [6, 8, 1],
               [1, 8, 1]])
    """
    input_neg = repeat_equally(input_pos, n)
    mode = np.random.randint(input_pos.shape[1])
    idx_low, idx_high = idx_ranges[mode]
    input_neg[:, mode] = np.random.randint(idx_low, idx_high, size=n)
    output_neg = np.zeros(n, dtype=np.float32)
    return input_neg, output_neg


def add_negatives(positives, n_negatives, idx_ranges):
    """
    Add a negative sampler on top of a set of positive tuples
    Args:
        positives: an iterator over positive tuples
        n_negatives: number of generated negatives

    Returns:
        iterator over positive and negative values

    Examples:
        >>> np.random.seed(2)
        >>> pos1 = (np.array([[1, 2, 3], [1, 3, 2]]), np.array([7, 6]))
        >>> pos2 = (np.array([[4, 5, 6], [7, 6, 7]]), np.array([9, 8]))
        >>> [x for x in add_negatives([pos1, pos2], 3, [(0, 100)] * 3)]
        [(array([[ 1,  2,  3],
               [ 1,  3,  2],
               [15,  2,  3],
               [72,  3,  2],
               [22,  3,  2]]), array([ 7.,  6.,  0.,  0.,  0.])), (array([[ 4,  5,  6],
               [ 7,  6,  7],
               [ 4,  5, 75],
               [ 7,  6,  7],
               [ 7,  6, 34]]), array([ 9.,  8.,  0.,  0.,  0.]))]
    """
    for input_pos, output_pos in positives:
        input_neg, output_neg = generate_negatives(input_pos, n_negatives, idx_ranges)
        yield np.concatenate([input_pos, input_neg]), np.concatenate([output_pos, output_neg])


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


def feed_dict_sampler(iterable, names=None, placeholders=None):
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
        placeholders = []
        for a, s in zip(first_items, names):
            if isinstance(a, str):  # just in case the input is a string, there is a TensorFlow type for that
                placeholders.append(tf.placeholder(tf.string, 1, name=s))
            else:  # most of the time, an array-like is provided
                a = np.array(a, dtype=np.float32)  # just in case the input is not an array
                placeholders.append(tf.placeholder(a.dtype.name, a.shape, name=s))

    dict_iterable = AutoReset(lambda: map(lambda items: dict(zip(placeholders, items)), iter(iterable)))
    return dict_iterable, placeholders


def tuple_sampler(tuples, minibatch_size, prop_negatives=0.5, idx_ranges=None):
    """

    Args:
        tuples: list of (input, output) pairs, where input is a tuple of integer indices, and output is a real number
        minibatch_size: integer giving the size of the minibatch
        prop_negatives: fraction of negative samples in every minibatch
        idx_ranges: array of pairs (a, b) of length L, where
            - L is the arity of the tuples,
            - (a, b) is the range of integer indices (low, high) to sample the negatives for the corresponding mode.

    By default, idx_range spans over all possible indices seen in the tuple dataset.

    Returns:
        iterator over positive and negative tuples

    Examples:
        >>> np.random.seed(1)
        >>> t = [((1, 2), 1), ((2, 3), 2), ((3, 1), 3), ((4, 2), 4), ((5, 4), 5)]
        >>> [x for x in tuple_sampler(t, 3)]
        [(array([[4, 2],
               [4, 0],
               [4, 0]]), array([ 4.,  0.,  0.])), (array([[3, 1],
               [3, 4],
               [3, 4]]), array([ 3.,  0.,  0.])), (array([[2, 3],
               [1, 3],
               [2, 3]]), array([ 2.,  0.,  0.])), (array([[1, 2],
               [2, 2],
               [4, 2]]), array([ 1.,  0.,  0.])), (array([[5, 4],
               [5, 4],
               [5, 2]]), array([ 5.,  0.,  0.]))]
    """
    if idx_ranges is None:
        arity = len(tuples[0][0])
        maxi = np.max([np.max(t) for t, v in tuples])
        idx_ranges = [(0, maxi)] * arity
    n_neg = np.floor(minibatch_size * prop_negatives) + (np.random.rand() < prop_negatives)
    n_pos = minibatch_size - n_neg

    for sample in add_negatives(positive_minibatch_generator(tuples, n_pos), n_negatives=n_neg, idx_ranges=idx_ranges):
        yield sample


def product_range(shape):
    """
    Iterator over all possible indices in a tensor
    Args:
        shape: shape of the tensor

    Returns:
        an iterator over the indices (tuples the same length as the shape)

    """
    pools = [tuple(range(n)) for n in shape]
    result = [[]]
    for pool in pools:
        result = [x + [y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)


def all_tensor_indices(shape, batch_size):
    """
    Iterator over all the tensor indices
    Args:
        shape: shape of the tensor
        batch_size: size of each tuple

    Returns:
        an iterator that returns batches of all the possible indices of a tensor of the specified shape.
        each iterator return batch_size of such indices in the rows of the tensor
        the last batch_size is smaller is batch_size is not a divisor of prod(shape)

    >>> a = [x for x in all_tensor_indices((2, 3, 4), batch_size=7)]
    >>> unique_rows(np.concatenate(a)).shape
    (24, 3)
    """

    p = product_range(shape)
    while True:
        a = []
        for i in range(batch_size):
            try:
                a.append(next(p))
            except StopIteration:
                if a:
                    yield np.array(a)
                raise StopIteration
        if a:
            yield np.array(a)


def round_robin(*iterables):
    """
    Alternate between multiple iterators
    Args:
        *iterables: list of iterators

    Returns:

    round_robin('ABC', 'D', 'EF') --> A D E B F C"

    """
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))

# deprecated samplers


def chunk_index_generator(n, b):
    for i in range(n // b):
        yield i * b, i * b + b
    yield n // b * b, n


def unique_differences(shape):
    if len(shape) == 1:
        for i in range(shape[0]):
            yield i,
    elif len(shape) == 2:
        for i in range(shape[0]):
            yield i, 0
        for i in range(shape[1] - 1):
            yield 0, i + 1
    else:
        raise NotImplementedError('does not handle tuples of arity>2')


def simple_tuple_generator(tuples, minibatch_size, n_iter, eval_freq, negative_prop, n_ent,
                           bigram=False, dictionaries=None):
    # the generator of minibatches
    loss_type = get_loss_type(tuples[0][1])  # takes the first tuple as a type example

    if not bigram:
        minibatch_generator, n_emb, arity, minibatch_size = \
            tuples_minibatch_generator(tuples, minibatch_size=minibatch_size, n_iter=n_iter, eval_freq=eval_freq,
                                       negative_prop=negative_prop, loss_type=loss_type, n_ent=n_ent)

    if bigram:
        minibatch_generator, n_emb, arity, minibatch_size = \
            bigram_minibatch_generator(tuples, minibatch_size=minibatch_size, n_iter=n_iter, eval_freq=eval_freq,
                                       negative_prop=negative_prop, loss_type=loss_type, n_ent=n_ent,
                                       dictionaries=dictionaries)

    # print("n_iter",n_iter)
    # model: inputs, outputs and parameters
    inputs = tf.placeholder("int32", [(1 + negative_prop) * minibatch_size, arity])
    outputs = tf.placeholder("float32", [(1 + negative_prop) * minibatch_size])
    # inputs = tf.placeholder("int32", [minibatch_size, arity])
    # outputs = tf.placeholder("float32", [minibatch_size])
    return inputs, outputs, minibatch_generator


def tuples_minibatch_generator(tuples, minibatch_size=100, n_iter=1000, eval_freq=50,
                               negative_prop=0.0, loss_type=0, n_ent=None):
    train_inputs = np.array([x for x, y in tuples])
    train_outputs = np.array([y for x, y in tuples])
    # print( "train_inputs.shape", train_inputs.shape)
    arity = train_inputs.shape[1]
    if n_ent == None:
        n_ent = np.max(train_inputs) + 1
    n_t = train_inputs.shape[0]
    minibatch_size = min(minibatch_size, n_t)  # for small datasets
    minibatch_type = loss_type

    if negative_prop > 0:  # Pre-allocation, avoid memory alloc at each batch generation
        new_train_inputs = np.empty((minibatch_size * (negative_prop + 1), train_inputs.shape[1]))
        new_train_outputs = np.empty(minibatch_size * (negative_prop + 1))

    def next_minibatch():  # a generator function
        # nonlocal train_inputs, train_outputs, minibatch_size, n_iter, eval_freq, n_t
        # global train_inputs, train_outputs, minibatch_size, eval_freq, n_t
        epoch = 0
        while epoch < n_iter:
            minibatch_indices, n_rem = create_minibatch_indices(n_t, minibatch_size)
            for ids in minibatch_indices:
                if negative_prop > 0:  # Negative generation

                    this_batch_size = len(ids)  # Manage shorter batches (sometimes last ones are shorter)

                    # Pre copying everyting as many times as necessary
                    new_train_inputs[:(this_batch_size * (negative_prop + 1)), :] = np.tile(train_inputs[ids, :],
                                                                                            (negative_prop + 1, 1))
                    new_train_outputs[:(this_batch_size * (negative_prop + 1))] = np.tile(train_outputs[ids],
                                                                                          negative_prop + 1)
                    # Pre-sample everything, faster
                    rdm_entities = np.random.randint(0, n_ent, this_batch_size * negative_prop)
                    rdm_choices = np.random.random(this_batch_size * negative_prop)

                    for i in range(this_batch_size):
                        # for i in range(np.floor(this_batch_size/float(negative_prop))):

                        for j in range(negative_prop):
                            cur_idx = i * negative_prop + j

                            if rdm_choices[cur_idx] < 0.5:
                                new_train_inputs[this_batch_size + cur_idx, 0] = rdm_entities[cur_idx]
                            else:
                                # Here train_inputs.shape[-1]-1 makes negative generation generic for matrices and
                                # tensors: the altered index is either
                                # the first or last dimension to alter the input indexes
                                new_train_inputs[this_batch_size + cur_idx, train_inputs.shape[-1] - 1] = rdm_entities[
                                    cur_idx]

                            new_train_outputs[this_batch_size + cur_idx] = 0.0

                    minibatch_inputs = new_train_inputs[:this_batch_size * (negative_prop + 1),
                                       :]  # truncate arrays in case of shorter batch
                    minibatch_outputs = new_train_outputs[:this_batch_size * (negative_prop + 1)]
                else:  # No negative generation
                    minibatch_inputs = train_inputs[ids, :]
                    minibatch_outputs = train_outputs[ids]

                eval_step = epoch % eval_freq == 0 or epoch == n_iter - 1
                yield epoch, eval_step, (minibatch_inputs, minibatch_outputs, minibatch_type)
                epoch += 1

    return next_minibatch, n_ent, arity, minibatch_size


# Bigram minibatch generator. Identical to tuples_minibatch_generator, but different negative sampling.
def bigram_minibatch_generator(tuples, minibatch_size=100, n_iter=1000, eval_freq=50,
                               negative_prop=0.0, loss_type=0, n_ent=None,
                               dictionaries=None):
    # use dictionaries for negative sampling:
    (Global_Dict, D_ent, D_rel) = dictionaries
    Tuple = namedtuple("Tuple", ["subj", "rel", "obj"])

    train_inputs = np.array([x for x, y in tuples])
    train_outputs = np.array([y for x, y in tuples])
    print("train_inputs.shape", train_inputs.shape)
    arity = train_inputs.shape[1]  # tuple length
    if n_ent == None:
        # n_ent = np.max(train_inputs) + 1
        n_ent = len(D_ent)
    n_t = train_inputs.shape[0]
    minibatch_size = min(minibatch_size, n_t)  # for small datasets
    minibatch_type = loss_type

    if negative_prop > 0:  # Pre-allocation, avoid memory alloc at each batch generation
        new_train_inputs = np.empty((minibatch_size * (negative_prop + 1), train_inputs.shape[1]))
        new_train_outputs = np.empty(minibatch_size * (negative_prop + 1))

    def next_minibatch():  # a generator function
        # nonlocal train_inputs, train_outputs, minibatch_size, n_iter, eval_freq, n_t
        # global train_inputs, train_outputs, minibatch_size, eval_freq, n_t
        epoch = 0
        while epoch < n_iter:
            minibatch_indices, n_rem = create_minibatch_indices(n_t, minibatch_size)

            for ids in minibatch_indices:

                if negative_prop > 0:  # Negative generation
                    # TODO
                    this_batch_size = len(ids)  # Manage shorter batches (sometimes last ones are shorter)

                    # Pre copying everyting as many times as necessary
                    new_train_inputs[:(this_batch_size * (negative_prop + 1)), :] = np.tile(train_inputs[ids, :],
                                                                                            (negative_prop + 1, 1))
                    new_train_outputs[:(this_batch_size * (negative_prop + 1))] = np.tile(train_outputs[ids],
                                                                                          negative_prop + 1)

                    # Pre-sample everything, faster
                    rdm_entities = np.random.randint(1, n_ent, this_batch_size * negative_prop)
                    rdm_choices = np.random.random(this_batch_size * negative_prop)

                    for i in range(this_batch_size):
                        # for i in range(np.floor(this_batch_size/float(negative_prop))):

                        for j in range(negative_prop):
                            cur_idx = i * negative_prop + j

                            if rdm_choices[cur_idx] < 0.5:
                                subj_idx = rdm_entities[cur_idx]
                                new_train_inputs[this_batch_size + cur_idx, 0] = subj_idx

                                # adapt bigram tuple entries accordingly, first get strings
                                subj_str = D_ent[subj_idx]
                                verb_str = D_rel[new_train_inputs[this_batch_size + cur_idx, 1]]
                                obj_str = D_ent[new_train_inputs[this_batch_size + cur_idx, 2]]

                                # now use strings to obtain pairwise interaction entries
                                i01 = Global_Dict[Tuple(subj=subj_str, rel=verb_str, obj=None)]
                                i02 = Global_Dict[Tuple(subj=subj_str, rel=None, obj=obj_str)]

                                # set tuple entries for interactions
                                new_train_inputs[this_batch_size + cur_idx, 3] = i01
                                new_train_inputs[this_batch_size + cur_idx, 5] = i02



                            else:
                                obj_idx = rdm_entities[cur_idx]
                                new_train_inputs[this_batch_size + cur_idx, 2] = obj_idx

                                # adapt bigram tuple entries accordingly, first get strings
                                subj_str = D_ent[new_train_inputs[this_batch_size + cur_idx, 0]]
                                verb_str = D_rel[new_train_inputs[this_batch_size + cur_idx, 1]]
                                obj_str = D_ent[obj_idx]

                                # now use strings to obtain pairwise interaction entries
                                i02 = Global_Dict[Tuple(subj=subj_str, rel=None, obj=obj_str)]
                                i12 = Global_Dict[Tuple(subj=None, rel=verb_str, obj=obj_str)]

                                # set tuple entries for interactions
                                new_train_inputs[this_batch_size + cur_idx, 5] = i02
                                new_train_inputs[this_batch_size + cur_idx, 4] = i12

                            new_train_outputs[this_batch_size + cur_idx] = 0.0

                    minibatch_inputs = new_train_inputs[:this_batch_size * (negative_prop + 1),
                                       :]  # truncate arrays in case of shorter batch
                    minibatch_outputs = new_train_outputs[:this_batch_size * (negative_prop + 1)]
                else:  # No negative generation
                    minibatch_inputs = train_inputs[ids, :]
                    minibatch_outputs = train_outputs[ids]

                eval_step = epoch % eval_freq == 0 or epoch == n_iter - 1
                yield epoch, eval_step, (minibatch_inputs, minibatch_outputs, minibatch_type)
                epoch += 1

    return next_minibatch, n_ent, arity, minibatch_size


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












        # if __name__ == '__main__':
        #     pass
        # l = ['ABCDEF', 'GH']
        # print([x for x in roundrobin(*l)])
        # [print(x) for x in chunk_index_generator(100, 7)]
        # [print(x, x[1]-x[0]) for x in unique_differences((2, 3))]
        # a = [x for x in all_tensor_indices((10, 9), 77)]
        # [print(x) for x in product_range((3, 4, 5))]

# arity = len(shape)
# ord = np.argsort(shape)
# shape_ord = [shape[i] for i in ord]
# maxi = shape_ord[-1]
# indices = np.zeros((0, arity))
# c=0
# for start_idx in unique_differences(shape_ord[0:-1]):
#     for start, end in chunk_index_generator(maxi, batch_size):
#         rep = 0
#         stop = False
#         while not stop:
#             idx0 = np.arange(start + 1, end + 1 + rep * batch_size)
#             arr = []
#             for j in range(arity - 1):
#                 arr.append((start_idx[j] + idx0) % shape_ord[j])
#             arr.append(idx0 % shape_ord[-1])
#             indices = np.concatenate((indices, np.vstack(arr).T), axis=0)
#             print(indices)
#             print(tuple(indices[-1][ord[:-1]]), start_idx)
#             if tuple(indices[-1]) == start_idx:  #last index arrives at the first
#                 stop = True
#             if indices.shape[0] >= batch_size:
#                 yield indices[0:batch_size]
#                 indices = indices[batch_size:]
# yield indices
