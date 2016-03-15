# Test the larcqy model: l(a(r(c),q),y)

from typing import Tuple
import numpy as np
import tensorflow as tf
from naga.shared.learning import learn, data_to_batches, placeholder_feeder
from factorix.scoring import multilinear, multilinear_grad
# from factorix.Dictionaries import NEW_ID
# from factorix.losses import loss_quadratic_grad
# from naga.shared.tf_addons import tf_eval, tf_show, tf_debug_gradient
from factorix.losses import loss_quadratic_grad, total_loss_logistic
from naga.shared.tf_addons import tf_eval, tf_show


def embedding_updater_model(variables, rank,
                            n_slots,
                            init_params=None,
                            n_ents=None,
                            init_noise=0.0,
                            loss=total_loss_logistic,
                            scoring=multilinear,
                            reg=0.0):
    qc, yc, wc, q, y = variables
    # model definition
    # initialization
    if init_params is not None:
        emb0_val = init_params[0]
        emb0_val += np.random.randn(n_ents, rank) * init_noise
    else:
        emb0_val = np.random.randn(n_ents, rank)
    emb0 = tf.Variable(np.array(emb0_val, dtype=np.float32))

    # reading and answering steps
    emb1 = reader(emb0=emb0, context=(qc, yc), weights=wc, n_slots=n_slots, loss_grad=loss_quadratic_grad)
    pred = answerer(emb1, q, scoring=scoring)
    objective = loss(pred, y)
    if reg > 0:
        objective += reg * tf.nn.l2_loss(emb0)

    return objective, pred, y


def multitask_to_tuples(x: np.ndarray, y: np.ndarray, intercept=True):
    if y.ndim == 1:
        y = y.reshape((-1, 1))
    k = y.shape[1]
    data = []
    for i in range(x.shape[0]):
        if intercept:
            inputs = [((0, k + 1), 1.0)] + [((0, j + k + 2), x[i, j]) for j in range(x.shape[1])]
        else:
            inputs = [((0, j + k + 1), x[i, j]) for j in range(x.shape[1])]
        outputs = [((0, k + 1), y[i, k]) for k in range(y.shape[1])]
        data.append((inputs, outputs))
    return data


# class LogisticRegressionEmbeddingUpdater(EmbeddingUpdater):
#     def __init__(self, rank, n_ents, reg, n_slots=1, max_epochs=500, verbose=True, preprocessing=None):
#

class EmbeddingUpdater(object):
    def __init__(self, rank, n_ents, reg, n_slots=1, max_epochs=500, verbose=True, preprocessing=None):
        self.verbose = verbose
        self.rank = rank
        self.n_ents = n_ents
        self.n_slots = n_slots
        self.max_epochs = max_epochs
        self.reg = reg*0.001
        self.params = None
        self.preprocessing = preprocessing

    def logistic2embeddings(self, coefs, intercept=0.0):
        self.params = [np.array([[0.0, 1.0, intercept] + coefs.ravel().tolist()], dtype='float32').T]

    def fit(self, data_train, *args):
        if self.preprocessing:
            data_train = self.preprocessing(data_train, *args)
        with tf.Graph().as_default() as _:
            # create sampler and variables
            variables, sampler = machine_reading_sampler(data_train, batch_size=None)
            # main graph
            objective, _, _ = embedding_updater_model(variables, rank=self.rank,
                                                      n_ents=self.n_ents, n_slots=self.n_slots, reg=self.reg)
            # tf_debug_gradient(emb0, objective, verbose=False)  # This creates new variables...
            # train the model
            optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
            hooks = []
            if self.verbose:
                hooks += [lambda it, e, xy, f: it and ((it % 1) == 0) and print("%d) loss=%f" % (it, f[0]))]
            self.params = learn(objective, sampler, optimizer=optimizer, hooks=hooks, max_epochs=self.max_epochs)

    def predict(self, data, *args):
        if self.preprocessing:
            data = self.preprocessing(data, *args)
        with tf.Graph().as_default() as _:
            variables_test, sampler_test = machine_reading_sampler(data, batch_size=None, shuffling=False)
            ops = embedding_updater_model(variables_test, rank=self.rank, n_ents=self.n_ents, n_slots=self.n_slots,
                                          init_params=self.params)
            nll, pred, y = tf_eval(ops)
        return pred, y, nll

    @property
    def coef_(self):
        return self.params[0][0] * self.params[0][2:]

    @property
    def intercept_(self):
        return self.params[0][0] * self.params[0][1]




def force_list_length(l, n):
    if len(l) > n:
        return l[0:n]
    elif len(l) < n:
        return l + [l[0] for _ in range(n - len(l))]
    else:
        return l


def machine_reading_sampler(data, batch_size=None, n_ents=None, shuffling=True):
    data_arr = vectorize_samples(data)
    if batch_size is not None:
        batches = data_to_batches(data_arr, batch_size, dtypes=[np.int64, np.float32, np.float32, np.int64, np.float32],
                                  shuffling=shuffling)
        qc = tf.placeholder(np.int64, (batch_size, n_ents, 2), name='question_in_context')
        yc = tf.placeholder(np.float32, (batch_size, n_ents), name='answer_in_context')
        wc = tf.placeholder(np.float32, (batch_size, n_ents), name='answer_in_context')
        q = tf.placeholder(np.int64, (batch_size, 1, 2), name='question')
        y = tf.placeholder(np.float32, (batch_size, 1), name='answer')
        sampler = placeholder_feeder((qc, yc, wc, q, y), batches)
    else:
        batches = data_to_batches(data_arr, len(data), dtypes=[np.int64, np.float32, np.float32, np.int64, np.float32],
                                  shuffling=shuffling)
        qc0, yc0, wc0, q0, y0 = [x for x in batches][0]
        qc = tf.Variable(qc0, trainable=False)
        yc = tf.Variable(yc0, trainable=False)
        wc = tf.Variable(wc0, trainable=False)
        q = tf.Variable(q0, trainable=False)
        y = tf.Variable(y0, trainable=False)
        sampler = None
    return (qc, yc, wc, q, y), sampler


def vectorize_samples(data, max_context_length=None):
    if max_context_length is None:
        max_context_length = np.max([len(d[0]) for d in data])
    arr = []
    for d in data:
        c, qa = d
        l = min(len(c), max_context_length)
        qc_data = np.array(force_list_length([[idx for idx in ex[0][0:l]] for ex in c], max_context_length))
        yc_data = np.array([ex[1] for ex in c[0:l]] + [0.0 for _ in range(max_context_length - l)])
        wc_data = np.array([1.0 for _ in c[0:l]] + [0.0 for _ in range(max_context_length - l)])
        q_data = np.array([[idx for idx in ex[0]] for ex in qa])
        y_data = np.array([ex[1] for ex in qa])
        arr.append((qc_data, yc_data, wc_data, q_data, y_data))
    return arr


def reader(context: Tuple[tf.Variable, tf.Variable], emb0: tf.Variable, n_slots: int,
           weights=None,
           loss_grad=loss_quadratic_grad,
           emb_update=multilinear_grad):
    """
    Read a series of data and update the embeddings accordingly
    Args:
        context (Tuple[tf.Variable, tf.Variable]): contextual information
        emb0 (tf.Variable): initial embeddings
        n_slots (int): number of slots to update
        weights: weights give to every observation in the inputs. Size: (batch_size, n_obs)
        loss_grad: gradient of the loss
        emb_update: update of the embeddings (could be the gradient of the score with respect to the embeddings)

    Returns:
        The variable representing updated embeddings
    """
    slot_dim = 0
    if context is None:  # empty contexts are not read
        return emb0

    context_inputs, context_ouputs = context
    n_data, n_obs, order = [d.value for d in context_inputs.get_shape()]
    rank = emb0.get_shape()[1].value
    shift_indices = tf.constant(
            n_slots * np.reshape(np.outer(range(n_data), np.ones(n_obs)), (n_data, n_obs)), dtype='int64')
    step_size = tf.Variable(1.0, name='step_size', trainable=False)

    grad_score, preds = emb_update(emb0, context_inputs, score=True)
    update_strength = tf.tile(tf.reshape(loss_grad(preds, context_ouputs) * weights,
                                         (n_data, n_obs, 1)), (1, 1, rank))

    grad_loss = tf.reshape(grad_score, (n_data, n_obs, rank)) * update_strength

    if False:  # legacy code that might not work (was not working anyway due to the scatter_add that need reset)
        zeros0 = tf.Variable(np.zeros((n_data * n_slots, rank), dtype=np.float32),
                             name='initial_slot_embeddings', trainable=False)

        zeros = tf.assign(zeros0, np.zeros((n_data * n_slots, rank), dtype=np.float32))
        indices = context_inputs[:, :, slot_dim] + shift_indices
        total_grad_loss = tf.reshape(tf.scatter_add(zeros, indices, grad_loss), (n_data, n_obs, n_slots, rank))
        # could also try tf.dynamic_partition(data, context_inputs[:, :, slot_dim], num_partitions)
    else:
        one_hot = tf.Variable(np.eye(n_slots, n_slots, dtype=np.float32), trainable=False)
        indic_mat = tf.gather(one_hot, context_inputs[:, :, slot_dim])  # shape: (n_data, n_obs, n_slots)
        total_grad_loss = tf.batch_matmul(indic_mat, grad_loss, adj_x=True)

    initial_slot_embs = tf.reshape(tf.tile(emb0[:n_slots, :], (n_data, 1)), (n_data, n_slots, rank))
    return initial_slot_embs - total_grad_loss * step_size  # size of the output: (n_data, n_slots, rank)


def answerer(embeddings, tuples: tf.Variable, scoring=multilinear):
    """
    Evaluate the score of tuples with embeddings that are specific to every data sample

    Args:
        embeddings (tf.Variable): embedding tensor with shape (n_data, n_slots, rank)
        tuples: question tensor with int64 entries and shape (n_data, n_tuples, order)
        scoring: operator that is used to compute the scores

    Returns:
        scores (tf.Tensor): scores tensor with shape (n_data, n_tuples)

    """
    n_data, n_slots, rank = [d.value for d in embeddings.get_shape()]
    n_data, n_tuples, order = [d.value for d in tuples.get_shape()]

    shift_indices = tf.constant(np.reshape(
            np.outer(range(n_data), np.ones(n_tuples * n_slots)) * n_slots, (n_data, n_tuples, n_slots)), dtype='int64')
    questions_shifted = tuples + shift_indices

    preds = scoring(
            tf.reshape(embeddings, (n_data * n_slots, rank)),
            tf.reshape(questions_shifted, (n_data * n_tuples, order)))

    return tf.reshape(preds, (n_data, n_tuples))


