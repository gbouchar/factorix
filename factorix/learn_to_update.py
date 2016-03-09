# Test the larcqy model: l(a(r(c),q),y)

from typing import Tuple
import numpy as np
import tensorflow as tf
# from naga.shared.learning import learn, data_to_batches, placeholder_feeder, feed_dict_sampler, OperatorHook
# from factorix.Dictionaries import NEW_ID
from factorix.losses import loss_quadratic_grad
from factorix.scoring import multilinear, multilinear_grad
# from naga.shared.tf_addons import tf_eval, tf_show, tf_debug_gradient


def force_list_length(l, n):
    if len(l) > n:
        return l[0:n]
    elif len(l) < n:
        return l + [l[0] for _ in range(n - len(l))]
    else:
        return l


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


