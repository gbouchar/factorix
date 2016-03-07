# Test the larcqy model: l(a(r(c),q),y)


from typing import Tuple
import numpy as np
import tensorflow as tf
from naga.shared.learning import learn, data_to_batches, placeholder_feeder, feed_dict_sampler, OperatorHook
from factorix.Dictionaries import NEW_ID
from factorix.losses import total_loss_quadratic, loss_quadratic_grad, total_loss_logistic
from factorix.scoring import multilinear, multilinear_grad
from naga.shared.tf_addons import tf_eval, tf_show, tf_debug_gradient


def test_matrix_factorization(verbose=False):
    np.random.seed(1)
    n, m, rank = 7, 6, 3
    mat = np.random.randn(n, rank).dot(np.random.randn(rank, m))
    tuples = [([i, n + j], mat[i, j]) for i in range(n) for j in range(m)]
    tuple_iterable = data_to_batches(tuples, minibatch_size=n * m)
    sampler, (x, y) = feed_dict_sampler(tuple_iterable, types=[np.int64, np.float32])
    emb_var = tf.Variable(tf.cast(np.random.randn(n + m, rank), 'float32'))
    offset = tf.Variable(tf.cast(1.0, 'float32'))
    loss_op = tf.reduce_mean(tf.square(tf.reduce_sum(tf.reduce_prod(tf.gather(emb_var, x), 1), 1) + offset - y))
    emb, offset_val = learn(loss_op, sampler, max_epochs=200, variables=[emb_var, offset])
    mat_est = emb[:n, :].dot(emb[n:, :].T)
    if verbose:
        print(np.linalg.norm(mat_est - mat) ** 2)  # we should have recovered the low-rank matrix
    else:
        assert (np.linalg.norm(mat_est - mat) < 1e-3)


def vectorize_samples(data):
    arr = []
    for d in data:
        qc_data = np.array([[idx for idx in ex[0]] for ex in d[0]])
        yc_data = np.array([ex[1] for ex in d[0]])
        q_data = np.array([[idx for idx in ex[0]] for ex in d[1]])
        y_data = np.array([ex[1] for ex in d[1]])
        arr.append((qc_data, yc_data, q_data, y_data))
    return arr


def test_multitask_learning(verbose=False):
    # factorization of the parameters with multiple linear regressions
    np.random.seed(1)
    n_data, n_features, n_labels, rank = 6, 5, 4, 3
    batch_size = n_data
    n_emb = 1 + n_features + n_labels

    x_mat = np.random.randn(n_data, n_features)
    w_mat = np.random.randn(n_features, rank).dot(np.random.randn(rank, n_labels))
    y_mat = x_mat.dot(w_mat)

    if False:
        x = tf.placeholder(np.float32, [batch_size, n_features], name='answer')
        y = tf.placeholder(np.float32, [batch_size, n_labels], name='answer')
        sampler = placeholder_feeder((x, y), [(x_mat, y_mat)])  # only one sample
        emb = tf.Variable(tf.cast(np.random.randn(n_features + n_labels, rank), 'float32'))
        w = tf.matmul(emb[:n_features, :], emb[n_features:, :], transpose_b=True)
        preds = tf.matmul(x, w)
        objective = tf.reduce_sum(tf.square(preds - y))
        hooks = [lambda it, e, xy, f: it and ((it % 10) == 0) and print("%d) loss=%f" % (it, f))]
        emb_val, = learn(objective, sampler, hooks=hooks, max_epochs=2000, variables=[emb])
        mat_est = x_mat.dot(emb_val[:n_features, :]).dot(emb_val[n_features:, :].T)
    else:
        data = []
        for i in range(n_data):
            data.append(([((NEW_ID, j + 1), x_mat[i, j]) for j in range(n_features)],
                         [((NEW_ID, k + 1 + n_features), y_mat[i, k]) for k in range(n_labels)]))
        data_arr = vectorize_samples(data)
        batches = data_to_batches(data_arr, minibatch_size=batch_size)
        qc = tf.placeholder(np.int64, [batch_size, n_features, 2], name='question_in_context')
        yc = tf.placeholder(np.float32, [batch_size, n_features], name='answer_in_context')
        q = tf.placeholder(np.int64, [batch_size, n_labels, 2], name='question')
        y = tf.placeholder(np.float32, [batch_size, n_labels], name='answer')
        sampler = placeholder_feeder((qc, yc, q, y), batches)
        [print(s) for s in sampler]
        emb0 = tf.Variable(tf.cast(np.random.randn(n_emb, rank), 'float32'))

        def reader(context_inputs, context_ouputs):
            context_embs = tf.gather(emb0, context_inputs[:, :, 1])
            # preds = tf.reshape(tf.matmul(
            #             tf.reshape(context_embs, (n_data * n_features, rank)),
            #             tf.reshape(emb0[0, :], [rank, 1])),
            #         (n_data, n_features))
            # residues = tf.tile(tf.reshape(preds - yc, (n_data, n_features, 1)), [1, 1, rank])
            # embs_after_reading = tf.tile(tf.reshape(emb0[0, :], (1, rank)), (n_data, 1)) \
            #     + tf.reduce_mean(context_embs * residues, 1) * step_size
            # step_size = tf.Variable(tf.cast(1.0, 'float32'), trainable=True)
            yc_rep = tf.tile(tf.reshape(context_ouputs, (n_data, n_features, 1)), (1, 1, rank))
            embs_after_reading = 0 * tf.tile(tf.reshape(emb0[0, :], (1, rank)), (n_data, 1)) \
                                 + tf.reduce_sum(context_embs * yc_rep, 1)  # * step_size
            return embs_after_reading

        def answerer(embeddings, question):
            embs_after_reading_mat = tf.tile(tf.reshape(embeddings, [n_data, 1, 1, rank]), [1, n_labels, 1, 1])
            fixed_embs = tf.reshape(tf.gather(emb0, question[:, :, 1]), [n_data, n_labels, 1, rank])
            emb1_question = tf.concat(2, [fixed_embs, embs_after_reading_mat])
            return tf.reduce_sum(tf.reduce_prod(emb1_question, 2), 2)

        def loss(pred, gold):
            return tf.nn.l2_loss(pred - gold)

        objective = loss(answerer(reader(qc, yc), q), y)

        hooks = [lambda it, e, xy, f: it and ((it % 10) == 0) and print("%d) loss=%f" % (it, f))]
        emb_val, = learn(objective, sampler, hooks=hooks, max_epochs=2000, variables=[emb0])
        mat_est = x_mat.dot(emb_val[1:n_features + 1, :]).dot(emb_val[n_features + 1:, :].T)
    if verbose:
        print(0.5 * np.linalg.norm(mat_est - y_mat) ** 2)  # we should have recovered the low-rank matrix
    else:
        assert (np.linalg.norm(mat_est - y_mat) < 1e-3)


class MultiTaskReaderAnswerer(object):
    def __init__(self, n_data, n_slots, n_features, n_labels, rank):
        n_embs = n_features + n_labels + 1
        self.n_data, self.n_features, self.n_labels, self.rank = n_data, n_features, n_labels, rank
        self.emb0 = tf.Variable(tf.cast(np.random.randn(n_embs, rank), 'float32'))
        self.step_size = tf.Variable(1.0)

    def reader(self, context=None, emb0=None):
        if emb0 is None:  # by default, can use the initial embedding
            emb0 = self.emb0
        if context is None:  # empty contexts are not read
            return emb0
        context_inputs, context_ouputs = context
        context_embs = tf.gather(emb0, context_inputs[:, :, 1])
        preds = tf.reshape(tf.matmul(
                tf.reshape(context_embs, (self.n_data * self.n_features, self.rank)),
                tf.reshape(emb0[0, :], [self.rank, 1])),
                (self.n_data, self.n_features))
        update_strength = tf.tile(tf.reshape(loss_quadratic_grad(preds, context_ouputs),
                                             (self.n_data, self.n_features, 1)), (1, 1, self.rank))
        embs_after_reading = tf.tile(tf.reshape(emb0[0, :], (1, self.rank)), (self.n_data, 1)) \
                             - tf.reduce_sum(context_embs * update_strength, 1) * self.step_size
        return embs_after_reading  # size of the output: (n_data, rank)

    def answerer(self, embeddings, question):
        embs_after_reading_mat = tf.tile(tf.reshape(embeddings, [self.n_data, 1, 1, self.rank]),
                                         [1, self.n_labels, 1, 1])
        fixed_embs = tf.reshape(tf.gather(self.emb0, question[:, :, 1]), [self.n_data, self.n_labels, 1, self.rank])
        emb1_question = tf.concat(2, [fixed_embs, embs_after_reading_mat])
        preds = tf.reduce_sum(tf.reduce_prod(emb1_question, 2), 2)
        return preds

    def numeric_eval(self, parameters, x_mat):
        emb_val = parameters[0]
        step_size_val = parameters[1]
        return x_mat.dot(emb_val[1:self.n_features + 1, :]).dot(emb_val[self.n_features + 1:, :].T) * step_size_val


def reader(context: Tuple[tf.Variable, tf.Variable], emb0: tf.Variable, n_slots: int,
           loss_grad=loss_quadratic_grad,
           emb_update=multilinear_grad):
    """
    Read a series of data and update the embeddings accordingly
    Args:
        context:
        emb0:
        n_slots:
        emb_update:

    Returns:

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
    update_strength = tf.tile(tf.reshape(loss_grad(preds, context_ouputs),
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
    return initial_slot_embs * 0 - total_grad_loss * step_size  # size of the output: (n_data, n_slots, rank)


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


def test_larcqy(verbose=False):
    # factorization of the parameters with multiple linear regressions

    # input_types = {'index', 'features'}
    input_types = {'index'}
    input_types = {'features'}
    n1, n2, d1, d2, rank_gold = 7, 6, 5, 4, 3

    # random data generation
    np.random.seed(1)
    emb_noise, noise = 1, 0
    batch_size = n1 * n2
    t = lambda x: np.round(x, 1)
    data_emb1 = t(np.random.randn(n1, rank_gold) * emb_noise)
    data_emb2 = t(np.random.randn(n2, rank_gold) * emb_noise)
    feat_emb1 = t(np.random.randn(d1, rank_gold) * emb_noise)
    feat_emb2 = t(np.random.randn(d2, rank_gold) * emb_noise)
    x1_mat = data_emb1.dot(feat_emb1.T)
    x2_mat = data_emb2.dot(feat_emb2.T)
    a1 = x1_mat.dot(feat_emb1)
    a2 = x2_mat.dot(feat_emb2)
    y_mat = a1.dot(a2.T) + np.random.randn(n1, n2) * noise

    # data stuff
    data = []
    n_ents = 2
    for i in range(n1):
        for j in range(n2):
            inputs = []
            if 'features' in input_types:
                inputs += [((0, k + 2), x1_mat[i, k]) for k in range(d1)] \
                          + [((1, k + 2 + d1), x2_mat[j, k]) for k in range(d2)]
                n_ents = d1 + d2 + 2
            if 'index' in input_types:
                inputs += [((0, d1 + d2 + 2 + i), 1.0), ((1, d1 + d2 + 2 + n1 + j), 1.0)]
                n_ents = d1 + d2 + n1 + n2 + 2
            outputs = [((0, 1), y_mat[i, j])]
            data.append((inputs, outputs))
    data_arr = vectorize_samples(data)
    batches = data_to_batches(data_arr, batch_size, dtypes=[np.int64, np.float32, np.int64, np.float32])
    if False:
        qc = tf.placeholder(np.int64, (batch_size, n_ents, 2), name='question_in_context')
        yc = tf.placeholder(np.float32, (batch_size, n_ents), name='answer_in_context')
        q = tf.placeholder(np.int64, (batch_size, 1, 2), name='question')
        y = tf.placeholder(np.float32, (batch_size, 1), name='answer')
        sampler = placeholder_feeder((qc, yc, q, y), batches)
        sampler = [x for x in sampler]
    else:
        qc0, yc0, q0, y0 = [x for x in batches][0]
        # print(qc0, yc0, q0, y0)
        qc = tf.Variable(qc0)
        yc = tf.Variable(yc0)
        q = tf.Variable(q0)
        y = tf.Variable(y0)

    # model definition
    rank = min(rank_gold, max(min(n1, n2), min(d1, d2)))
    print(rank)
    if False:
        emb0_val = np.concatenate((np.zeros((2, rank)), feat_emb1, feat_emb2))
        emb0_val += np.random.randn(n_ents, rank) * 0.1
        # emb0_val = np.round(emb0_val, 1)
    else:
        emb0_val = np.random.randn(n_ents, rank)
    emb0 = tf.constant(emb0_val.tolist(), dtype=np.float32, shape=(n_ents, rank))
    loss = total_loss_quadratic

    # reading step
    emb1 = reader(emb0=emb0, context=(qc, yc), n_slots=2)
    # answering step
    objective = loss(answerer(emb1, q), y)  # + 1e-6 * tf.nn.l2_loss(emb0)

    # tf_debug_gradient(emb0, objective, verbose=False)  # This creates new variables...

    # train the model
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    hooks = [lambda it, e, xy, f: it and ((it % 100) == 0) and print("%d) loss=%f" % (it, f[0]))]
    # hooks += [OperatorHook(tf_show(grads[0]), sampler[0][1])]
    params = learn(objective, optimizer=optimizer, hooks=hooks, max_epochs=300, variables=[emb0])

    # mat_est = model.numeric_eval(params, x_mat)
    #
    if verbose:
        #     print(0.5 * np.linalg.norm(mat_est - y_mat) ** 2)  # we should have recovered the low-rank matrix
        print(params[-1])
        pass
        # else:
        #     assert (np.linalg.norm(mat_est - y_mat) < 1e-3)


def test_larcqy_logistic(verbose=False):
    # factorization of the parameters with multiple linear regressions

    # input_types = {'index', 'features'}
    input_types = {'index'}
    input_types = {'features'}
    n1, n2, d1, d2, rank_gold = 7, 6, 5, 4, 3
    scoring = multilinear


    # random data generation
    np.random.seed(1)
    emb_noise, noise = 1, 0
    batch_size = n1 * n2
    t = lambda x: np.round(x, 1)
    data_emb1 = t(np.random.randn(n1, rank_gold) * emb_noise)
    data_emb2 = t(np.random.randn(n2, rank_gold) * emb_noise)
    feat_emb1 = t(np.random.randn(d1, rank_gold) * emb_noise)
    feat_emb2 = t(np.random.randn(d2, rank_gold) * emb_noise)
    x1_mat = data_emb1.dot(feat_emb1.T)
    x2_mat = data_emb2.dot(feat_emb2.T)
    a1 = x1_mat.dot(feat_emb1) > 0
    a2 = x2_mat.dot(feat_emb2) > 0
    y_mat = a1.dot(a2.T) + np.random.randn(n1, n2) * noise > 0

    # data stuff
    data = []
    n_ents = 2
    for i in range(n1):
        for j in range(n2):
            inputs = []
            if 'features' in input_types:
                inputs += [((0, k + 2), x1_mat[i, k]) for k in range(d1)] \
                         + [((1, k + 2 + d1), x2_mat[j, k]) for k in range(d2)]
                n_ents = d1 + d2 + 2
            if 'index' in input_types:
                inputs += [((0, d1 + d2 + 2 + i), 1.0), ((1, d1 + d2 + 2 + n1 + j), 1.0)]
                n_ents = d1 + d2 + n1 + n2 + 2
            outputs = [((0, 1), y_mat[i, j])]
            data.append((inputs, outputs))
    data_arr = vectorize_samples(data)
    batches = data_to_batches(data_arr, batch_size, dtypes=[np.int64, np.float32, np.int64, np.float32])
    if False:
        qc = tf.placeholder(np.int64, (batch_size, n_ents, 2), name='question_in_context')
        yc = tf.placeholder(np.float32, (batch_size, n_ents), name='answer_in_context')
        q = tf.placeholder(np.int64, (batch_size, 1, 2), name='question')
        y = tf.placeholder(np.float32, (batch_size, 1), name='answer')
        sampler = placeholder_feeder((qc, yc, q, y), batches)
        sampler = [x for x in sampler]
    else:
        qc0, yc0, q0, y0 = [x for x in batches][0]
        # print(qc0, yc0, q0, y0)
        qc = tf.Variable(qc0)
        yc = tf.Variable(yc0)
        q = tf.Variable(q0)
        y = tf.Variable(y0)

    # model definition
    rank = min(rank_gold, max(min(n1, n2), min(d1, d2)))
    print(rank)
    if False:
        emb0_val = np.concatenate((np.zeros((2, rank)), feat_emb1, feat_emb2))
        emb0_val += np.random.randn(n_ents, rank) * 0.1
        # emb0_val = np.round(emb0_val, 1)
    else:
        emb0_val = np.random.randn(n_ents, rank)
    emb0 = tf.constant(emb0_val.tolist(), dtype=np.float32, shape=(n_ents, rank))
    loss = total_loss_logistic

    # reading step
    emb1 = reader(emb0=emb0, context=(qc, yc), n_slots=2, loss_grad=loss_quadratic_grad)
    # answering step
    objective = loss(answerer(emb1, q), y)  # + 1e-6 * tf.nn.l2_loss(emb0)

    # tf_debug_gradient(emb0, objective, verbose=False)  # This creates new variables...

    # train the model
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    hooks = [lambda it, e, xy, f: it and ((it % 10) == 0) and print("%d) loss=%f" % (it, f[0]))]
    # hooks += [OperatorHook(tf_show(grads[0]), sampler[0][1])]
    params = learn(objective, optimizer=optimizer, hooks=hooks, max_epochs=500, variables=[emb0])

    # mat_est = model.numeric_eval(params, x_mat)
    #
    if verbose:
        #     print(0.5 * np.linalg.norm(mat_est - y_mat) ** 2)  # we should have recovered the low-rank matrix
        print(params[-1])
        pass
    # else:
    #     assert (np.linalg.norm(mat_est - y_mat) < 1e-3)


if __name__ == "__main__":
    # n_data = 6
    # n_features = 5
    # mat = tf.constant(np.outer(range(n_data), np.ones(n_features)), dtype='int64')
    # print(tf_eval(mat))
    #
    test_larcqy_logistic(True)
