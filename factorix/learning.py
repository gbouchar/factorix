import numpy as np
import warnings

import tensorflow as tf

from tfrnn.hooks import LoadModelHook, SpeedHook, SaveModelHook, LossHook, AccuracyHook

from factorix.dataset_reader import mat2tuples
from factorix.toy_examples import toy_factorization_problem, svd_factorize_matrix
from factorix.samplers import tuple_sampler, feed_dict_sampler
from factorix.losses import loss_func
from factorix.scoring import multilinear_tuple_scorer


def learn(loss_op, sampler, optimizer, hooks=(), max_epochs=500, variables=None):
    """

    Args:
        loss_op: TensorFlow operator that computes the loss to minimize
        sampler: sampler that generate dictionary inputs
        optimizer: TensorFlow optimization object
        hooks: functions that are called during training
        max_epochs: maximal number of epochs through the data

    Returns:

    Example:

    >>> it, (x, y) = feed_dict_sampler([([[1.0, 2]], [2.0]), ([[4, 5]], [6.5]), ([[7, 8]], [11])])
    >>> w = tf.Variable(np.zeros((2, 1), dtype=np.float32))
    >>> optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    >>> loss_op = tf.nn.l2_loss(tf.matmul(x, w) - y)
    >>> w_opt = learn(loss_op, it, optimizer, hooks=[simple_hook], max_epochs=500, variables=[w])
         0) Epoch    0: loss=2.0
       100) Epoch   33: loss=0.00151753
       200) Epoch   66: loss=0.001789
       300) Epoch  100: loss=0.00755719
       400) Epoch  133: loss=0.000480593
       500) Epoch  166: loss=0.00038268
       600) Epoch  200: loss=0.00117665
       700) Epoch  233: loss=6.05594e-05
       800) Epoch  266: loss=3.57539e-05
       900) Epoch  300: loss=8.13957e-05
      1000) Epoch  333: loss=3.36935e-06
      1100) Epoch  366: loss=1.4538e-06
      1200) Epoch  400: loss=2.3936e-06
      1300) Epoch  433: loss=7.79419e-08
      1400) Epoch  466: loss=2.36396e-08
    >>> w_opt
    [array([[ 1. ],
           [ 0.5]], dtype=float32)]

    # demo of a rectangular matrix factorization with square loss:
    # # >>> y_mat = toy_factorization_problem(n=7, m=6, rk=4, noise=1)
    # # >>> emb0 = svd_factorize_matrix(y_mat, rank=4)  # exact svd solution
    # # >>> sampler = tuple_sampler(mat2tuples(y_mat), minibatch_size=42)
    # # >>> params = learn(dot_product, sampler, max_epochs=500)
    # # >>> x_mat_est2 = np.dot(u2[:7], u2[7:].T)  # the initial matrix
    # # >>> np.linalg.norm(emb0[:7].dot(emb0[7:].T)-x_mat_est2) < 1e-3
    # True
    """
    if variables is None:
        variables = tf.trainable_variables()
    minimization_op = optimizer.minimize(loss_op)
    # Launch the graph.
    with tf.Session() as session:  # we close the session at the end of the training
        session.run(tf.initialize_all_variables())
        epoch = 0
        iteration = 0
        while epoch < max_epochs:
            for feed_dict in sampler:
                _, current_loss = session.run([minimization_op, loss_op], feed_dict=feed_dict)
                for hook in hooks:
                    hook(session, epoch, iteration, loss_op, current_loss)
                iteration += 1
            for hook in hooks:  # calling post-epoch hooks
                hook(session, epoch, None, loss_op, 0)
            epoch += 1
        final_params = session.run(variables)
    return final_params


def simple_hook(session, epoch, iteration, loss_op, current_loss):
    if iteration is not None and ((iteration % 100) == 0):
        print("%6d) Epoch %4d: loss=%s" % (iteration, epoch, str(current_loss)))


if __name__ == '__main__':
    y_mat = toy_factorization_problem(n=7, m=6, rk=4, noise=1)
    rank = 2
    batch_size = 42
    sampler, (x, y) = feed_dict_sampler(tuple_sampler(mat2tuples(y_mat), minibatch_size=batch_size))
    loss_op = tf.reduce_mean(loss_func(multilinear_tuple_scorer(x, rank=rank), y, 'quadratic'))
    hooks = [simple_hook]
    params = learn(loss_op, sampler, tf.train.AdamOptimizer(learning_rate=0.1), hooks, max_epochs=500)

