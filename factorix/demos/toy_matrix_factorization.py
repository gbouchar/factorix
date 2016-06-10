import numpy as np
import tensorflow as tf

from naga.shared.learning import data_to_batches, feed_dict_sampler, learn


n, m, rank = 7, 6, 3
mat = np.random.randn(n, rank).dot(np.random.randn(rank, m))
tuples = [([i, n + j], mat[i, j]) for i in range(n) for j in range(m)]
tuple_iterable = data_to_batches(tuples, minibatch_size=n * m)
sampler, (x, y) = feed_dict_sampler(tuple_iterable, types=[np.int64, np.float32])
emb_var = tf.Variable(tf.cast(np.random.randn(n + m, rank), 'float32'))
loss_op = tf.reduce_mean(tf.square(tf.reduce_sum(tf.reduce_prod(tf.gather(emb_var, x), 1), 1) - y))
emb, = learn(loss_op, sampler,  max_epochs=200)
mat_est = emb[:n, :].dot(emb[n:, :].T)
print(np.linalg.norm(mat_est - mat))   # we should have recovered the low-rank matrix
