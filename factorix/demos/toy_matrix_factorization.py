import sys
import numpy as np

import tensorflow as tf

from factorix.utils.learning import data_to_batches, feed_dict_sampler, learn


n, m, rank, std_noise = 7, 6, 3, 0.1  # number of rows, columns and rank of the simulated matrix
U_true = np.random.randn(n, rank)  # create a random low-rank matrix
V_true = np.random.randn(m, rank)  # create a random low-rank matrix
embeddings_true = np.vstack((U_true, V_true))
mat = U_true.dot(V_true.T)
mat += np.random.randn(n, m) * std_noise


U0, spectrum0, V0 = np.linalg.svd(mat)

U0 = U0[:, :rank]  # select the singular vectors associated with the highest singular values
V0 = V0[:rank, :].T  # select the singular vectors associated with the highest singular values
spectrum0 = spectrum0[:rank]
print(U0)
print(V0)
print(np.diag(spectrum0))
recovered_mat = U0.dot(np.diag(spectrum0)).dot(V0.T)

# print(mat)
# print(recovered_mat)
# sys.exit()
missing_values = True

if missing_values:
    mat0 = np.nan * np.ones((n, m))
    observed_indices = [(np.random.randint(n), np.random.randint(m)) for _ in range(int(n * m))]
    print(observed_indices)
    for i, j in observed_indices:
        mat0[i,j] = mat[i, j]
    mat = mat0
    print(mat)
    tuples = [([i, n + j], mat[i, j]) for (i, j) in observed_indices]
else:
    tuples = [([i, n + j], mat[i, j]) for i in range(n) for j in range(m)]


tuple_iterable = data_to_batches(tuples, minibatch_size=n * m)

batches = [batch for batch in tuple_iterable]


print(batches[0])

sampler, (x, y) = feed_dict_sampler(tuple_iterable, types=[np.int64, np.float32])
# sampler is an iterator return (bucket_id, dictionary containing the batch)
# x and y are "placeholder" (equivalent of symbolic variables used as input to your model)

# print([s for s in sampler])

print(x)

emb_var = tf.Variable(tf.cast(np.random.randn(n + m, rank), 'float32', name='embeddings'))
# emb_var = tf.Variable(tf.cast(embeddings_true, 'float32', name='embeddings'))



# computation graph

selected_embeddings = tf.gather(emb_var, x)
scores = tf.reduce_sum(tf.reduce_prod(selected_embeddings, 1), 1)  # multilinear dot product
losses = tf.square(scores - y)
loss_op = tf.reduce_mean(losses)

# operators

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
update = optimizer.minimize(loss_op)
initializer = tf.initialize_all_variables()

# embeddings, = learn(loss_op, sampler,  max_epochs=200)

tf.get_default_graph().finalize()  # not mandatory but highly recommended

with tf.Session() as session:
    session.run(initializer)
    # ask for food (data)
    feed_dict = [sample for sample in sampler][0][1]
    print(feed_dict)
    embeddings0, loss_value0 = session.run([emb_var, loss_op], feed_dict)
    print(loss_value0)
    print('difference between the embeddings', np.sum(np.square(embeddings0 - embeddings_true)))
    # print(embeddings)
    # print('initial value of the loss', loss_value0)

    for iteration in range(1000):
        loss_value, _ = session.run([loss_op, update], feed_dict)
        print(iteration, loss_value)

    embeddings_final = session.run(emb_var)

U = embeddings_final[:n, :]  # row embeddings
V = embeddings_final[n:, :]  # column embeddings

mat_est = U.dot(V.T) # estimated matrix
print('original matrix', mat)
print('SVD recovery', recovered_mat)
print('estimated matrix', mat_est)
sys.exit()

print(np.linalg.norm(mat_est - mat))   # we should have recovered the low-rank matrix
