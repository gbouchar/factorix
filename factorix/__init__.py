import tensorflow as tf
# from naga.factorix.scoring import sparse_multilinear_dot_product


def tf_eval(expr):
    """
    TensorFlow evaluation of an expression
    :param expr: TensorFlow expression without placeholder
    :return: result of the expression
    """
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        res = sess.run(expr)
    return res

__all__ = [name for name, x in locals().items() if not name.startswith('_')]

# simple functions often useful

