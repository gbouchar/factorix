
import tensorflow as tf
from naga.shared.tf_addons import tf_eval

a = tf.Variable([[1, 2], [3, 4], [5, 6]])
b = tf.constant(3)
print(tf_eval(tf.minimum(a, 2)))