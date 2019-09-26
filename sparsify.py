import numpy as np
import tensorflow as tf

def sparsify(dense):
    indices = np.array(dense[0].nonzero()).T
    values = [dense[0][tuple(index)] for index in indices]
    sparse = tf.sparse.SparseTensor(indices, values, dense[0].shape)
    return sparse

def densify(sparse):
    with tf.Session() as sess:
        s = sess.run(sparse)
        return tf.to_dense(s)

def print_tensor(t):
    with tf.Session() as sess:
        real_t = sess.run(t)
        print(real_t)

def get(tensor, index):
    with tf.Session() as sess:
        tensor_view = sess.run(tensor)
        return tensor_view[tuple(index)]
