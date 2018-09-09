import numpy as np
import tensorflow as tf

from deeplp.models.deeplp import DeepLP
from deeplp.utils import sparse_tensor_dense_tensordot


class DeepLP_Edge(DeepLP):
    """
    Feature based model for graph data with features.
    """

    def _propagate(self, j, h):
        """
        Propagate labels similarly to standard LP.
        """
        return sparse_tensor_dense_tensordot(self.weights, h, axes=1)

    def _summary(self):
        super()._summary()
        tf.summary.histogram("theta", self.theta)
        tf.summary.histogram("weights", self.weights.values)

    def _init_weights(self):
        self.theta = tf.Variable(
            np.zeros((1, self.num_features)), dtype=tf.float32)

        features_tf = tf.constant(self.features, dtype=tf.float32)
        values = tf.reduce_sum(features_tf * self.theta, axis=1) + 1
        nonzero_where = tf.squeeze(tf.where(tf.not_equal(values, 0)))
        nonzero_values = tf.gather(values, nonzero_where)
        indices_tf = tf.constant(self.indices, dtype=tf.int64)
        nonzero_indices = tf.gather(indices_tf, nonzero_where, axis=0)

        weights_unnormalized = tf.SparseTensor(
            nonzero_indices, nonzero_values, [self.num_nodes, self.num_nodes])
        weights = tf.sparse_softmax(weights_unnormalized)

        return weights

    def _regularize_loss(self):
        regularizer = tf.contrib.layers.l2_regularizer(scale=self.lamda)
        theta_penalty = tf.contrib.layers.apply_regularization(
            regularizer, [self.theta])
        return theta_penalty
