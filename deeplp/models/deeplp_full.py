import numpy as np
import tensorflow as tf

from deeplp.models.deeplp import DeepLP
from deeplp.utils import sparse_tensor_dense_tensordot


class DeepLP_Full(DeepLP):
    def _propagate(self, j, h):
        """
        Propagate labels similarly to standard LP.
        """
        return sparse_tensor_dense_tensordot(self.weights, h, axes=1)

    def _summary(self):
        super()._summary()
        tf.summary.histogram("weights", self.weights.values)

    def _init_weights(self):
        indices = np.vstack((self.graph.tocoo().row, self.graph.tocoo().col)).T

        values = tf.get_variable('weights', shape=self.graph.tocoo().row.shape)

        weights_unnormalized = tf.SparseTensor(
            indices, values, [self.num_nodes, self.num_nodes])
        weights = tf.sparse_softmax(weights_unnormalized)

        return weights

    def _regularize_loss(self):
        regularizer = tf.contrib.layers.l2_regularizer(scale=self.lamda)
        theta_penalty = tf.contrib.layers.apply_regularization(
            regularizer, [self.weights])
        return theta_penalty
