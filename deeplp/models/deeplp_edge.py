from __future__ import print_function
import tensorflow as tf
import numpy as np
import networkx as nx
import csv
from deeplp.models.deeplp_wrbf import DeepLP_WRBF
from sklearn.metrics.pairwise import paired_distances, pairwise_distances
from scipy.spatial.distance import braycurtis, canberra, chebyshev, cityblock, correlation, cosine, euclidean, jaccard, mahalanobis, minkowski


class DeepLP_Edge(DeepLP_WRBF):
    """
    Deep label propagation with distance metrics.
    """

    def _save_params(self, epoch, data):
        theta_np = self._eval(self.theta)
        b_np = self._eval(self.b)
        self.thetas.append(theta_np)
        self.bs.append(b_np)
        print("== theta_edge:", theta_np)
        print("== b:", b_np)
        super()._save_params(epoch, data)
        with open("thetas.csv", "a") as fp:
            wr = csv.writer(fp, dialect='excel')
            wr.writerow(self.thetas[0])

    def train(self, data, full_data, epochs):
        self.thetas = []
        self.bs = []
        super().train(data, full_data, epochs)

    def smooth_relu(self, X):
        return tf.exp(X) * tf.cast(tf.greater(-X,0),tf.float32) \
             + (X + 1) * tf.cast(tf.greater(X,0),tf.float32)

    def _init_weights(self, features, graph, seed):
        """Initialize weights
        Create a sparse weight tensor from graph and features.
        Theta (weights of RBF kernel) is the tf variable.
        Arguments:
            features: features for ecah node
            graph: adjacency matrix of the nodes
            seed: seed to be used for random initialization of parameters
        Returns:
            weights in tf variable
        """

        num_features = features.shape[1]
        print(num_features)
        num_edges = features.shape[0]
        num_nodes = graph.shape[0]
        features = tf.constant(features, dtype=tf.float32)

        if seed == 0:
            theta = np.zeros((1, num_features))
            # theta = np.array([])
        else:
            np.random.seed(seed)
            theta = np.random.normal(
                loc=0.0, scale=4.0, size=(1, num_features))

        self.theta = tf.Variable(theta, dtype=tf.float32)
        #
        # if self.weight_normalization == 'smooth_relu':
        #     self.b     = tf.Variable(1, dtype=tf.float32)
        #     self.values = tf.reduce_sum(features * self.theta + self.b,axis=1)
        #     indices = tf.constant(list(zip(graph.tocoo().row,graph.tocoo().col)), dtype=tf.int64)
        #     self.weights_unnormalized = tf.SparseTensor(indices, self.smooth_relu(self.values), [num_nodes, num_nodes])
        #     weights = self._normalize_weights(self.weights_unnormalized)

        self.b = tf.constant(1, dtype=tf.float32)
        self.values = tf.reduce_sum(features * self.theta, axis=1) + self.b
        indices = tf.constant(
            list(zip(graph.tocoo().row,
                     graph.tocoo().col)), dtype=tf.int64)

        self.nonzero_where = tf.squeeze(tf.where(tf.not_equal(self.values, 0)))
        self.nonzero_values = tf.gather(self.values, self.nonzero_where)
        nonzero_indices = tf.gather(indices, self.nonzero_where, axis=0)

        self.weights_unnormalized = tf.SparseTensor(
            nonzero_indices, self.nonzero_values, [num_nodes, num_nodes])
        weights = tf.sparse_softmax(self.weights_unnormalized)

        return weights
