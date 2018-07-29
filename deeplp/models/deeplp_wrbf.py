from __future__ import print_function
from scipy import sparse
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import networkx as nx

from deeplp.models.deeplp import DeepLP


class DeepLP_WRBF(DeepLP):
    """
    Deep label propagation with weights for RBF kernel as parameters.
    """

    def _save_params(self,epoch,data):
        theta_np = self._eval(self.theta)
        theta_np = theta_np[0]
        self.thetas.append(theta_np)
        print("== theta:",theta_np[:10].T)
        super()._save_params(epoch,data)

    def train(self,data,full_data,epochs):
        self.thetas = []
        super().train(data,full_data,epochs)

    def _init_weights(self, features, weights, seed):
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
        np.random.seed(seed)
        if seed==0:
            theta = np.ones((1,features.shape[1]))
        else:
            theta = np.random.uniform(0,2,(1,features.shape[1]))
        self.theta   = tf.Variable(theta, dtype=tf.float32)

        row_features = tf.constant(features[weights.tocoo().row].toarray(), dtype=tf.float32)
        col_features = tf.constant(features[weights.tocoo().col].toarray(), dtype=tf.float32)

        rbf_values = tf.exp(tf.divide(-1 * tf.reduce_sum((self.theta * (row_features - col_features)) ** 2,axis=1),self.sigma))
        rbf_indices = tf.constant(list(zip(weights.tocoo().row,weights.tocoo().col)), dtype=tf.int64)
        num_nodes = features.shape[0]
        self.weights_unnormalized = tf.SparseTensor(rbf_indices, rbf_values, [num_nodes, num_nodes])
        print(self.weights_unnormalized)
        self.weights = self._normalize_weights(self.weights_unnormalized)
        return self.weights


    def _regularize_loss(self,regularize_theta,regularize_weight):
        print('=====================================')
        print(regularize_weight,regularize_theta)
        # regularize_thetas = tf.constant(regularize,dtype=tf.float32)
        regularizer_theta = tf.contrib.layers.l2_regularizer(scale=regularize_theta)
        theta_penalty = tf.contrib.layers.apply_regularization(regularizer_theta, [self.theta])

        if self.sparse_edges:
            print("sparse edges")
            # regularize_weights = 100 * self.epoch ** (1/2)
            l1_regularizer = tf.contrib.layers.l1_regularizer(scale=regularize_weight)
            weight_penalty = tf.contrib.layers.apply_regularization(regularizer=l1_regularizer, weights_list=[self.weights_unnormalized.values])
            return theta_penalty + weight_penalty
        else:
            return theta_penalty
