from __future__ import print_function

import networkx as nx
import numpy as np
import tensorflow as tf

from deeplp.models.deeplp import DeepLP

class DeepLP_RBF(DeepLP):
    """
    Deep LP with scalor divisor as a parameter.
    See our paper for details.
    """
    def _save_params(self,epoch,data):
        """Save and print parameters specific to each link function
        Save and print sigma.
        Arguments:
            epoch: current epoch
            data: data for evaluating the parameters specific to the link function
        Returns:
            None
        """
        sigma_np = self._eval(self.sigma)
        self.sigmas.append(sigma_np)
        print("sigma:",sigma_np)
        super()._save_params(epoch,data)

    def _init_weights(self, features, graph, seed):
        """Initialize weights
        Create a sparse weight tensor from graph and features.
        Sigma (hyperparameter in typical RBF) is the tf variable.
        Arguments:
            features: features for ecah node
            graph: adjacency matrix of the nodes
            seed: seed to be used for random initialization of parameters
        Returns:
            weights in tf variable
        """
        np.random.seed(seed)

        row_features, col_features, edges = self._get_row_col_features(graph,features)

        if self.sigma == None:
            self.sigma = 10*features.shape[1]
        self.sigma = tf.Variable(self.sigma, dtype=tf.float32)
        # values and indices for sparse tensor initialization
        values = tf.exp(-1 * tf.divide(tf.reduce_sum((row_features - col_features)**2,axis=1),self.sigma**2))
        indices = tf.constant(edges, dtype=tf.int64)
        weights = tf.SparseTensor(indices, values, graph.shape)
        return weights

    def _get_row_col_features(self,graph,features):
        # get node features for each edge
        half_edges = list(nx.to_edgelist(nx.from_numpy_matrix(graph)))
        half_edges = np.array(half_edges)[:,:2].astype(int)
        edges = []
        for edge in half_edges:
            edges.append(edge)
            edges.append(np.array([edge[1],edge[0]]))
        edges = np.array(edges)
        rows,cols = edges[:,0],edges[:,1]
        # declare tensors for features
        rows_tf = tf.constant(np.take(features,rows,axis=0), dtype=tf.float32)
        cols_tf = tf.constant(np.take(features,cols,axis=0), dtype=tf.float32)
        return rows_tf, cols_tf, edges

    def train(self,train_data,validation_data,epochs):
        """See DeepLP for details"""
        self.sigmas = []
        super().train(train_data,validation_data,epochs)
