from __future__ import print_function
import tensorflow as tf
import numpy as np
import networkx as nx
# import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from model.DeepLP_WRBF import DeepLP_WRBF

class DeepLP_WInner(DeepLP_WRBF):
    '''
    Deep label propagation with weighted inner products.
    '''
    def __init__(self, num_nodes,
                       num_classes,
                       features,
                       graph,
                       theta,
                       num_iter=100,
                       loss_type='mse',          # 'mse' or 'log'
                       lr=0.1,
                       regularize=0,       # add L1 regularization to loss
                       regularize_type='l1',
                       regularize_m_1=True,
                       regularize_param=True, # if false, regularize the weights
                       clamp=False,
                       graph_sparse=False, # make the graph sparse
                       print_freq=10,      # print frequency when training
                       multi_class=False): # implementation for multiclass
        self.theta   = tf.Variable(theta, dtype=tf.float32)
        self.weights  = self._init_weights(features, graph, self.theta, num_nodes)
        self.regularize_type = regularize_type
        self.regularize_m_1 = regularize_m_1
        self.regularize_param = regularize_param
        self._build_graph(num_iter,
                          num_classes,
                          num_nodes,
                          loss_type,
                          lr,
                          regularize,
                          graph_sparse,
                          print_freq,
                          multi_class,
                          clamp)

    def train(self,data,full_data,epochs,grad_norm_threshold=0):
        self.thetas = []
        self.clamps = []
        super().train(data,full_data,epochs,grad_norm_threshold)

    def _init_weights(self, features, graph, theta, num_nodes):

        edges = np.array(list(nx.to_edgelist(nx.from_numpy_matrix(graph))))[:,:2].astype(int)
        new_edges = []
        for edge in edges:
            new_edges.append(edge)
            new_edges.append(np.array([edge[1],edge[0]]))
        new_edges = np.array(new_edges)
        rows,columns = new_edges[:,0],new_edges[:,1]

        row_features = tf.constant(np.take(features,rows,axis=0), dtype=tf.float32)
        col_features = tf.constant(np.take(features,columns,axis=0), dtype=tf.float32)
        print('yay')

        inner_products = tf.reduce_sum(((theta * row_features) * (theta * col_features)),axis=1)
        # inner_products = tf.reduce_sum(((theta * row_features) * (theta * col_features)),axis=1) / (tf.reduce_sum((theta * row_features)**2,axis=1) * tf.reduce_sum((theta * col_features)**2,axis=1)) ** (1/2)
        rbf_indices = tf.constant(new_edges, dtype=tf.int64)
        W = tf.SparseTensor(rbf_indices, inner_products, [num_nodes, num_nodes])
        return W

    def _tnorm(self, weights):
        Tnorm = tf.sparse_softmax(weights)

        return weights
