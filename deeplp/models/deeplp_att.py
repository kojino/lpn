from __future__ import print_function
import tensorflow as tf
import numpy as np
import networkx as nx
import time
import sys
import scipy as sp

from deeplp.models.deeplp import DeepLP

class DeepLP_ATT(DeepLP):
    """
    Base class for Deep Label Propagation (DeepLP).
    Inherit this class to implement different link functions.
    This class by itself will not run to give a meaningful result,
    as variables might not exist.
    """

    def __init__(self, # ATT specific
                       has_features,
                       change_b,
                       node_features,
                       # common inputs
                       profile,
                       model,
                       log_name,
                       features,
                       graph,
                       num_classes,
                       unlabeled_indices,
                       confidence=False,
                       clamp=False,
                       loss_class_mass=False,
                       lr=0.01,
                       num_iter=100,
                       regularize=0,
                       regularize_theta=0,
                       regularize_weight=0,
                       seed=0,
                       sparse_edges=0,
                       clamp_scale=0.1):
        """Create weight matrix (depends on link functions) and build graph.
        Arguments:
            num_classes: number of label classes
            weights: numpy weight matrix
            clamp: if True, apply softmax to the hidden output each layer
            loss_type: type of loss function, 'mse' or 'log'
            lr: learning rate of the optimizer
            param: parameters used for initialization if seed randomization
                   is not prefered
            num_iter: number of hidden layers in the graph
            regularize: regularization parameter, if 0, no regularization
            seed: seed to be used for random initialization of parameters
        """

        self.change_b = change_b
        self.has_features = has_features
        self.node_features = node_features

        super().__init__(
            profile, model, log_name, features, graph, num_classes, unlabeled_indices, \
            confidence, clamp, loss_class_mass, lr, num_iter, regularize_theta, regularize_weight, \
            seed, sparse_edges, clamp_scale)

    def _init_weights(self, edge_features, graph, seed):
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
        tf.set_random_seed(seed)
        np.random.seed(seed)

        self.row_indices,self.col_indices = graph.tocoo().row,graph.tocoo().col
        self.edges = list(zip(graph.tocoo().row,graph.tocoo().col))
        self.num_edges = len(self.edges)

        if self.change_b:
            print("self loop")
            # if seed == 0:
            self.self_theta = [tf.Variable(tf.constant(np.zeros((6,1)),dtype=tf.float32)), # node features
                               tf.Variable(tf.constant(0.0,dtype=tf.float32)), # t
                               tf.Variable(tf.constant(0.0,dtype=tf.float32)), # (t-T/2)^2
                               tf.Variable(tf.constant(0.0,dtype=tf.float32)), # entropy
                               tf.Variable(tf.constant(0.0,dtype=tf.float32)), # avg neigh
                               tf.Variable(tf.constant(0.0,dtype=tf.float32))  # max neigh
                               ]
            #
            # else:
            #     self.self_theta = [tf.Variable(tf.constant(np.random.normal(loc=0,scale=1,size=(6,1)),dtype=tf.float32)), # node features
            #                        tf.Variable(tf.constant(np.random.normal(loc=0,scale=1),dtype=tf.float32)), # t
            #                        tf.Variable(tf.constant(np.random.normal(loc=0,scale=1),dtype=tf.float32)), # (t-T/2)^2
            #                        tf.Variable(tf.constant(np.random.normal(loc=0,scale=1),dtype=tf.float32)), # entropy
            #                        tf.Variable(tf.constant(np.random.normal(loc=0,scale=1),dtype=tf.float32)), # avg neigh
            #                        tf.Variable(tf.constant(np.random.normal(loc=0,scale=1),dtype=tf.float32))  # max neigh
            #                        ]
            # self.self_theta = [tf.constant(0.0,dtype=tf.float32)]
            # self.self_weight = tf.sigmoid(tf.Variable(tf.constant(0.0)))
        else:
            self.self_theta = [tf.constant(0.0,dtype=tf.float32)]
            print("no self loop")
            self.self_weight = tf.constant(0.0,dtype=tf.float32)

        if seed == 0: # zero theta, zero edge
            thetas_np = np.zeros((2,self.num_classes))
            edge_weights_np = np.zeros([1,self.num_features])
        else: # random theta, random edge
            thetas_np = np.random.normal(loc=0,scale=1,size=(2,self.num_classes))
            edge_weights_np = np.random.uniform(low=-1.0,high=1.0,size=(1,self.num_features))

        self.thetas = tf.Variable(tf.constant(thetas_np,dtype=tf.float32))
        self.theta_times = tf.Variable(tf.constant(np.zeros((3,1)),dtype=tf.float32))
        if self.has_features:
            # self.edge_weights = tf.Variable(tf.constant(edge_weights_np,dtype=tf.float32))
            self.edge_weights = tf.Variable(tf.constant(edge_weights_np,dtype=tf.float32))
        else:
            self.edge_weights = tf.constant(edge_weights_np,dtype=tf.float32)

    def _propagate(self,curr_iter,h):
        eps = 0.000001

        def kl_divergence(p,q):
            return p * tf.log(p/(q+eps)+eps)
        entropies_mat = tf.reduce_sum(tf.expand_dims(tf.expand_dims(self.thetas[0],axis=0),axis=0) * (-1) * h * tf.log(h+eps),axis=2)
        # row_entropies = (np.log(self.num_classes) - tf.gather(entropies_mat,self.row_indices,axis=1))/np.log(self.num_classes)
        col_entropies = (np.log(self.num_classes) - tf.gather(entropies_mat,self.col_indices,axis=0))/np.log(self.num_classes)

        row_features = tf.gather(h,self.row_indices,axis=0)
        col_features = tf.gather(h,self.col_indices,axis=0)
        kl_divergences = tf.reduce_sum(tf.expand_dims(tf.expand_dims(self.thetas[1],axis=0),axis=0) * kl_divergence(col_features,row_features),axis=2)/np.log(1/eps+eps)

        if self.has_features:
            edge_features = tf.reduce_sum(self.edge_features * self.edge_weights,axis=1)

        new_hs = tf.ones([1,self.num_nodes,self.num_classes], tf.float32) # base array for concatenating new results
        i = tf.constant(0)
        num_samples = tf.shape(h)[1]
        W_edges = self.edges
        #
        # times = (self.theta_times[0] * curr_iter) / self.num_iter + \
        #         (self.theta_times[1] * curr_iter**2.0) / self.num_iter ** 2 + \
        #         self.theta_times[2] * tf.log(curr_iter+1.0)

        def condition(i,m):
            return i < num_samples

        def loop(sample_id,new_hs):

            W_vals =  tf.gather(col_entropies,sample_id,axis=1) + tf.gather(kl_divergences,sample_id,axis=1)
            if self.has_features:
                W_vals =  W_vals + edge_features
            W_shape = [self.num_nodes,self.num_nodes]
            W = tf.SparseTensor(W_edges, W_vals, W_shape)
            Wnorm = tf.sparse_softmax(W)

            new_h = tf.sparse_tensor_dense_matmul(Wnorm, tf.gather(h,sample_id, axis=1))
            new_hs = tf.concat([new_hs, [new_h]], 0)
            return [sample_id+1,new_hs]

        _,new_hs = tf.while_loop(condition, loop, loop_vars=[i,new_hs],shape_invariants=[i.get_shape(),
                                                           tf.TensorShape([None,self.num_nodes,self.num_classes])])
        new_hs = tf.gather(new_hs,tf.range(1,num_samples+1), axis=0)
        new_hs = tf.transpose(new_hs,perm=[1, 0, 2])

        #
        if self.change_b:
            col_entropies_mat = col_entropies
            norm_graph = self.graph / tf.sparse_reduce_sum(self.graph,axis=1,keep_dims=True)
            avg_entropy = tf.sparse_tensor_dense_matmul(norm_graph,tf.transpose(col_entropies_mat))
            avg_entropy = tf.expand_dims(tf.transpose(avg_entropy),axis=0)


            t = tf.to_float(curr_iter)
            T = tf.constant(self.num_iter,dtype=tf.float32)
            # self_node_features = tf.transpose(tf.expand_dims(tf.expand_dims(tf.reduce_sum(self.self_theta[0] * self.node_features,axis=0),axis=1),axis=2),perm=[1,2,0])
            self_time = self.self_theta[1] * t/T + self.self_theta[2] * (t-T/2)**2 / T ** 2
            self_entropy = tf.expand_dims(col_entropies_mat,axis=0)
            self_entropy = self.self_theta[3] * (np.log(self.num_classes) - self_entropy)/np.log(self.num_classes)
            self_avg_neigh = avg_entropy
            self_avg_neigh = self.self_theta[4] * (np.log(self.num_classes) - self_avg_neigh)/np.log(self.num_classes)
            # self_min_neigh = tf.expand_dims(self.self_theta[5] * min_neigh_entropies, axis=0)
            # self.self_weight =  tf.sigmoid(self_node_features + self_time + self_entropy)
            print(self_time,self_entropy,self_avg_neigh)
            self.self_weight = tf.sigmoid(self_time + self_entropy + self_avg_neigh)
            print('heya!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print(self.self_weight)
        return new_hs

    def _regularize_loss(self,regularize_theta,regularize_weight):

        regularize = tf.constant(regularize_theta,dtype=tf.float32)

        regularizer = tf.contrib.layers.l2_regularizer(scale=regularize)

        to_regularize = [self.thetas]
        if self.has_features:
            to_regularize.append(self.edge_weights)
        regularization_penalty = tf.contrib.layers.apply_regularization(regularizer, to_regularize)

        return regularization_penalty
