from __future__ import print_function
import tensorflow as tf
import numpy as np
import networkx as nx
import time
import sys

from deeplp.models.deeplp import DeepLP

class DeepLP_ATT(DeepLP):
    """
    Base class for Deep Label Propagation (DeepLP).
    Inherit this class to implement different link functions.
    This class by itself will not run to give a meaningful result,
    as variables might not exist.
    """

    def __init__(self, log_name,
                       edge_features,
                       graph,
                       num_classes,
                       unlabeled_indices,
                       has_features=False,
                       loss_type='log',
                       lr=0.01,
                       acc_class_mass=False,
                       loss_class_mass=False,
                       num_iter=100,
                       optimizer='adam',
                       regularize=0,
                       regularize_type='l2',
                       seed=1):
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
        def convert_sparse_matrix_to_sparse_tensor(X):
            coo = X.tocoo()
            indices = np.mat([coo.row, coo.col]).transpose()
            return tf.SparseTensor(indices, coo.data, coo.shape)

        self.weights = convert_sparse_matrix_to_sparse_tensor(graph)
        self.model = 'att'
        self.seed = seed
        self.log_name = log_name
        self.unlabeled_indices = unlabeled_indices
        self.edge_features = tf.constant(edge_features,tf.float32)
        # initialize instance variables
        self.has_features=has_features
        self.loss_type   = loss_type
        self.lr          = lr
        self.num_classes = num_classes
        if self.has_features:
            self.num_features= num_classes + edge_features.shape[1]
        else:
            self.num_features= num_classes
        self.num_iter    = num_iter
        self.num_nodes   = graph.shape[0]
        self.regularize  = regularize
        print("------hey--------")
        print(self.regularize)
        self.regularize_type = regularize_type
        self.optimizer   = optimizer

        self.acc_class_mass=acc_class_mass
        self.loss_class_mass=loss_class_mass

        self._init_weights(graph, seed)

        # initialize placeholders
        shape             = [self.num_classes, None, self.num_nodes]
        self.X            = tf.placeholder("float", shape=shape)
        self.y            = tf.placeholder("float", shape=shape)
        self.labeled      = tf.placeholder("float", shape=shape)
        self.true_labeled = tf.placeholder("float", shape=shape)
        self.masked       = tf.placeholder("float", shape=shape)
        self.yhat         = self._forwardprop(self.X,
                                              self.labeled,
                                              self.num_iter,
                                              self.edge_features)
        self.update, self.metrics = self._backwardprop(self.y,
                                                         self.yhat,
                                                         self.labeled,
                                                         self.true_labeled,
                                                         self.masked,
                                                         self.regularize,
                                                         self.lr)

    def _save(self,epoch,train_data,validation_data):
        """Evaluate important parameters and save some in a list
        Arguments:
            epoch: current epoch
            train_data: train data with leave-k-out
            validation_data: validation data without leave-k-out
        Returns:
            None
        """
        # metrics related to train data
        unlabeled_loss, leave_k_out_loss, regularize_val, objective, accuracy, thetas, b \
        = self._eval([self.metrics['unlabeled_loss'],
                      self.metrics['leave_k_out_loss'],
                      self.metrics['regularize_val'],
                      self.metrics['objective'],
                      self.metrics['accuracy'],
                      self.thetas,
                      self.b],
                      train_data)
        # normalize regularization value by regularization parameter
        regularize_val = regularize_val / self.regularize if self.regularize else 0
        # metrics related to validation data
        # self.yhat_np, true_loss, true_accuracy \
        # = self._eval([self.yhat,
        #               self.metrics['true_loss'],
        #               self.metrics['true_accuracy']],
        #               validation_data)

        weight_vals = self._eval(self.values,train_data)
        row_entropy_vals = self._eval(self.row_entropies,train_data)
        col_entropy_vals = self._eval(self.col_entropies,train_data)
        divergence_vals = self._eval(self.divergences,train_data)
        h_vals = self._eval(self.h,train_data)
        Wnorm_vals = self._eval(self.Wnorms,train_data)

        print('weight_vals yeah')

        np.savetxt('indices.csv',weight_vals[0].indices,delimiter=',')
        res = []
        for el in weight_vals:
            res.append(list(el.values))
        res = np.array(res)
        np.savetxt('weights'+str(epoch)+'.csv',res,delimiter=',')
        res = []
        for el in row_entropy_vals:
            res.append(list(el))
        res = np.array(res)
        np.savetxt('row_entropy_vals'+str(epoch)+'.csv',res,delimiter=',')

        res = []
        for el in Wnorm_vals:
            res.append(list(el.values))
        res = np.array(res)
        np.savetxt('wnorm'+str(epoch)+'.csv',res,delimiter=',')

        res = []
        for el in col_entropy_vals:
            res.append(list(el))
        res = np.array(res)
        np.savetxt('col_entropy_vals'+str(epoch)+'.csv',res,delimiter=',')
        res = []
        for el in divergence_vals:
            res.append(list(el))
        res = np.array(res)
        np.savetxt('divergence_vals'+str(epoch)+'.csv',res,delimiter=',')
        res = []
        for iter,el in enumerate(h_vals):
            res.append(list(el))
            np.savetxt('h_vals'+str(epoch)+'_' + str(iter) + '.csv',el,delimiter=',')
        # res = np.array(res)

        # for item in weight_vals:
        #     fd.write("%s\n" % item)
        # fd.write(weight_vals)
        # fd.close()

        # save important parameters
        self.unlabeled_losses.append(unlabeled_loss)
        self.leave_k_out_losses.append(leave_k_out_loss)
        self.accuracies.append(accuracy)
        # self.true_losses.append(true_loss)
        # self.true_accuracies.append(true_accuracy)
        self.objectives.append(objective)

        # print parameters every epoch
        print("== ",
              "epoch:",epoch,
              "labeled loss:",leave_k_out_loss,
              "regularize_val",regularize_val,
              "regularize_param",self.regularize,
              "objective",objective,
              "unlabeled loss:",unlabeled_loss,
              "accuracy:",accuracy,
              # "true unlabeled loss:",true_loss,
              # "true accuracy:",true_accuracy,
              "thetas:", thetas,
              "b:", b
              )

    def _forwardprop(self, X,
                           labeled,
                           num_iter,
                           edge_features):
        """Forward prop which mimicks LP
        Arguments:
            X: input labels where some labeled nodes are masked
            weights: graph adjacency weight matrix
            labeled: boolean indicating whether the labels are labeled
            num_iter: number of hidden layers
        Returns:
            accuracy
        """

        self.values = []
        self.row_entropies = []
        self.col_entropies = []
        self.divergences = []
        self.h = []
        self.Wnorms = []
        # normalize weights
        start_time = time.time()
        h = X
        for i in range(num_iter):
            h = self._propagate(h)
            h = h * (1-labeled) + X * labeled
            print(h,tf.gather(h,0,axis=1))
            self.h.append(tf.gather(h,0,axis=1))
        yhat = h
        # def condition(j,h):
        #     return j < num_iter
        #
        # def loop(j,h):
        #     h = self._propagate(h)
        #     h = h * (1-labeled) + X * labeled
        #
        #     return j+1,h
        #
        # _,yhat = tf.while_loop(condition, loop, loop_vars=[0,X])

        return yhat

    def _init_weights(self, graph, seed):
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

        self.row_indices,self.col_indices = graph.tocoo().row,graph.tocoo().col
        self.edges = list(zip(graph.tocoo().row,graph.tocoo().col))
        self.num_edges = len(self.edges)

        self.thetas = tf.Variable(tf.constant([0,0,0],dtype=tf.float32))
        self.b = tf.constant(0.0,dtype=tf.float32)

    def _propagate(self,h):

        def kl_divergence(p,q):
            return p * tf.log(p/(q+0.00001)+0.00001)

        entropies = tf.reduce_sum(-1 * h * tf.log(h+0.00001),axis=0)
        row_entropies = tf.gather(entropies,self.row_indices,axis=1)
        col_entropies = tf.gather(entropies,self.col_indices,axis=1)

        row_features = tf.gather(h,self.row_indices,axis=2)
        col_features = tf.gather(h,self.col_indices,axis=2)

        kl_divergences = tf.reduce_sum(kl_divergence(row_features,col_features),axis=0)

        new_hs = tf.ones([1,self.num_classes,self.num_nodes], tf.float32) # base array for concatenating new results
        i = tf.constant(0)
        num_samples = tf.shape(h)[1]
        W_edges = self.edges


        for sample_id in range(25):
            sub_row_entropies = (np.log(self.num_classes) - tf.gather(row_entropies,sample_id,axis=0))/np.log(self.num_classes)
            sub_col_entropies = (np.log(self.num_classes) - tf.gather(col_entropies,sample_id,axis=0))/np.log(self.num_classes)
            sub_divergence = (np.log(1/0.00001+0.00001) - tf.gather(kl_divergences,sample_id,axis=0))/np.log(1/0.00001+0.00001)
            W_vals = self.thetas[0] * sub_row_entropies \
                   + self.thetas[1] * sub_col_entropies \
                   + self.thetas[2] * sub_divergence \
                   + self.b
            W_shape = [self.num_nodes,self.num_nodes]
            W = tf.SparseTensor(W_edges, W_vals, W_shape)

            # Wnorm = tf.sparse_softmax(W)
            Wnorm = tf.sparse_softmax(W)
            # Wnorm = tf.SparseTensor(W_edges, tf.where(tf.is_nan(Wnorm.values), tf.zeros_like(Wnorm.values), Wnorm.values), W_shape)
            new_h = tf.transpose(tf.sparse_tensor_dense_matmul(Wnorm, tf.transpose(tf.gather(h,sample_id, axis=1))))
            new_hs = tf.concat([new_hs, [new_h]], 0)
            if sample_id == 0:
                self.row_entropies.append(sub_row_entropies)
                self.col_entropies.append(sub_col_entropies)
                self.divergences.append(sub_divergence)
                self.values.append(W)
                self.Wnorms.append(Wnorm)



        #
        # def condition(i,m):
        #     return i < num_samples
        #
        # def loop(sample_id,new_hs):
        #     W_vals = self.thetas[0] * tf.gather(row_entropies,sample_id,axis=0) + self.thetas[1] * tf.gather(col_entropies,sample_id,axis=0) + self.thetas[2] * tf.gather(kl_divergences,sample_id,axis=0) + self.b
        #     W_shape = [self.num_nodes,self.num_nodes]
        #     W = tf.SparseTensor(W_edges, W_vals, W_shape)
        #     # self.values.append(W)
        #     print("wvalues...............")
        #     print(W_vals)
        #     Wnorm = tf.sparse_softmax(W)
        #     print(Wnorm,tf.transpose(tf.gather(h,sample_id, axis=1)))
        #     Wnorm = tf.SparseTensor(W_edges, tf.where(tf.is_nan(Wnorm.values), tf.zeros_like(Wnorm.values), Wnorm.values), W_shape)
        #     new_h = tf.transpose(tf.sparse_tensor_dense_matmul(Wnorm, tf.transpose(tf.gather(h,sample_id, axis=1))))
        #     new_hs = tf.concat([new_hs, [new_h]], 0)
        #     return [sample_id+1,new_hs]
        #
        # _,new_hs = tf.while_loop(condition, loop, loop_vars=[i,new_hs],shape_invariants=[i.get_shape(),
        #                                                    tf.TensorShape([None,self.num_classes,self.num_nodes])])
        new_hs = tf.gather(new_hs,tf.range(1,num_samples+1), axis=0)
        new_hs = tf.transpose(new_hs,perm=[1, 0, 2])
        return new_hs

    def _regularize_loss(self,regularize):

        regularize = tf.constant(regularize,dtype=tf.float32)

        if self.regularize_type == 'l1':
            regularizer = tf.contrib.layers.l1_regularizer(scale=regularize)
        elif self.regularize_type == 'l2':
            regularizer = tf.contrib.layers.l2_regularizer(scale=regularize)

        to_regularize = [self.thetas]
        regularization_penalty = tf.contrib.layers.apply_regularization(regularizer, to_regularize)


        return regularization_penalty
