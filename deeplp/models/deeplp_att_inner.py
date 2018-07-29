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
                       loss_type='mse',
                       lr=0.01,
                       acc_class_mass=False,
                       loss_class_mass=False,
                       num_hidden=5,
                       num_iter=100,
                       optimizer='adam',
                       regularize=0,
                       regularize_type='l1',
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
        self.num_hidden  = num_hidden
        self.num_iter    = num_iter
        self.num_nodes   = graph.shape[0]
        self.regularize  = regularize
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

    def train(self,train_data,validation_data,epochs=1000):
        """Train the model and store the performance every epoch.
        Arguments:
            train_data: data to train the model, labeled nodes are masked
            validation_data: data to evaluate the performance,
                             labeleed nodes are unmasked
            epochs: number of epochs to train the model
        Returns:
            None
        """
        self._open_sess()
        start_time = time.time()
        # lists to store the performance metrics
        self.clamp_params = []
        self.unlabeled_losses = []
        self.leave_k_out_losses = []
        self.accuracies = []
        self.true_losses = []
        self.true_accuracies = []
        self.objectives = []
        self._save(-1,train_data,validation_data)
        self.y_np = self._eval(self.y, validation_data)
        num_samples = train_data['X'].shape[1]
        writer = tf.summary.FileWriter('log_att/'+self.log_name, self.sess.graph)
        summary = self._eval(self.summary_op,validation_data)
        writer.add_summary(summary, global_step=-1)


        for epoch in range(epochs):
            np.random.seed(None)
            mini_data_ids = np.random.choice([i for i in range(num_samples)],10,replace=False)
            # print(mini_data_ids)
            mini_train_data = {
                'X': train_data['X'][:,mini_data_ids,:],
                'y': train_data['y'],
                'labeled': train_data['labeled'][:,mini_data_ids,:],
                'true_labeled': train_data['true_labeled'],
                'masked': train_data['masked'][:,mini_data_ids,:]
            }
            # Train with each example
            _ = self._eval(self.update,mini_train_data)
            summary = self._eval(self.summary_op,validation_data)
            print("--- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()
            # self._save(epoch,mini_train_data,validation_data)
            sys.stdout.flush()
            writer.add_summary(summary, global_step=epoch)
        writer.close()
        print("stopping after epoch limit")

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
        unlabeled_loss, leave_k_out_loss, regularize_val, objective, accuracy, U_np, V_np, b_np \
        = self._eval([self.metrics['unlabeled_loss'],
                      self.metrics['leave_k_out_loss'],
                      self.metrics['regularize_val'],
                      self.metrics['objective'],
                      self.metrics['accuracy'],
                      self.U,
                      self.V,
                      self.b],
                      train_data)
        # normalize regularization value by regularization parameter
        regularize_val = regularize_val / self.regularize if self.regularize else 0
        # metrics related to validation data
        self.yhat_np, true_loss, true_accuracy \
        = self._eval([self.yhat,
                      self.metrics['true_loss'],
                      self.metrics['true_accuracy']],
                      validation_data)
        # save important parameters
        self.unlabeled_losses.append(unlabeled_loss)
        self.leave_k_out_losses.append(leave_k_out_loss)
        self.accuracies.append(accuracy)
        self.true_losses.append(true_loss)
        self.true_accuracies.append(true_accuracy)
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
              "true unlabeled loss:",true_loss,
              "true accuracy:",true_accuracy,
              "U:",U_np[:10],
              "V:",V_np[:10],
              "b:",b_np
              # "yhat:",self.yhat_np[:10]
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
        # normalize weights
        start_time = time.time()
        def condition(j,h):
            return j < num_iter

        def loop(j,h):
            Z = h
            N = h
            h = self._propagate(h,Z,N)
            # # clamp the labels
            # h = tf.nn.softmax(self.clamp_param * h,dim=0)
            # labeled nodes will remain the same
            h = h * (1-labeled) + X * labeled

            return j+1,h

        _,yhat = tf.while_loop(condition, loop, loop_vars=[0,X])
        #
        # for iter in range(num_iter):
        #     Z = h
        #     N = h
        #     h = self._propagate(h,Z,N)
        #     # h = tf.nn.softmax(1 * h,dim=0)
        #     h = h * (1-labeled) + X * labeled
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

        # self.U = tf.Variable(tf.zeros([self.num_hidden,self.num_classes]))
        # self.V = tf.Variable(tf.zeros([self.num_hidden,self.num_features]))
        if seed == 0:
            self.U = tf.Variable(tf.zeros([self.num_hidden,self.num_classes]))
            self.V = tf.Variable(tf.zeros([self.num_hidden,self.num_features]))
        if seed == 1:
            self.num_hidden = self.num_classes
            self.U = tf.Variable(0.1 * tf.eye(self.num_classes))
            self.V = tf.Variable(0.1 * tf.eye(self.num_features))
        if seed == 2:
            self.num_hidden = self.num_classes
            self.U = tf.Variable(0.1 * tf.eye(self.num_classes) + tf.random_normal([self.num_classes,self.num_classes],mean=0, stddev=0.001))
            self.V = tf.Variable(0.1 * tf.eye(self.num_features)+ tf.random_normal([self.num_classes,self.num_classes],mean=0, stddev=0.001))
        else:
            self.U = tf.Variable(tf.random_normal([self.num_hidden,self.num_classes], mean=0, stddev=0.1))
            self.V = tf.Variable(tf.random_normal([self.num_hidden,self.num_features], mean=0, stddev=0.1))
        self.b = tf.Variable(1, dtype=tf.float32)

    def _propagate(self,h,Z,N):
        start_time = time.time()
        # U*Z then take the indices
        row_features = tf.gather(Z,self.row_indices,axis=2)
        col_features = tf.gather(N,self.col_indices,axis=2)
        if self.has_features:
            col_features = tf.concat([col_features,tf.tile(tf.expand_dims(tf.transpose(self.edge_features), 1),[1,tf.shape(col_features)[1],1])],axis=0)
        A=tf.tensordot(self.U,row_features,axes=1)
        A.set_shape([self.num_hidden,None,self.num_edges])
        B=tf.tensordot(self.V,col_features,axes=1)
        B.set_shape([self.num_hidden,None,self.num_edges])
        new_hs = tf.ones([1,self.num_classes,self.num_nodes], tf.float32) # base array for concatenating new results
        i = tf.constant(0)
        num_samples = tf.shape(h)[1]
        W_edges = self.edges

        def condition(i,m):
            return i < num_samples

        def loop(sample_id,new_hs):
            W_vals = tf.reduce_sum(tf.gather(A,sample_id, axis=1)*tf.gather(B,sample_id, axis=1),axis=0) + self.b
            W_shape = [self.num_nodes,self.num_nodes]
            W = tf.SparseTensor(W_edges, W_vals, W_shape)
            Wnorm = tf.sparse_transpose(tf.sparse_softmax(tf.sparse_transpose(W)))
            new_h = tf.transpose(tf.sparse_tensor_dense_matmul(Wnorm, tf.transpose(tf.gather(h,sample_id, axis=1))))
            new_hs = tf.concat([new_hs, [new_h]], 0)
            return [sample_id+1,new_hs]

        _,new_hs = tf.while_loop(condition, loop, loop_vars=[i,new_hs],shape_invariants=[i.get_shape(),
                                                           tf.TensorShape([None,self.num_classes,self.num_nodes])])
        new_hs = tf.gather(new_hs,tf.range(1,num_samples+1), axis=0)
        new_hs = tf.transpose(new_hs,perm=[1, 0, 2])

        return new_hs

    def _regularize_loss(self,regularize):

        regularize = tf.constant(regularize,dtype=tf.float32)

        if self.regularize_type == 'l1':
            regularizer = tf.contrib.layers.l1_regularizer(scale=regularize)
        elif self.regularize_type == 'l2':
            regularizer = tf.contrib.layers.l2_regularizer(scale=regularize)

        to_regularize = [self.U, self.V]
        regularization_penalty = tf.contrib.layers.apply_regularization(regularizer, to_regularize)

        print('hiiiiiiiiiiiiiiiiiiiiiii')

        return regularization_penalty
