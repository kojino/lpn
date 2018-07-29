from __future__ import print_function
import sys
from scipy import sparse
from scipy.sparse.csgraph import laplacian
from sklearn.metrics import f1_score
from tensorflow.python.client import timeline
from deeplp.models.utils import sparse_tensor_dense_tensordot
import tensorflow as tf
import numpy as np
import scipy as sp
import time
import math
import csv


class DeepLP:
    """
    Base class for Deep Label Propagation (DeepLP).
    Inherit this class to implement different link functions.
    This class by itself will not run to give a meaningful result.
    """
    def __init__(self, profile,
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
                       regularize_theta=0,
                       regularize_weight=0,
                       seed=1,
                       sparse_edges=0,
                       clamp_scale=0.1):
        """Create weight matrix (depends on link functions) and build graph.
        Arguments:
            profile: if 1, store profiling json file every epoch
            log_name: folder to store tensorboard files to
            features: features used to construct the weights of the graph
            graph: graph of the nodes
            num_classes: number of label classes
            unlabeled_indices: set of unlabeled node indices
            confidence: if 1, add an additional "confidence" class to LP
            clamp: if 1, apply clamping to the output each iteration
            loss_class_mass: if 1, loss is calculated as the weighted sum
                             according to the class prior probability
            lr: learning rate of the optimizer
            num_iter: number of LP iterations
            regularize: regularization parameter x, 2**x is used
            seed: seed to be used for random initialization of parameters
            weo
        """
        np.random.seed(seed)
        tf.set_random_seed(seed)

        if regularize_theta < 2**-25:
            regularize_theta = 0.0
        print(regularize_theta)
        # mandatory parameters
        self.model       = model
        self.log_name    = log_name
        self.graph = self._scipy_sparse_to_sparse_tensor(graph)
        self.num_classes = num_classes
        self.unlabeled_indices = unlabeled_indices
        self.profile     = profile
        self.sparse_edges = sparse_edges

        # optional parameters
        self.confidence  = confidence
        self.clamp       = clamp
        self.clamp_scale = clamp_scale
        self.loss_class_mass = loss_class_mass
        self.lr          = lr
        self.num_iter    = num_iter
        self.regularize  = regularize_theta
        regularize_theta = regularize_theta
        regularize_weight = regularize_weight
        self.seed = seed

        # initialize weights
        self.num_nodes   = graph.shape[0]
        self.labeled_indices = np.delete(np.arange(self.num_nodes),unlabeled_indices)

        if len(features) == 0:
            self.num_features = 0
        else:
            self.num_features = features.shape[1]
        self.weights     = self._init_weights(features, graph, seed)

        # see g(x) under _forwardprop for how these parameters are used
        if self.clamp:
            print("cllaaaaamp")
            self.c = 1 + self.clamp_scale * tf.Variable(0,dtype=tf.float32)
            self.a = self.clamp_scale * tf.Variable(0,dtype=tf.float32)
            self.d = 0.1 * self.clamp_scale * tf.Variable(0,dtype=tf.float32)
        else:
            self.c = tf.constant(0.0,dtype=tf.float32)
            self.a = tf.constant(0.0,dtype=tf.float32)
            self.d = tf.constant(0.0,dtype=tf.float32)
        #
        # if self.change_b:
        #     print("self loop")
        #     self.self_weight = tf.sigmoid(tf.Variable(tf.constant(-1.0)))
        # else:
        #     print("no self loop")
        #     self.self_weight = tf.constant(0.0,dtype=tf.float32)

        # initialize placeholders
        shape             = [self.num_nodes, None, self.num_classes]
        self.epoch        = tf.placeholder("float", shape=())
        self.X            = tf.placeholder("float", shape=shape)
        self.y            = tf.placeholder("float", shape=shape)
        self.labeled      = tf.placeholder("float", shape=shape)
        self.true_labeled = tf.placeholder("float", shape=shape)
        self.validation_labeled = tf.placeholder("float", shape=shape)
        self.masked       = tf.placeholder("float", shape=shape)

        # ratio of labeled nodes in each class
        count = tf.reduce_sum(self.y * self.true_labeled,axis=0,keep_dims=True)
        self.class_prior = count / tf.reduce_sum(count,axis=2,keep_dims=True)

        self.yhat         = self._forwardprop(self.X,
                                              self.weights,
                                              self.labeled,
                                              self.num_iter)
        self.update, self.metrics = self._backwardprop(self.y,
                                                         self.yhat,
                                                         self.labeled,
                                                         self.true_labeled,
                                                         self.validation_labeled,
                                                         self.masked,
                                                         regularize_theta,
                                                         regularize_weight,
                                                         self.lr)

    def train(self,train_data,validation_data,epochs=1000,profile=False):
        """Train the model and store the performance every epoch.
        Arguments:
            train_data: data to train the model, labeled nodes are masked
            validation_data: data to evaluate the performance,
                             labeleed nodes are unmasked
            epochs: number of epochs to train the model
        Returns:
            None
        """
        self.epoch_val = -1
        if self.profile:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self._open_sess(config)
        else:
            self._open_sess()
        start_time = time.time()
        writer = tf.summary.FileWriter('log_local2/'+self.log_name, self.sess.graph)
        self.summary = tf.Summary()
        saver = tf.train.Saver()

        if self.profile:
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            builder = tf.profiler.ProfileOptionBuilder
            opts = builder(builder.time_and_memory()).order_by('micros').build()
            pctx = tf.contrib.tfprof.ProfileContext('memory_dir',
                                                  trace_steps=[],
                                                  dump_steps=[])
        else:
            options = None
            run_metadata = None


        # lists to store the performance metrics
        self.clamp_params = []
        self.entropies = []
        self.unlabeled_losses = []
        self.leave_k_out_losses = []
        self.accuracies = []
        self.true_losses = []
        self.true_accuracies = []
        self.validation_accuracies = []
        self.objectives = []
        self.micro_f1s = []
        self.macro_f1s = []
        self.regularize_vals = []
        self._save(-1,train_data,validation_data)
        self.y_np = self._eval(self.y, validation_data)
        self.nanned = False
        self.clamp_params = []

        import os
        if not os.path.exists('ckpt'):
            os.makedirs('ckpt')

        # writer = tf.summary.FileWriter('log_local/test2', self.sess.graph)
        for epoch in range(epochs):
            self.epoch_val = epoch
            if self.profile:
                pctx.trace_next_step()
            # Train with each example
            self.summary = tf.Summary()
            _,summary = self._eval([self.update,self.summary_op],train_data,options=options,run_metadata=run_metadata)
            # _ = self._eval(self.update,train_data,options=options,run_metadata=run_metadata)
            print("--- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()
            self._save(epoch,train_data,validation_data)
            if self.nanned:
                print("oh no! nan!!")
                break
            writer.add_summary(summary, global_step=epoch)
            writer.add_summary(self.summary, global_step=epoch)
            writer.flush()
            if self.profile:
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open('timeline_02_step_%d.json' % epoch, 'w') as f:
                    f.write(chrome_trace)
                pctx.dump_next_step()
                pctx.profiler.profile_operations(options=opts)
            if epoch % 100 == 0:
                print('saving checkpoint')
                save_path = saver.save(self.sess, f"ckpt/{self.log_name}/model.ckpt")
        writer.close()
        print("stopping after epoch limit")

    def _backwardprop(self, y,
                            yhat,
                            labeled,
                            true_labeled,
                            validation_labeled,
                            masked,
                            regularize_theta,
                            regularize_weight,
                            lr):
        """Backward prop on unlabeled + masked labeled nodes.
        Objective is the sum of leave-k-out loss and regularization.
        Calculate loss and accuracy for both train and validation dataset.
        Arguments:
            y: true labels
            yhat: predicted labels from forward prop
            labeled: boolean indicating labels of labels unmasked
            true_labeled: boolean indicating labels of true labels
            masked: boolean indicating labeled nodes that are masked
        Returns:
            None
        """
        start_time = time.time()
        l_o_loss       = self._calc_loss(y,yhat,masked)
        # l_o_loss = tf.constant(0.0)
        regularize_val = self._regularize_loss(regularize_theta,regularize_weight)
        objective      = l_o_loss + regularize_val
        optimizer = tf.train.AdamOptimizer(lr)
        update = tf.contrib.slim.learning.create_train_op(objective, optimizer,summarize_gradients=True)

        # evaluate performance
        loss          = self._calc_loss(y,yhat,1-labeled)
        accuracy      = self._calc_accuracy(y,yhat,1-labeled)
        entropy       = self._calc_entropy(yhat,1-labeled)
        kl_uniform    = self._calc_kl_uniform(yhat,1-labeled)
        kl_class_prior= self._calc_kl_class_prior(yhat,1-labeled)
        true_loss     = self._calc_loss(y,yhat,1-true_labeled)
        true_accuracy = self._calc_accuracy(y,yhat,1-true_labeled)
        validation_accuracy = self._calc_accuracy(y,yhat,1-validation_labeled)

        self.y=y
        self.yhat=yhat

        np.random.seed(self.seed)
        #
        tf.summary.scalar('unlabeled_loss', loss)
        tf.summary.scalar('leave_k_out_loss', l_o_loss)
        tf.summary.scalar('objective', objective)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('entropy', entropy)
        # tf.summary.scalar('kl_uniform', kl_uniform)
        # tf.summary.scalar('kl_class_prior', kl_class_prior)
        # tf.summary.scalar('true_loss', true_loss)
        # tf.summary.scalar('true_accuracy', true_accuracy)
        tf.summary.scalar('validation_accuracy', validation_accuracy)

        tf.summary.scalar('clamp_a', self.a)
        tf.summary.scalar('clamp_c', self.c)
        tf.summary.scalar('clamp_d', self.d)

        if self.model == 'wrbf':
            tf.summary.histogram("theta", self.theta)
        if self.model == 'edge':
            tf.summary.histogram("b", self.b)
            tf.summary.histogram("theta", self.theta)
            tf.summary.histogram("weights", self.weights.values)
            tf.summary.histogram("weights_unnormalized", self.weights_unnormalized.values)
        if self.model == 'att':
            # tf.summary.scalar("clamping", self.clamp_param)
            tf.summary.histogram("theta", self.thetas)
            tf.summary.histogram("theta_row", self.thetas[0])
            tf.summary.histogram("theta_div", self.thetas[1])
            # tf.summary.histogram("self_theta", self.self_theta)
            if self.has_features:
                tf.summary.histogram("edge_weights", self.edge_weights)

            # if self.change_b:
            #     tf.summary.scalar("self_weight", self.self_weight)
                # tf.summary.histogram("node_features", self.self_theta[0])
                # tf.summary.scalar("t", self.self_theta[1])
                # tf.summary.scalar("t quad", self.self_theta[2])
                # tf.summary.scalar("entropy", self.self_theta[3])
                # tf.summary.scalar("avg_neigh", self.self_theta[4])
                # tf.summary.scalar("max_neigh", self.self_theta[5])
            # tf.summary.histogram("theta_div", self.thetas[2])
            # tf.summary.histogram("b", self.b)

        self.summary_op = tf.summary.merge_all()

        metrics = {
            'unlabeled_loss': loss,
            'leave_k_out_loss': l_o_loss,
            'regularize_val': regularize_val,
            'objective': objective,
            'accuracy': accuracy,
            'entropy': entropy,
            'kl_uniform': kl_uniform,
            'kl_class_prior': kl_class_prior,
            'true_loss': true_loss,
            'true_accuracy': true_accuracy,
            'validation_accuracy': validation_accuracy,
        }
        return update, metrics

    def _calc_accuracy(self,y,yhat,mask):
        """Calculate prediction accuracy
        Arguments:
            yhat: num_classes by num_samples by num_nodes matrix of predictions
            y: true labels of same dimension
            mask: booleans indicating where the predictions should be evaluated
        Returns:
            accuracy
        """
        yhat = tf.argmax(yhat,axis=2)
        y = tf.argmax(y,axis=2)
        mask = tf.reduce_max(mask,axis=2)
        acc_mat = mask * tf.cast(tf.equal(yhat,y),tf.float32)
        return tf.reduce_sum(acc_mat) / tf.count_nonzero(mask,dtype=tf.float32)

    def _calc_loss(self,y,yhat,mask, eps=0.00001):
        """Calculate prediction loss
        Arguments:
            yhat: num_classes by num_samples by num_nodes matrix of predictions
            y: true labels of same dimension
            mask: booleans indicating where the predictions should be evaluated
            eps: theshold for log loss
        Returns:
            loss
        """

        yhat = tf.minimum(tf.maximum(yhat, eps), 1)
        # loss_mat = mask * ((y - yhat) ** 2)
        # loss = (tf.reduce_sum(loss_mat) / tf.count_nonzero(mask,dtype=tf.float32)) * self.num_classes

        if self.loss_class_mass:
            # counts = tf.reduce_sum(tf.reduce_sum(y,axis=1),axis=1)
            # proportion = tf.cast(tf.shape(mask)[1], tf.float32) / tf.cast(tf.shape(mask)[0], tf.float32) * counts
            # proportion = proportion / tf.reduce_sum(proportion)
            # proportion = tf.expand_dims(tf.expand_dims(proportion,axis=-1),axis=-1)
            loss_mat = self.class_prior * mask * y * tf.log(yhat) * (-1)
        else:
            loss_mat = mask * y * tf.log(yhat) * (-1)
        loss = tf.reduce_sum(loss_mat) / tf.count_nonzero(mask*y,dtype=tf.float32)
        return loss

    def _calc_entropy(self,yhat,mask, eps=0.0000000001):
        self.entropy_mat = tf.reduce_sum(yhat * tf.log(yhat+0.000000001) * mask * -1,axis=2)
        entropy = tf.reduce_sum(self.entropy_mat)/tf.count_nonzero(tf.reduce_mean(mask,axis=2),dtype=tf.float32)
        return entropy

    def _calc_kl_uniform(self,yhat,mask, eps=0.0000000001):
        self.kl_mat = tf.reduce_sum(tf.log(yhat+0.000000001) * mask * -1,axis=2)
        entropy = tf.reduce_sum(self.kl_mat)/tf.count_nonzero(tf.reduce_mean(mask,axis=2),dtype=tf.float32)
        return entropy

    def _calc_kl_class_prior(self,yhat,mask, eps=0.0000000001):
        self.kl_mat = tf.reduce_sum(self.class_prior * tf.log(yhat+0.000000001) * mask * -1,axis=2)
        entropy = tf.reduce_sum(self.kl_mat)/tf.count_nonzero(tf.reduce_mean(mask,axis=2),dtype=tf.float32)
        return entropy

    def _eval(self,vals,data=None,options=None,run_metadata=None):
        """Evaluate list of tensors with given data
        Arguments:
            vals: list of tensors to evaluate
            data: data to be passed into feed_dict
        Returns:
            evaluated values
        """
        if data:
            return self.sess.run(vals, feed_dict={self.X:data['X'],
                                                  self.y:data['y'],
                                                  self.labeled:data['labeled'],
                                                  self.true_labeled:data['true_labeled'],
                                                  self.validation_labeled:data['validation_labeled'],
                                                  self.masked:data['masked'],
                                                  self.epoch:self.epoch_val},
                                       options=options,
                                       run_metadata=run_metadata)
        else:
            return self.sess.run(vals)

    def _forwardprop(self, X,
                           weights,
                           labeled,
                           num_iter):
        """Forward prop which mimicks LP
        Arguments:
            X: input labels where some labeled nodes are masked
            weights: graph adjacency weight matrix
            labeled: boolean indicating whether the labels are labeled
            num_iter: number of hidden layers
        Returns:
            yhat: predictions over all nodes
        """
        print('start')
        # self.Wu = tf.gather(self.weights,self.unlabeled_indices,axis=0)
        # self.Ws = tf.gather(self.weights,self.labeled_indices,axis=0)
        # self.masked_s = tf.gather(self.masked,self.labeled_indices,axis=0)
        # self.ys = tf.gather(X,self.labeled_indices,axis=0)

        def g1(x):
            return 1 / (1 + (x / (1 - x)) ** (-self.c))

        def g2(x):
            return x ** self.c

        def g3(x,t):
            return x ** (self.c + self.a * t)

        def condition(j,h):
            return j < float(num_iter)

        def loop(j,h):
            print('before prpagate')
            propa = self._propagate(j,h)
            print('after prpagate')
            if self.model == 'att':
                if self.change_b:
                    print(self.self_weight,h)
                    h = self.self_weight * h+ (1 - self.self_weight) * propa
                else:
                    h = propa
            else:
                h = propa
            # else:
            # # hu = propa
            #     h = tf.transpose(tf.map_fn(lambda x: tf.contrib.sparsemax.sparsemax(x), tf.transpose(h,perm=[2, 0, 1])),perm=[1,2,0])

            # clamp the labels
            if self.clamp:
                print('clamping.......................')
                eps = 0.0001
                if self.clamp == 1:
                    h = g1(tf.clip_by_value(h,eps,1-eps))
                elif self.clamp == 2:
                    h = g2(tf.clip_by_value(h,eps,1-eps))
                elif self.clamp == 3:
                    h = g3(tf.clip_by_value(h,eps,1-eps),j)
                h = h / tf.reduce_sum(h,axis=2,keep_dims=True)

            #     h = tf.contrib.sparsemax.sparsemax(self.clamp_param * h)

                # h = tf.nn.softmax(self.c * h,axis=2)
            # h = tf.nn.softmax(1 * h,dim=0)
            # labeled nodes will remain the same
            h = h * (1-labeled) + X * labeled

            return j+1.0,h
        print(num_iter)
        _,yhat = tf.while_loop(condition, loop, loop_vars=[0.0,X])
        return yhat

    def _propagate(self,j,h):
        # # apply LP per class
        # num_samples = tf.shape(h)[1]
        # new_hs = tf.ones([1,self.num_nodes,num_samples], tf.float32)
        # i = tf.constant(0)
        # def condition(i,new_hs):
        #     return i < self.num_classes
        #
        # def loop(i,new_hs):
        #     new_h = tf.sparse_tensor_dense_matmul(self.weights, tf.transpose(h[i]))
        #     # if self.clamp:
        #     #     new_h = tf.contrib.sparsemax.sparsemax(new_h)
        #     new_hs = tf.concat([new_hs, [new_h]], 0)
        #     return [i+1,new_hs]
        #
        # _,new_hs = tf.while_loop(condition, loop, loop_vars=[i,new_hs],shape_invariants=[i.get_shape(),
        #                                                    tf.TensorShape([None,self.num_nodes,None])])
        # new_hs = tf.gather(new_hs,tf.range(1,self.num_classes+1))
        # new_hs = tf.transpose(new_hs,perm=[0, 2, 1])
        print('-------------propagate------------')
        print(h,self.weights)
        return sparse_tensor_dense_tensordot(self.weights, h, axes=1)
        # hu = self.Wu @ h
        # hs = (self.Ws @ h) * (1 - self.masked_s) + self.ys * self.masked_s


    def _init_weights(self, features, graph, param, seed):
        """Initialize weights
        This function is typically overwritten depending on the link function.
        Arguments:
            features: features for ecah node
            graph: adjacency matrix of the nodes
            param: parameters used for initialization if seed randomization
                   is not prefered
            seed: seed to be used for random initialization of parameters
        Returns:
            weights in tf variable
        """
        weights = tf.convert_to_tensor(graph, np.float32)
        return tf.Variable(weights)

    def _normalize_weights(self, weights):
        """Column normalize weights
        Arguments:
            weights: weight matrix
        Returns:
            normalized weight matrux
        """
        Tnorm = weights / tf.sparse_reduce_sum(weights, axis = 1, keep_dims=True) # TODO make sure this is correct
        return Tnorm

    def _open_sess(self,config=None):
        """Open tf session for training the data"""
        if config:
            self.sess = tf.Session(config=config)
        else:
            self.sess = tf.Session()
        init      = tf.global_variables_initializer()
        self.sess.run(init)

    def _regularize_loss(self,regularize):
        """Return regularization value.
        This function is typically overwritten depending on the link function.
        Arguments:
            regularize: regularization parameter
        Returns:
            regularization value
        """
        return 0

    def _sparse_tensor_value_to_scipy_sparse(self,sparse_tensor_value):
        return sparse.csr_matrix((sparse_tensor_value.values,
                    (sparse_tensor_value.indices[:,0],sparse_tensor_value.indices[:,1])),
                    shape=sparse_tensor_value.dense_shape)

    def _scipy_sparse_to_sparse_tensor(self,X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, tf.constant(coo.data,dtype=tf.float32), coo.shape)

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

        unlabeled_loss, leave_k_out_loss, regularize_val, objective, accuracy, \
        regularize_val, yhat_np, clamp_c, clamp_a, clamp_d \
        = self._eval([self.metrics['unlabeled_loss'],
                      self.metrics['leave_k_out_loss'],
                      self.metrics['regularize_val'],
                      self.metrics['objective'],
                      self.metrics['accuracy'],
                      self.metrics['regularize_val'],
                      self.yhat,
                      self.c,
                      self.a,
                      self.d],
                      train_data)
        # Wunnorm_np = self._sparse_tensor_value_to_scipy_sparse(Wunnorm)
        # print(Wunnorm_np.shape,np.sum(Wunnorm_np,axis=0),np.sum(Wunnorm_np,axis=0).shape)
        # print(Wunnorm_np[self.labeled_indices])
        # Wunp = self._sparse_tensor_value_to_scipy_sparse(W)
        # print(W_np.shape,np.sum(W_np,axis=0),np.sum(W_np,axis=0).shape)
        # print(W_np[self.unlabeled_indices])
        if self.model != 'att':
            weights_sparse_tensor_value, weights_unnormalized_sparse_tensor_value, values \
            = self._eval([self.weights,
                          self.weights_unnormalized,
                          self.weights.values],
                          train_data)
            weights_unnormalized_np = self._sparse_tensor_value_to_scipy_sparse(weights_unnormalized_sparse_tensor_value)
            # L = laplacian(weights_unnormalized_np, normed=False)
            # vals, vecs = sparse.linalg.eigsh(L,k=2,which='SM')
            # self.lambda2 = vals[1].real
            # self.summary.value.add(tag="lambda2", simple_value=self.lambda2)
            # print("# Edges:",len(values))
            # print("10^-6 < 10^-5:",np.sum(np.logical_and(values > 10**(-6) , values < 10**(-5))))
            # print("10^-5 < 10^-4:",np.sum(np.logical_and(values > 10**(-5) , values < 10**(-4))))
            # print("10^-4 < 10^-3:",np.sum(np.logical_and(values > 10**(-4) , values < 10**(-3))))
            # print("10^-3 < 10^-2:",np.sum(np.logical_and(values > 10**(-3) , values < 10**(-2))))
            # print("10^-2 < 10^-1:",np.sum(np.logical_and(values > 10**(-2) , values < 10**(-1))))
            # print("10^-1 < 10^0:",np.sum(np.logical_and(values > 10**(-1) , values < 10**0)))
            # print("0.1 < 0.2:",np.sum(np.logical_and(values > 0.1 , values < 0.2)))
            # print("0.2 < 0.5:",np.sum(np.logical_and(values > 0.2 , values < 0.5)))
            # print("0.5 < :",np.sum(values > 0.5))
            # print(values[:20])
        else:
            thetas, theta_times, self_weight \
            = self._eval([self.thetas,
                          self.theta_times,
                          self.self_weight],
                          train_data)

        # normalize regularization value by regularization parameter
        regularize_val = regularize_val / self.regularize if self.regularize else 0
        # metrics related to validation data
        yhat_np, y_np, entropy, kl_uniform, kl_class_prior, true_loss, true_accuracy, validation_accuracy, entropy_mat \
        = self._eval([self.yhat,
                      self.y,
                      self.metrics['entropy'],
                      self.metrics['kl_uniform'],
                      self.metrics['kl_class_prior'],
                      self.metrics['true_loss'],
                      self.metrics['true_accuracy'],
                      self.metrics['validation_accuracy'],
                      self.entropy_mat],
                      validation_data)


        entropy_sd = np.std(entropy_mat[0])
        signaltonoise = entropy / entropy_sd
        self.summary.value.add(tag="entropy_sd", simple_value=entropy_sd)
        self.summary.value.add(tag="signaltonoise", simple_value=signaltonoise)
        print(np.sum(yhat_np,axis=2),np.sum(yhat_np,axis=2).shape)


        y_true = np.argmax(y_np[self.unlabeled_indices,0,:].T,axis=1)
        y_pred = np.argmax(yhat_np[self.unlabeled_indices,0,:].T,axis=1)
        micro_f1=f1_score(y_true,y_pred,average='micro')
        macro_f1=f1_score(y_true,y_pred,average='macro')
        self.summary.value.add(tag="micro_f1", simple_value=micro_f1)
        self.summary.value.add(tag="macro_f1", simple_value=macro_f1)

        # save important parameters
        self.unlabeled_losses.append(unlabeled_loss)
        self.leave_k_out_losses.append(leave_k_out_loss)
        self.accuracies.append(accuracy)
        self.true_losses.append(true_loss)
        self.true_accuracies.append(true_accuracy)
        self.validation_accuracies.append(validation_accuracy)
        self.objectives.append(objective)
        self.entropies.append(entropy)
        self.micro_f1s.append(micro_f1)
        self.macro_f1s.append(macro_f1)
        self.regularize_vals.append(regularize_val)

        with open(f'accs/{self.log_name}_accs.csv', 'a') as csvfile:
        # with open(f'accs.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([true_accuracy,validation_accuracy])

        # print parameters every epoch
        print("== ",
              "epoch:",epoch,
              "labeled loss:",leave_k_out_loss,
            #   "regularize_val",regularize_val,
            #   "regularize_param",self.regularize,
            #   "objective",objective,
            #   "unlabeled loss:",unlabeled_loss,
              "accuracy:",accuracy,
            #   "entropy:",entropy,
            #   "kl_uniform:",kl_uniform,
            #   "kl_class_prior:",kl_class_prior,
            #   "true unlabeled loss:",true_loss,
            #   "true accuracy:",true_accuracy,
            #   "validation_accuracy:",validation_accuracy,
            #   "clamp_c:",clamp_c,
            #   "clamp_a:",clamp_a,
            #   "clamp_d:",clamp_d,
            #   "signaltonoise:",signaltonoise,
            #   "entropy_sd:",entropy_sd
              )
        # if self.model != 'att':
        #     print("=== ",
        #           "weights:", weights_sparse_tensor_value.values[:10],
        #           "weights_unnormalized:", weights_unnormalized_sparse_tensor_value.values[:10])

        # else:
        #     print("=== ",
        #           "thetas:", thetas,
        #           "theta_times:", theta_times,
        #           "self_weight:",self_weight)
        #           # "node_features:", node_features,
        #           # "t:",t,
        #           # "neg_t:",neg_t,
        #           # "entropy:",entropy,
        #           # "avg_neigh:",avg_neigh,
        #           # "max_neigh:",max_neigh,
        sys.stdout.flush()


        #     wr = csv.writer(fp, dialect='excel')
        #     wr.writerow([clamp_a])
        # with open("bs.csv", "a") as fp:
        #     wr = csv.writer(fp, dialect='excel')
        #     wr.writerow([clamp_c])

        if math.isnan(objective):
            self.nanned = True
        # save and print parameters specific to each link function
        self._save_params(epoch,train_data)

    def _save_params(self,epoch,data):
        """Save and print parameters specific to each link function
        This function is typically overwritten depending on the link function.
        Arguments:
            epoch: current epoch
            data: data for parameter evaluation specific to the link function
        Returns:
            None
        """
        return
