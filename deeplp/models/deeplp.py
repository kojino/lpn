import numpy as np
import scipy as sp
import tensorflow as tf

EPS = 10e-10


class DeepLP:
    """Base class for Deep Label Propagation (DeepLP)."""

    def __init__(self,
                 graph,
                 features,
                 num_classes,
                 num_features,
                 num_nodes,
                 weighted_loss=False,
                 lr=0.01,
                 num_layers=100,
                 lamda=0,
                 seed=1,
                 bifurcation=None,
                 decay=0):

        self.graph = graph
        self.features = features
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_nodes = num_nodes
        self.weighted_loss = weighted_loss
        self.num_layers = num_layers
        self.lamda = lamda
        self.seed = seed
        self.bifurcation = bifurcation
        self.row_indices, self.col_indices = graph.tocoo().row, graph.tocoo(
        ).col
        self.indices = list(zip(self.row_indices, self.col_indices))

        # initialize placeholders
        shape = [self.num_nodes, None, self.num_classes]
        self.X = tf.placeholder("float", shape=shape)
        self.y = tf.placeholder("float", shape=shape)
        self.labeled = tf.placeholder("float", shape=shape)
        self.true_labeled = tf.placeholder("float", shape=shape)
        self.validation_labeled = tf.placeholder("float", shape=shape)
        self.masked = tf.placeholder("float", shape=shape)
        self.weights = self._init_weights()
        self.global_step = tf.placeholder(tf.int32)
        if decay:
            self.lr = tf.train.cosine_decay_restarts(lr, self.global_step,
                                                     decay)
        else:
            self.lr = lr

        # see bifurcate function for how these parameters are used
        if self.bifurcation != None:
            self.a = self.bifurcation * tf.Variable(
                0, dtype=tf.float32, name='bif_a')
            self.b = self.bifurcation * tf.Variable(
                0, dtype=tf.float32, name='bif_b')

        self.opt_op = self._build()
        self._summary()
        self.summary_op = tf.summary.merge_all()

    def _init_weights(self):
        pass

    def _build(self):
        """
        Forward prop which mimicks LP
        """

        def bifurcate(h, t):
            return h**(self.a * t + self.b + 1.0)

        def condition(h, t):
            return t < self.num_layers

        def loop(h, t):
            h = self._propagate(t, h)
            if self.bifurcation:
                h = bifurcate(h, t)
                h = h / tf.reduce_sum(h + EPS, axis=2, keepdims=True)
            h = h * (1 - self.labeled) + self.X * self.labeled
            return h, t + 1.0

        self.yhat, _ = tf.while_loop(condition, loop, loop_vars=[self.X, 0.0])

        self.l_o_loss = self._loss(self.y, self.yhat, self.masked)
        regularize_val = self._regularize_loss()
        self.objective = self.l_o_loss + regularize_val
        optimizer = tf.train.AdamOptimizer(self.lr)
        opt_op = tf.contrib.slim.learning.create_train_op(
            self.objective, optimizer, summarize_gradients=True)
        return opt_op

    def _propagate(self, j, h):
        return h

    def _summary(self):
        # evaluate performance
        self.loss = self._loss(self.y, self.yhat, 1 - self.true_labeled)
        self.accuracy = self._accuracy(self.y, self.yhat,
                                       1 - self.true_labeled)
        self.validation_accuracy = self._accuracy(self.y, self.yhat,
                                                  self.validation_labeled)

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('leave_k_out_loss', self.l_o_loss)
        tf.summary.scalar('objective', self.objective)
        tf.summary.scalar('accuracy', self.accuracy)

        if self.bifurcation != None:
            tf.summary.scalar('bifurcation_a', self.a)
            tf.summary.scalar('bifurcation_b', self.b)

    def _accuracy(self, y, yhat, mask):
        yhat = tf.argmax(yhat, axis=2)
        y = tf.argmax(y, axis=2)
        mask = tf.reduce_max(mask, axis=2)
        acc_mat = mask * tf.cast(tf.equal(yhat, y), tf.float32)
        return tf.reduce_sum(acc_mat) / tf.count_nonzero(
            mask, dtype=tf.float32)

    def _loss(self, y, yhat, mask):
        yhat = tf.minimum(tf.maximum(yhat, EPS), 1.0)

        if self.weighted_loss:
            # ratio of labeled nodes in each class
            count = tf.reduce_sum(
                self.y * self.true_labeled, axis=0, keepdims=True)
            class_prior = count / tf.reduce_sum(count, axis=2, keepdims=True)
            loss_mat = class_prior * mask * y * tf.log(yhat) * (-1)
        else:
            loss_mat = mask * y * tf.log(yhat) * (-1)
        loss = tf.reduce_sum(loss_mat) / tf.count_nonzero(
            mask * y, dtype=tf.float32)
        return loss

    def _regularize_loss(self):
        return 0
