import numpy as np
import scipy as sp
import tensorflow as tf

from deeplp.models.deeplp import DeepLP

EPS = 10e-10


class DeepLP_ATT(DeepLP):
    """
    Attention based model for graph data with no features.
    """

    def _init_weights(self):
        """
        Initialize weights for the weighted sum of entropy and divergence.
        Note that the weights differ each layer in attention model,
        so it's not initialized here.
        """
        self.theta = tf.Variable(
            tf.constant(np.zeros((2, self.num_classes)), dtype=tf.float32))

    def _summary(self):
        """
        Summarize weights for entropy and kl divergence
        """
        super()._summary()
        tf.summary.histogram("theta_entropy", self.theta[0])
        tf.summary.histogram("theta_kl", self.theta[1])

    def _propagate(self, t, h):
        """
        Propagate labels for each sample with different weights.
        There are two features:
        - entropy feature: column (source) entropy for each edge
        - kl divergence feature: divergence of two nodes in each edge
        Weight matrix differs for each sample because entropy and divergence
        are different according to the masked labels.
        As a result, we loop over samples and propagate sample by sample.
        """

        def kl_divergence(p, q):
            # calculate kl divergence terms given two tensors
            return p * tf.log(p / (q + EPS) + EPS)

        ## entropy features ##
        # reshape entropy weights
        entropy_theta = tf.expand_dims(
            tf.expand_dims(self.theta[0], axis=0), axis=0)
        # calculate weighted entropy tensor
        entropy_tensor = -1 * entropy_theta * h * tf.log(h + EPS)
        entropies_mat = tf.reduce_sum(entropy_tensor, axis=2)
        # get source entropy for each directed edge
        col_entropies = tf.gather(entropies_mat, self.col_indices, axis=0)
        # normalize entropy by its maximum value
        max_entropy = np.log(self.num_classes)
        col_entropies = (max_entropy - col_entropies) / max_entropy

        ## kl divergence feature ##
        kl_theta = tf.expand_dims(
            tf.expand_dims(self.theta[1], axis=0), axis=0)
        row_features = tf.gather(h, self.row_indices, axis=0)
        col_features = tf.gather(h, self.col_indices, axis=0)
        # kl of source and sink nodes
        kl_tensor = kl_theta * kl_divergence(col_features, row_features)
        kl_mat = tf.reduce_sum(kl_tensor, axis=2) / np.log(1 / EPS + EPS)

        ## propagation for each sample ##
        # base array for concatenating new results
        new_h = tf.ones([1, self.num_nodes, self.num_classes], tf.float32)
        curr_sample = tf.constant(0)
        num_samples = tf.shape(h)[1]

        def condition(curr_sample, new_h):
            return curr_sample < num_samples

        def loop(curr_sample, new_h):
            # weight values are the sum of entropy and kl features
            W_vals = tf.gather(col_entropies, curr_sample, axis=1) + \
                tf.gather(kl_mat, curr_sample, axis=1)
            W_shape = [self.num_nodes, self.num_nodes]
            W = tf.SparseTensor(self.indices, W_vals, W_shape)
            Wnorm = tf.sparse_softmax(W)
            # propagate labels
            new_sample = tf.sparse_tensor_dense_matmul(
                Wnorm, tf.gather(h, curr_sample, axis=1))
            # append to result
            new_h = tf.concat([new_h, [new_sample]], 0)
            return [curr_sample + 1, new_h]

        _, new_h = tf.while_loop(
            condition,
            loop,
            loop_vars=[curr_sample, new_h],
            shape_invariants=[
                curr_sample.get_shape(),
                tf.TensorShape([None, self.num_nodes, self.num_classes])
            ])
        # the first row is ignored
        new_h = tf.gather(new_h, tf.range(1, num_samples + 1), axis=0)
        # transpose back to the original shape
        new_h = tf.transpose(new_h, perm=[1, 0, 2])

        return new_h

    def _regularize_loss(self):
        """
        L2 reguralization of the weights.
        """
        regularizer = tf.contrib.layers.l2_regularizer(scale=self.lamda)
        theta_penalty = tf.contrib.layers.apply_regularization(
            regularizer, [self.theta])
        return theta_penalty
