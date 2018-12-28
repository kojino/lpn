import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import os
import pandas as pd
from scipy.special import comb
from scipy.sparse import csr_matrix


def array_to_one_hot(vec, num_classes=None):
    "Convert multiclass labels to one-hot encoding"
    num_nodes = len(vec)
    if not num_classes:
        num_classes = len(set(vec))
    res = np.zeros((num_nodes, num_classes))
    res[np.arange(num_nodes), vec.astype(int)] = 1
    return res


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(data_path, seed, unlabel_prob):
    """Load data, train/val only."""
    np.random.seed(seed)

    print("Loading graph...")
    graph = np.loadtxt(data_path + '/Gsym.csv', delimiter=',')
    num_nodes = int(max(max(graph[:, 0]), max(graph[:, 1])) + 1)
    adj = csr_matrix(
        (graph[:, 2], (graph[:, 0].astype(int), graph[:, 1].astype(int))),
        shape=(num_nodes, num_nodes))

    print("Loading features...")
    features = np.loadtxt(data_path + '/x.csv', delimiter=',')
    num_nodes = int(np.max(features[:, 0]) + 1)
    num_features = int(np.max(features[:, 1]) + 1)
    features = csr_matrix(
        (features[:, 2], (features[:, 0], features[:, 1])),
        shape=(num_nodes, num_features))

    print("Loading labels...")
    true_labels = np.loadtxt(data_path + '/y.csv', delimiter=',')

    true_labels = array_to_one_hot(true_labels)
    labels = true_labels.copy().astype(float)
    unlabels = true_labels.copy().astype(float)

    train_mask = np.zeros(num_nodes)
    train_mask.fill(True)

    num_classes = labels.shape[1]
    print(num_classes)
    labeled_indices_from_class = []
    for class_id in range(num_classes):
        labeled_indices_from_class.append(
            np.random.choice(np.where(labels[:, class_id])[0]))
    # sample indices to unlabel
    unlabeled_indices = np.array(
        sorted(
            np.random.choice(
                [
                    i for i in range(num_nodes)
                    if i not in labeled_indices_from_class
                ],
                int(num_nodes * unlabel_prob),
                replace=False)))
    labeled_indices = np.delete(np.arange(num_nodes), unlabeled_indices)

    train_mask.ravel()[unlabeled_indices] = False
    labels[unlabeled_indices] = 0
    unlabels[labeled_indices] = 0

    val_mask = np.logical_not(train_mask)

    # return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
    return adj, features, labels, unlabels, train_mask, val_mask


def load_data_2(data_path, seed, unlabel_prob):
    """Load data, train/val/test"""
    np.random.seed(seed)
    """Load data."""
    print("Loading graph...")
    graph = np.loadtxt(data_path + '/Gsym.csv', delimiter=',')
    num_nodes = int(max(max(graph[:, 0]), max(graph[:, 1])) + 1)
    adj = csr_matrix(
        (graph[:, 2], (graph[:, 0], graph[:, 1])),
        shape=(num_nodes, num_nodes))

    print("Loading features...")
    features = np.loadtxt(data_path + '/x.csv', delimiter=',')
    num_nodes = int(np.max(features[:, 0]) + 1)
    num_features = int(np.max(features[:, 1]) + 1)
    features = csr_matrix(
        (features[:, 2], (features[:, 0], features[:, 1])),
        shape=(num_nodes, num_features))

    print("Loading labels...")
    true_labels = np.loadtxt(data_path + '/y.csv', delimiter=',')
    true_labels = array_to_one_hot(true_labels)

    num_classes = true_labels.shape[1]
    print(num_classes)
    labeled_indices_from_class = []
    for class_id in range(num_classes):
        labeled_indices_from_class.append(
            np.random.choice(np.where(true_labels[:, class_id])[0]))
    # sample indices to unlabel
    unlabeled_indices = np.array(
        sorted(
            np.random.choice(
                [
                    i for i in range(num_nodes)
                    if i not in labeled_indices_from_class
                ],
                int(num_nodes * unlabel_prob),
                replace=False)))
    labeled_indices = np.delete(np.arange(num_nodes), unlabeled_indices)
    print(len(labeled_indices), len(unlabeled_indices))

    # labeled indices not used for validation
    unvalidation_labeled_indices_from_class = []
    for class_id in range(num_classes):
        unvalidation_labeled_indices_from_class.append(
            np.random.choice(
                np.where(true_labels[labeled_indices, class_id])[0]))
    validation_labeled_indices = np.array(
        sorted(
            np.random.choice(
                [
                    i for i in labeled_indices
                    if i not in unvalidation_labeled_indices_from_class
                ],
                int(len(labeled_indices) * 0.2),
                replace=False)))
    unvalidation_labeled_indices = np.array(
        [el for el in labeled_indices if el not in validation_labeled_indices])
    # print(labeled_indices,unvalidation_labeled_indices,validation_labeled_indices)
    # print(len(validation_labeled_indices),len(unvalidation_labeled_indices),len(labeled_indices))

    train_mask, val_mask, test_mask = np.zeros(num_nodes), np.zeros(
        num_nodes), np.zeros(num_nodes)
    train_mask.fill(False)
    val_mask.fill(False)
    test_mask.fill(False)
    train_mask.ravel()[unvalidation_labeled_indices] = True
    val_mask.ravel()[validation_labeled_indices] = True
    test_mask.ravel()[unlabeled_indices] = True

    # print(np.sum(train_mask),np.sum(val_mask),np.sum(test_mask),true_labels.shape[0])

    y_train, y_val, y_test = true_labels.copy().astype(
        float), true_labels.copy().astype(float), true_labels.copy().astype(
            float)
    y_train[unlabeled_indices] = 0
    y_train[validation_labeled_indices] = 0
    y_val[unvalidation_labeled_indices] = 0
    y_val[unlabeled_indices] = 0
    y_test[unvalidation_labeled_indices] = 0
    y_test[validation_labeled_indices] = 0

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update(
        {placeholders['support'][i]: support[i]
         for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(
        adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
