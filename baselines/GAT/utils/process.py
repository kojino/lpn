import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import os
"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""


def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


###############################################
# This section of code adapted from tkipf/gcn #
###############################################


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


def array_to_one_hot(vec, num_classes=None):
    "Convert multiclass labels to one-hot encoding"
    num_nodes = len(vec)
    if not num_classes:
        num_classes = len(set(vec))
    res = np.zeros((num_nodes, num_classes))
    res[np.arange(num_nodes), vec.astype(int)] = 1
    return res


def load_random_data(size):

    adj = sp.random(size, size, density=0.002)  # density similar to cora
    features = sp.random(size, 1000, density=0.015)
    int_labels = np.random.randint(7, size=(size))
    labels = np.zeros((size, 7))  # Nx7
    labels[np.arange(size), int_labels] = 1

    train_mask = np.zeros((size, )).astype(bool)
    train_mask[np.arange(size)[0:int(size / 2)]] = 1

    val_mask = np.zeros((size, )).astype(bool)
    val_mask[np.arange(size)[int(size / 2):]] = 1

    test_mask = np.zeros((size, )).astype(bool)
    test_mask[np.arange(size)[int(size / 2):]] = 1

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    # sparse NxN, sparse NxF, norm NxC, ..., norm Nx1, ...
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


def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


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


def preprocess_adj_bias(adj):
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose(
    )  # This is where I made a mistake, I used (adj.row, adj.col) instead
    # return tf.SparseTensor(indices=indices, values=adj.data, dense_shape=adj.shape)
    return indices, adj.data, adj.shape
