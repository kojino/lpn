import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys


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


def load_data(dataset_str, seed, setting, num_ratio):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("gcn_original/data/ind.{}.{}".format(dataset_str, names[i]),
                  'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "gcn_original/data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    num_nodes, num_classes = ally.shape
    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder),
            max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)
    G = nx.from_dict_of_lists(graph)
    gcc_indices = set(
        max(nx.connected_component_subgraphs(G), key=len).nodes())
    nogcc_indices = np.delete(
        np.arange(num_nodes), np.array(list(gcc_indices)))
    unlabeled_indices = np.delete(np.arange(num_nodes), idx_train)
    gcc_unlabeled_indices = list(
        set.intersection(gcc_indices, set(unlabeled_indices)))
    nogcc_unlabeled_indices = list(
        set.intersection(set(nogcc_indices), set(unlabeled_indices)))

    if seed != -1:
        print('randomize seed')
        print(len(idx_val))
        np.random.seed(seed)
        num_unlabeled = num_nodes - len(idx_train)
        # from each class, sample at least one index for labeled
        if 'random' in setting:
            if 'val_as_train' in setting:
                print('val_as_train')
                num_unlabeled = num_nodes - len(idx_train) - len(idx_val)
                print(num_nodes,num_unlabeled,len(idx_train), len(idx_val))
            print('random')
            labeled_indices_from_class = []
            for class_id in range(num_classes):
                labeled_indices_from_class.append(
                    np.random.choice(np.where(ally[:, class_id])[0]))

            # sample indices to unlabel
            indices_left = [
                i for i in range(num_nodes)
                if i not in labeled_indices_from_class
            ]
            unlabeled_indices = np.random.choice(
                indices_left, num_unlabeled, replace=False)
            unlabeled_indices = np.array(sorted(unlabeled_indices))
            labeled_indices = np.delete(
                np.arange(num_nodes), unlabeled_indices)
            test_val_indices = np.random.choice(unlabeled_indices,
                                                len(idx_val) + len(idx_test))
            validation_indices = np.random.choice(
                labeled_indices, int(0.2 * len(labeled_indices)), replace=False)
            new_labeled_indices = [el for el in list(labeled_indices) if el not in list(validation_indices)]
            labeled_indices = np.array(new_labeled_indices)
            test_indices = np.random.choice(unlabeled_indices, len(idx_test), replace=False)
            print(len(idx_val))
        elif 'balanced' in setting:
            print('balanced 20')
            labeled_indices = []
            for class_id in range(num_classes):
                labeled_indices += list(
                    np.random.choice(
                        np.where(ally[:, class_id])[0], 20, replace=False))
            unlabeled_indices = np.delete(
                np.arange(num_nodes), labeled_indices)
            test_val_indices = np.random.choice(unlabeled_indices,
                                                len(idx_val) + len(idx_test))
            validation_indices = test_val_indices[:len(idx_val)]
            test_indices = test_val_indices[len(validation_indices):]

        idx_train = list(labeled_indices)
        idx_val = list(validation_indices)
        idx_test = list(test_indices)
        gcc_unlabeled_indices = list(
            set.intersection(gcc_indices, set(unlabeled_indices)))
        nogcc_unlabeled_indices = list(
            set.intersection(set(nogcc_indices), set(unlabeled_indices)))
    else:
        print('fixed seed')

    train_mask = sample_mask(idx_train, labels.shape[0])
    num_val = int(len(idx_train) * num_ratio)
    if num_val > len(idx_val):
        num_val = len(idx_val)
    idx_val = np.random.choice(idx_val,num_val,replace=False)
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    unlabeled_mask = sample_mask(list(unlabeled_indices), labels.shape[0])
    gcc_mask = sample_mask(list(gcc_unlabeled_indices), labels.shape[0])
    nogcc_mask = sample_mask(list(nogcc_unlabeled_indices), labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_unlabeled = np.zeros(labels.shape)
    y_gcc = np.zeros(labels.shape)
    y_nogcc = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    y_unlabeled[unlabeled_mask, :] = labels[unlabeled_mask, :]
    y_gcc[gcc_mask, :] = labels[gcc_mask, :]
    y_nogcc[nogcc_mask, :] = labels[nogcc_mask, :]
    print(len(idx_train),len(idx_val),len(idx_test))

    return adj, features, y_train, y_val, y_test, y_unlabeled, y_gcc, y_nogcc, train_mask, val_mask, test_mask, unlabeled_mask, gcc_mask, nogcc_mask


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
