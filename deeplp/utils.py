import logging
import os

import networkx as nx
import numpy as np
import scipy as sp
from community import community_louvain
import tensorflow as tf
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial.distance import (braycurtis, canberra, chebyshev, cityblock,
                                    correlation, cosine, euclidean, jaccard,
                                    mahalanobis, minkowski, pdist, squareform)
from scipy.special import comb
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression, lasso_path
from sklearn.metrics.pairwise import (euclidean_distances, paired_distances,
                                      rbf_kernel)
from sklearn.preprocessing import normalize

logger = logging.getLogger("deeplp")
EPS = 1e-10

num_layers_dict = {
    'linqs_citeseer': 90,
    'linqs_cora': 20,
    'linqs_pubmed': 20,
    'flip_cora': 10,
    'flip_dblp': 30,
    'flip_flickr': 10,
    'flip_imdb': 70,
    'flip_industry': 40
}


def load_data(data_path, model='edge', feature_type='all'):
    """
    Load data from given directory.
    Must have y.csv and G.csv where each row is:
        y.csv: (labels)
        G.csv: (source, sink)
    For attention, must have features as well:
        x.csv: (feature1, feature2, ...)
    """
    os.path.realpath(__file__)
    
    y_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', f'data/{data_path}/labels.csv'))
    G_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', f'data/{data_path}/graph_symmetric.csv'))
    if feature_type == 'all':
        x_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', f'data/{data_path}/features.csv'))
    elif feature_type == 'raw_reduced':
        x_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', f'data/{data_path}/features_raw_reduced.csv'))
    else:
        if model == 'wrbf':
            x_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', f'data/{data_path}/node_features.csv'))
        else:
            x_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', f'data/{data_path}/features_raw.csv'))
    assert os.path.isfile(y_path), "Label file labels.csv must exist."
    assert os.path.isfile(G_path), "Graph file graph_symmetric.csv must exist."
    if model == 'edge':
        assert os.path.isfile(x_path), "Label file features.csv must exist."

    true_labels = np.loadtxt(y_path, delimiter=',')
    true_labels = array_to_one_hot(true_labels)
    logger.info(f"Loaded labels: {true_labels.shape}")

    graph = np.loadtxt(G_path, delimiter=',')
    graph = edge_np_to_csr(graph)
    logger.info(f"Loaded graph: {graph.shape}")

    if model in ['edge', 'wrbf']:
        features = np.loadtxt(x_path, delimiter=',')
        if model == 'wrbf':
            features = node_features_np_to_dense(features)
        logger.info(f"Loaded features: {features.shape}")
    else:
        features = np.array([])

    logger.info("Loaded all data.")

    return true_labels, features, graph


def prepare_data(model, labels, is_labeled, labeled_indices, held_out_indices,
                 true_labels, leave_k, num_samples, seed):
    """
    Construct masked data and validation data to be fed into the network.
    Data will have its labeled nodes masked according to leave_k.

    There are two types of data returned:
    - validation_data: No leave-k-out, full data.
                       Hence, input is (num_nodes, 1, num_classes)
    - train_data: Each row corresponds to leave-k-out data.
                  Hence, input is (num_nodes, num_samples, num_classes)
    """
    np.random.seed(seed)
    num_nodes, num_classes = labels.shape

    # set number of samples
    max_num_samples = sp.special.comb(
        len(labeled_indices), leave_k, exact=True)
    if max_num_samples < num_samples:
        num_samples = max_num_samples

    # true labels: y
    true_labeled = np.repeat(
        is_labeled.reshape(num_nodes, 1), num_classes, axis=-1)
    true_labeled = true_labeled.reshape(num_nodes, 1, num_classes)
    y = np.tile(true_labels, 1).reshape(num_nodes, 1, num_classes)

    # boolean indicating labeled/masked nodes
    labeled = np.repeat(true_labeled, num_samples, axis=1)
    masked = np.zeros((num_nodes, num_samples, num_classes))

    # for cross validation, create a mask of validation indices
    if held_out_indices is not None:
        validation_labeled = np.zeros(num_nodes)
        validation_labeled.fill(False)
        validation_labeled.ravel()[held_out_indices] = True
        validation_labeled = validation_labeled.reshape(
            validation_labeled.shape[0], 1)
        validation_labeled = np.repeat(validation_labeled, num_classes, axis=1)
        validation_labeled = validation_labeled.reshape((num_nodes, 1,
                                                         num_classes))
        # np.max(validation_labeled,axis=2) * (np.sum(y,axis=2) == np.sum(y,axis=2)) / 
    else:
        validation_labeled = true_labeled
    # construct validation data
    validation_data = {
        model.X: labels.reshape(num_nodes, 1, num_classes),
        model.y: y,
        model.labeled: true_labeled,
        model.true_labeled: true_labeled,  # this will not be used
        model.validation_labeled: validation_labeled,
        model.masked: masked  # this will not be used
    }

    X = np.tile(labels, num_samples).reshape(num_nodes, num_samples,
                                             num_classes)
    if leave_k == 1:
        # if leave-1-out sample without replacment
        # such that no indices will be selected twice
        for i, index_to_mask in enumerate(
                np.random.permutation(labeled_indices)[:num_samples]):
            X[index_to_mask, i, :] = 1 / num_classes
            labeled[index_to_mask, i, :] = 0
            masked[index_to_mask, i, :] = 1
    else:
        # otherwise, it is unlikely that same index tuples are selected twice.
        # hence, sample with replacement
        for i in range(num_samples):
            indices_to_mask = np.random.permutation(labeled_indices)[:leave_k]
            X[indices_to_mask, i, :] = 1 / num_classes
            labeled[indices_to_mask, i, :] = 0
            masked[indices_to_mask, i, :] = 1

    # construct data used to train the model
    train_data = {
        model.X: X,
        model.y: y,
        model.labeled: labeled,
        model.true_labeled: true_labeled,
        model.validation_labeled: validation_labeled,
        model.masked: masked
    }

    return train_data, validation_data, num_samples



def random_unlabel(true_labels, unlabel_prob, seed=None):
    """
    Randomly unlabel nodes.
    Always have at least one node from each class.
    """
    np.random.seed(seed)
    num_nodes, num_classes = true_labels.shape
    num_unlabeled = int(num_nodes * unlabel_prob)

    # from each class, sample at least one index for labeled
    labeled_indices_from_class = []
    for class_id in range(num_classes):
        labeled_indices_from_class.append(
            np.random.choice(np.where(true_labels[:, class_id])[0]))

    # sample indices to unlabel
    indices_left = [
        i for i in range(num_nodes) if i not in labeled_indices_from_class
    ]
    unlabeled_indices = np.random.choice(
        indices_left, num_unlabeled, replace=False)
    unlabeled_indices = np.array(sorted(unlabeled_indices))
    labeled_indices = np.delete(np.arange(num_nodes), unlabeled_indices)
    logger.info(f"Randomly unlabel {num_unlabeled} indices.")

    return labeled_indices, unlabeled_indices


def calc_masks(true_labels, labeled_indices, unlabeled_indices):
    """
    From labeled and unlabeled indices, return masks indicating
    whether each node is labeled or not.
    """
    num_nodes = len(labeled_indices) + len(unlabeled_indices)

    # is_labeled: num_nodes x 1 matrix indicating labeling
    is_labeled = np.zeros(num_nodes)
    is_labeled.fill(True)
    is_labeled.ravel()[unlabeled_indices] = False
    is_labeled = is_labeled.reshape(is_labeled.shape[0], 1)

    # labels: num_nodes x num_classes matrix indicating labeling
    labels = true_labels.copy().astype(float)
    # assign uniform probability to unlabled nodes
    k = labels.shape[1]
    labels[unlabeled_indices] = 1 / k

    return labels, is_labeled


def create_seed_features(graph, labeled_indices, true_labels):
    """
    For each node, for each label, calculate the shortest and average
    distance to seed node with that label.
    """
    U = nx.from_scipy_sparse_matrix(graph)  # undirected
    B = U.to_directed()
    edges = np.array(B.edges())
    sources, _ = edges[:, 0], edges[:, 1]

    # calculate shortest path length to each seed node
    seed_to_node_lengths = []  # num_labeled * num_nodes matrix
    for i in labeled_indices:
        shortest_paths_seed = nx.shortest_path_length(B, source=int(i))
        path_lengths = [i[1] for i in sorted(shortest_paths_seed.items())]
        seed_to_node_lengths.append(path_lengths)
    seed_to_node_lengths = np.array(seed_to_node_lengths)

    # create label => list of seed indices dict
    labels_for_seeds = np.argmax(true_labels[labeled_indices], axis=1)
    labels_for_seeds_dict = {}
    for i, label in enumerate(labels_for_seeds):
        if label in labels_for_seeds_dict:
            labels_for_seeds_dict[label].append(i)
        else:
            labels_for_seeds_dict[label] = [i]

    # for each label, find the closest (or average) distance to
    # seed with that label
    seed_features = []
    for label in labels_for_seeds_dict:
        indices = labels_for_seeds_dict[label]
        label_seed_to_node_lengths = seed_to_node_lengths[indices]
        label_min_len_to_seed = np.min(
            label_seed_to_node_lengths, axis=0)[sources]
        label_mean_len_to_seed = np.mean(
            label_seed_to_node_lengths, axis=0)[sources]
        seed_features.append(label_min_len_to_seed)
        seed_features.append(label_mean_len_to_seed)

    # find overall closest (or average) distance to seed
    min_len_to_seed = np.min(seed_to_node_lengths, axis=0)[sources]
    mean_len_to_seed = np.mean(seed_to_node_lengths, axis=0)[sources]
    seed_features.append(min_len_to_seed)
    seed_features.append(mean_len_to_seed)

    # normalize
    seed_features = np.array(seed_features).T
    seed_features = pos_normalize(seed_features)
    logger.info("Generated seed features.")

    return seed_features


def array_to_one_hot(vec, num_classes=None):
    "Convert multiclass labels to one-hot encoding"
    num_nodes = len(vec)
    if not num_classes:
        num_classes = len(set(vec))
    res = np.zeros((num_nodes, num_classes))
    res[np.arange(num_nodes), vec.astype(int)] = 1
    return res


def edge_np_to_csr(graph, vals=[]):
    """
    Convert np array with each row being (row_index,col_index,value)
    of a graph to a scipy csr matrix.
    """
    num_nodes = int(max(max(graph[:, 0]), max(graph[:, 1])) + 1)
    if len(vals) == 0:
        vals = np.ones(len(graph[:, 0]))
    csr = csr_matrix(
        (vals, (graph[:, 0], graph[:, 1])), shape=(num_nodes, num_nodes))
    return csr


def node_features_np_to_dense(node_features, values=False):
    """
    Convert np array with each row being (row_index,col_index,value)
    of a graph to a scipy csr matrix.
    """
    num_rows = int(max(node_features[:, 0]) + 1)
    num_cols = int(max(node_features[:, 1]) + 1)
    vals = np.ones(len(node_features[:, 0]))
    csr = csr_matrix(
        (vals, (node_features[:, 0], node_features[:, 1])),
        shape=(num_rows, num_cols))
    return csr.toarray()


def node_features_np_to_sparse(node_features, values=False):
    """
    Convert np array with each row being (row_index,col_index,value)
    of a graph to a scipy csr matrix.
    """
    num_rows = int(max(node_features[:, 0]) + 1)
    num_cols = int(max(node_features[:, 1]) + 1)
    vals = np.ones(len(node_features[:, 0]))
    csr = csr_matrix(
        (vals, (node_features[:, 0], node_features[:, 1])),
        shape=(num_rows, num_cols))
    return csr


def indices_to_vec(labeled_indices, num_nodes):
    res = np.zeros(num_nodes)
    res[labeled_indices] = 1
    return res


def log_loss(y, yhat):
    "Calcualte log loss of predictions"
    yhat[yhat < EPS] = EPS
    yhat[yhat > 1 - EPS] = 1 - EPS
    log_yhat = np.log(yhat)
    return np.sum((y * log_yhat) / np.count_nonzero(y)) * (-1)


def mse(y, yhat, num_classes):
    "Calcualte mse loss of predictions"
    return np.mean((yhat - y)**2) * num_classes


def prob_to_one_hot(prob):
    "Convert prediction probabilities to one-hot by taking the argmax"
    return (prob == prob.max(axis=1)[:, None]).astype(int)


def pos_normalize(features):
    diff = np.max(
        features, axis=0, keepdims=True) - np.min(
            features, axis=0, keepdims=True)
    return (features - np.min(features, axis=0, keepdims=True)) / diff


def sparse_tensor_dense_tensordot(sp_a, b, axes, name=None):
    """Modification of tf tensordot API to allow for sparse tensor dot product."""

    def _tensordot_reshape(a, axes, flipped=False):
        """Helper method to perform transpose and reshape for contraction op."""
        if a.get_shape().is_fully_defined() and isinstance(
                axes, (list, tuple)):
            shape_a = a.get_shape().as_list()
            axes = [i if i >= 0 else i + len(shape_a) for i in axes]
            free = [i for i in range(len(shape_a)) if i not in axes]
            free_dims = [shape_a[i] for i in free]
            prod_free = int(np.prod([shape_a[i] for i in free]))
            prod_axes = int(np.prod([shape_a[i] for i in axes]))
            perm = list(axes) + free if flipped else free + list(axes)
            new_shape = [prod_axes,
                         prod_free] if flipped else [prod_free, prod_axes]
            reshaped_a = tf.reshape(tf.transpose(a, perm), new_shape)
            return reshaped_a, free_dims, free_dims
        else:
            if a.get_shape().ndims is not None and isinstance(
                    axes, (list, tuple)):
                shape_a = a.get_shape().as_list()
                axes = [i if i >= 0 else i + len(shape_a) for i in axes]
                free = [i for i in range(len(shape_a)) if i not in axes]
                free_dims_static = [shape_a[i] for i in free]
            else:
                free_dims_static = None
            shape_a = tf.shape(a)
            rank_a = tf.rank(a)
            axes = tf.convert_to_tensor(axes, dtype=tf.int32, name="axes")
            axes = tf.cast(axes >= 0, tf.int32) * axes + tf.cast(
                axes < 0, tf.int32) * (axes + rank_a)
            free, _ = tf.setdiff1d(tf.range(rank_a), axes)
            free_dims = tf.gather(shape_a, free)
            axes_dims = tf.gather(shape_a, axes)
            prod_free_dims = tf.reduce_prod(free_dims)
            prod_axes_dims = tf.reduce_prod(axes_dims)
            perm = tf.concat([axes_dims, free_dims], 0)
            if flipped:
                perm = tf.concat([axes, free], 0)
                new_shape = tf.stack([prod_axes_dims, prod_free_dims])
            else:
                perm = tf.concat([free, axes], 0)
                new_shape = tf.stack([prod_free_dims, prod_axes_dims])
            reshaped_a = tf.reshape(tf.transpose(a, perm), new_shape)
            return reshaped_a, free_dims, free_dims_static

    def _sparse_tensordot_reshape(a, axes, flipped=False):
        """Helper method to perform transpose and reshape for contraction op."""
        if a.get_shape().is_fully_defined() and isinstance(
                axes, (list, tuple)):
            shape_a = a.get_shape().as_list()
            axes = [i if i >= 0 else i + len(shape_a) for i in axes]
            free = [i for i in range(len(shape_a)) if i not in axes]
            free_dims = [shape_a[i] for i in free]
            prod_free = int(np.prod([shape_a[i] for i in free]))
            prod_axes = int(np.prod([shape_a[i] for i in axes]))
            perm = list(axes) + free if flipped else free + list(axes)
            new_shape = [prod_axes,
                         prod_free] if flipped else [prod_free, prod_axes]
            reshaped_a = tf.sparse_reshape(
                tf.sparse_transpose(a, perm), new_shape)
            return reshaped_a, free_dims, free_dims
        else:
            if a.get_shape().ndims is not None and isinstance(
                    axes, (list, tuple)):
                shape_a = a.get_shape().as_list()
                axes = [i if i >= 0 else i + len(shape_a) for i in axes]
                free = [i for i in range(len(shape_a)) if i not in axes]
                free_dims_static = [shape_a[i] for i in free]
            else:
                free_dims_static = None
            shape_a = tf.shape(a)
            rank_a = tf.rank(a)
            axes = tf.convert_to_tensor(axes, dtype=tf.int32, name="axes")
            axes = tf.cast(axes >= 0, tf.int32) * axes + tf.cast(
                axes < 0, tf.int32) * (axes + rank_a)
            free, _ = tf.setdiff1d(tf.range(rank_a), axes)
            free_dims = tf.gather(shape_a, free)
            axes_dims = tf.gather(shape_a, axes)
            prod_free_dims = tf.reduce_prod(free_dims)
            prod_axes_dims = tf.reduce_prod(axes_dims)
            perm = tf.concat([axes_dims, free_dims], 0)
            if flipped:
                perm = tf.concat([axes, free], 0)
                new_shape = tf.stack([prod_axes_dims, prod_free_dims])
            else:
                perm = tf.concat([free, axes], 0)
                new_shape = tf.stack([prod_free_dims, prod_axes_dims])
            reshaped_a = tf.sparse_reshape(
                tf.sparse_transpose(a, perm), new_shape)
            return reshaped_a, free_dims, free_dims_static

    def _sparse_tensordot_axes(a, axes):
        """Generates two sets of contraction axes for the two tensor arguments."""
        a_shape = a.get_shape()
        if isinstance(axes, tf.compat.integral_types):
            if axes < 0:
                raise ValueError("'axes' must be at least 0.")
            if a_shape.ndims is not None:
                if axes > a_shape.ndims:
                    raise ValueError(
                        "'axes' must not be larger than the number of "
                        "dimensions of tensor %s." % a)
                return (list(range(a_shape.ndims - axes, a_shape.ndims)),
                        list(range(axes)))
            else:
                rank = tf.rank(a)
                return (range(rank - axes, rank, dtype=tf.int32),
                        range(axes, dtype=tf.int32))
        elif isinstance(axes, (list, tuple)):
            if len(axes) != 2:
                raise ValueError("'axes' must be an integer or have length 2.")
            a_axes = axes[0]
            b_axes = axes[1]
            if isinstance(a_axes, tf.compat.integral_types) and \
                    isinstance(b_axes, tf.compat.integral_types):
                a_axes = [a_axes]
                b_axes = [b_axes]
            if len(a_axes) != len(b_axes):
                raise ValueError(
                    "Different number of contraction axes 'a' and 'b', %s != %s."
                    % (len(a_axes), len(b_axes)))
            return a_axes, b_axes
        else:
            axes = tf.convert_to_tensor(axes, name="axes", dtype=tf.int32)
        return axes[0], axes[1]

    with tf.name_scope(name, "SparseTensorDenseTensordot",
                       [sp_a, b, axes]) as name:
        #         a = tf.convert_to_tensor(a, name="a")
        b = tf.convert_to_tensor(b, name="b")
        sp_a_axes, b_axes = _sparse_tensordot_axes(sp_a, axes)
        sp_a_reshape, sp_a_free_dims, sp_a_free_dims_static = _sparse_tensordot_reshape(
            sp_a, sp_a_axes)
        b_reshape, b_free_dims, b_free_dims_static = _tensordot_reshape(
            b, b_axes, True)
        ab_matmul = tf.sparse_tensor_dense_matmul(sp_a_reshape, b_reshape)
        if isinstance(sp_a_free_dims, list) and isinstance(b_free_dims, list):
            return tf.reshape(
                ab_matmul, sp_a_free_dims + b_free_dims, name=name)
        else:
            sp_a_free_dims = tf.convert_to_tensor(
                sp_a_free_dims, dtype=tf.int32)
            b_free_dims = tf.convert_to_tensor(b_free_dims, dtype=tf.int32)
            product = tf.reshape(
                ab_matmul,
                tf.concat([sp_a_free_dims, b_free_dims], 0),
                name=name)
            if sp_a_free_dims_static is not None and b_free_dims_static is not None:
                product.set_shape(sp_a_free_dims_static + b_free_dims_static)
            return product


def load_graph(data_path):
    os.path.realpath(__file__)
    
    G_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', f'data/{data_path}/graph_symmetric.csv'))
    x_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', f'data/{data_path}/features_raw.csv'))
    assert os.path.isfile(G_path), "Graph file graph_symmetric.csv must exist."
    graph = np.loadtxt(G_path, delimiter=',')
    graph = edge_np_to_csr(graph)
    logger.info(f"Loaded graph: {graph.shape}")

    node_features = np.loadtxt(x_path, delimiter=',')
    node_features = node_features_np_to_dense(node_features)

    logger.info(f"Loaded node features: {node_features.shape}")

    Gdir = np.loadtxt(G_path, delimiter=',')
    Gdir = edge_np_to_csr(Gdir)
    logger.info(f"Loaded graph: {Gdir.shape}")

    # make graphs
    D = nx.from_numpy_matrix(Gdir.toarray(), create_using=nx.MultiDiGraph())
    U = nx.Graph(D)
    B = U.to_directed()
    R = D.reverse()

    logger.info("Loaded all data.")

    return U, D, B, R, node_features


def listify(d):
    return np.array(list(dict(d).values()))


def lisify_links(preds):
    return np.array([val for _, _, val in preds])


def rbf(x1, x2, sigma=None):
    if not sigma:
        sigma = (1 / x2.shape[0])
    return np.exp(-1 * np.sum((x1 - x2)**2)) / sigma


def get_lowest_weight(row):
    l = np.sort(row[np.nonzero(row)])[::-1]
    if len(l) == 0:
        return 0
    else:
        return l[-1]


def one_hot(a):
    x = len(a)
    y = len(np.unique(a))
    b = np.zeros((x, y))
    b[np.arange(x), a] = 1
    return b


def get_u_to_b_indices(U, edges):
    edges_u = np.array([set(edge[:2]) for edge in U.edges])
    u_to_b_indices = []
    for edge in edges:
        index = np.where(edges_u == set(edge))[0][0]
        u_to_b_indices.append(index)
    return np.array(u_to_b_indices)


def get_d_to_b_indices(D, edges):
    edges_d = np.array([edge[:2] for edge in D.edges])
    d_to_b_indices = []
    for edge_d in edges_d:
        index = np.where(
            np.logical_and(edges[:, 0] == edge_d[0],
                           edges[:, 1] == edge_d[1]))[0]
        if len(index) == 1:
            d_to_b_indices.append(index[0])
    return np.array(d_to_b_indices)


def get_r_to_b_indices(R, edges):
    edges_r = np.array([edge[:2] for edge in R.edges])
    r_to_b_indices = []
    for edge_r in edges_r:
        index = np.where(
            np.logical_and(edges[:, 0] == edge_r[0],
                           edges[:, 1] == edge_r[1]))[0]
        if len(index) == 1:
            r_to_b_indices.append(index[0])
    return np.array(r_to_b_indices)


# 'cosine' 'euclidean', 'rbf'
def node_feature_similarities(node_features, sources, sinks):
    similarities = []
    row_features = node_features[sources]
    col_features = node_features[sinks]
    for metric in [cosine, euclidean, rbf]:
        similarities.append(
            paired_distances(row_features, col_features, metric=metric))
    similarities = np.array(similarities).T
    logger.info(f'node_feature_similarities generated: {similarities.shape}')
    return similarities


# 'LSI_feature1', 'LSI_feature2', 'LSI_feature3'
def node_feature_reduction(node_features, sources, n_components=3):
    svd = TruncatedSVD(n_components=n_components, n_iter=7, random_state=42)
    LSI_features = svd.fit_transform(node_features)[sources]
    logger.info(f'node_feature_reduction generated: {LSI_features.shape}')
    return LSI_features


# 'log_node_in_degrees_D', 'log_node_out_degrees_D', 'average_neighbor_degree'
def node_centralities(B, D, R, U, sources, sinks):
    node_in_degrees_D = listify(D.degree)[sources]
    node_out_degrees_D = listify(D.degree)[sinks]
    # node_in_degrees_R = listify(R.degree)[sources]
    # node_out_degrees_R = listify(R.degree)[sinks]
    log_node_in_degrees_D = np.log(1 + node_in_degrees_D)
    log_node_out_degrees_D = np.log(1 + node_out_degrees_D)
    average_neighbor_degree = listify(nx.average_neighbor_degree(D))[sources]

    node_centralities = np.vstack(
        (log_node_in_degrees_D, log_node_out_degrees_D,
         average_neighbor_degree)).T
    logger.info(f'node_centralities generated: {node_centralities.shape}')
    return node_centralities


# 'core_same', 'louvain_same'
def node_partitions(U, sources, sinks):
    core_numbers = listify(nx.core_number(U))[sources]
    core_same = core_numbers[sources] == core_numbers[sinks]
    louvain = listify(community_louvain.best_partition(U))[sources]
    louvain_same = louvain[sources] == louvain[sinks]
    res = [core_same, louvain_same]
    node_partitions = np.vstack(res).T
    logger.info(f'node_partitions generated: {node_partitions.shape}')
    return node_partitions


# 'edge_betweenness_D_full', 'edge_betweenness_R_full', 'edge_betweenness_B_full',
# 'edge_current_flow_betweenness_full'
def edge_centralities(B, D, R, U, edges, d_to_b_indices, r_to_b_indices,
                      u_to_b_indices):
    edge_betweenness_D = listify(nx.edge_betweenness_centrality(D))
    edge_betweenness_D_full = np.zeros(edges.shape[0])
    edge_betweenness_D_full[d_to_b_indices] = edge_betweenness_D
    edge_betweenness_D_full = np.log(edge_betweenness_D_full + EPS)
    edge_betweenness_B_full = np.log(
        listify(nx.edge_betweenness_centrality(B)) + EPS)
    edge_betweenness_R = listify(nx.edge_betweenness_centrality(R))
    edge_betweenness_R_full = np.zeros(edges.shape[0])
    edge_betweenness_R_full[r_to_b_indices] = edge_betweenness_R
    edge_betweenness_R_full = np.log(edge_betweenness_R_full + EPS)

    # listify(nx.edge_load_centrality(U))
    # communicability_exp = nx.communicability_exp(U)
    edge_current_flow_betweenness = listify(
        nx.edge_current_flow_betweenness_centrality(U))
    edge_current_flow_betweenness_full = edge_current_flow_betweenness[
        u_to_b_indices]
    edge_centralities = np.vstack(
        (edge_betweenness_D_full, edge_betweenness_R_full,
         edge_betweenness_B_full, edge_current_flow_betweenness_full)).T
    logger.info(f'edge_centralities generated: {edge_centralities.shape}')
    return edge_centralities


# 'jaccard_coefficient', 'adamic_adar_index', 'preferential_attachment'
def link_predictions(U, edges):
    # resource_allocation_index = lisify_links(nx.resource_allocation_index(U,edges))
    jaccard_coefficient = lisify_links(nx.jaccard_coefficient(U, edges))
    adamic_adar_index = lisify_links(nx.adamic_adar_index(U, edges))
    preferential_attachment = np.log(
        lisify_links(nx.preferential_attachment(U, edges)) + EPS)
    link_features = np.vstack((jaccard_coefficient, adamic_adar_index,
                               preferential_attachment)).T
    logger.info(f'link_predictions generated: {link_features.shape}')
    return link_features


# 'communities_same', 'within_inter_cluster'
def community_features(U, node_features, edges, sinks, sources):
    communities_dict = nx.algorithms.community.asyn_fluidc(U, k=5)
    communities = []
    for community in communities_dict:
        communities.append(list(community))
    community_ids = np.zeros(node_features.shape[0])
    for i in range(5):
        community_ids[communities[i]] = i
    communities_same = (
        community_ids[sources] == community_ids[sinks]).astype(int)
    for i in range(node_features.shape[0]):
        U.node[i]['community'] = int(community_ids[i])
    within_inter_cluster = lisify_links(
        nx.within_inter_cluster(U, ebunch=edges))
    community_features = np.vstack([communities_same, within_inter_cluster]).T
    logger.info(f'community_features generated: {community_features.shape}')
    return community_features


def approx_chunk(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def sigma_for_rbf(features, num_neighbors=10):
    """
    Calculate optimal values of sigma for RBF kernel.
    """

    # use rbf kernel to estimate weights between nodes
    def get_kth_nearest_weight(row):
        return np.sort(row)[::-1][num_neighbors - 1]

    D = euclidean_distances(features, features)
    knn_dists = np.apply_along_axis(get_kth_nearest_weight, 0, D)
    sigma = np.mean(knn_dists)**2
    return sigma