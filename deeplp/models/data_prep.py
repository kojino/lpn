"""
Functions for loading and structuring datasets.

- Datasets from LINQS
(https://linqs.soe.ucsc.edu/data)
"""

import numpy as np
import os
import scipy as sp

from scipy.special import comb
from scipy.sparse import csr_matrix
from sklearn import decomposition
from sklearn import metrics
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import lasso_path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import normalize


from deeplp.models.utils import array_to_one_hot

def create_weighted_graph(features,graph):
    """Use RBF kernel to calculate the weights of edges
    Input:
        features: features to calculate the RBF kernel
        sigma: RBF kernel parameter if None, set it to num_features * 10
        graph: if [], construct a graph using RBF kernel, if given, use it
        num_neighbors: number of neighbors (based on edge weights) to form edges
    Returns:
        weight matrix and graph matrix
    """

    print('Constructing weights...')

    features_dense = features.toarray()

    # estimate sigma by taking the average of the lowest weights for each node
    def get_lowest_weight(row):
        return np.sort(row[np.nonzero(row)])[::-1][-1]
    D = euclidean_distances(features_dense, features_dense)
    lowest_dists = np.apply_along_axis(get_lowest_weight,0,D*graph)
    sigma = np.mean(lowest_dists)**2

    # use rbf kernel to estimate weights between nodes
    weights_dense = rbf_kernel(features_dense, gamma=1/sigma)

    weights_sp = graph.multiply(weights_dense)

    print('Done!')

    return weights, graph, sigma


def load_data(data,datatype,directed=0,confidence=False,model='edge'):
    """Load datasets
    Arguments:
        data: name of dataset to load
        num_components: number of features to extract when applying PCA or l1
                        when None, use the full feature
    Returns:
        true_labels: array of true labels in one-hot encoding
        features: features of each node
        graph: graph adjacency matrix, if it doesn't exist, an empty array
        num_classes: number of classes
        num_nodes: number of nodes
        sigma: some initialization of sigma for constructing the graph
    """
    print(f'-----------{data}-----------')
    file_path =os.path.dirname(__file__)
    if datatype == 'linqs':
        p = "../../datasets/linqs/"
    else:
        p = "../../datasets/flip/"

    data_path = os.path.normpath(os.path.join(file_path, p+data))

    print("Loading labels...")
    true_labels = np.loadtxt(data_path+'/y.csv',delimiter=',')
    true_labels_one_hot = array_to_one_hot(true_labels)

    print("Loading edge features...")
    if datatype == 'linqs':
        if directed:
            if model == 'wrbf':
                edge_features = np.loadtxt(data_path+'/node_features.csv', delimiter=',')
                edge_features = node_features_np_to_dense(edge_features)
            else:
                if confidence == 'raw':
                    edge_features = np.loadtxt(data_path+'/features_raw_reduced.csv',delimiter=',')
                elif confidence == 'raw_reduced':
                    edge_features = np.loadtxt(data_path+'/features_raw.csv',delimiter=',')
                else:
                    print('Weights: Asymmetric')
                    edge_features = np.loadtxt(data_path+'/Easym_normalized_reduced.csv',delimiter=',')
            
        else:
            print('Weights: Symmetric')
            # edge_features = np.loadtxt(data_path+'/Esym_new.csv',delimiter=',')
            edge_features = np.loadtxt(data_path+'/Esym.csv',delimiter=',')
    else:
        edge_features = []

    if datatype == 'linqs':
        node_features = np.loadtxt(data_path+'/Ndir.csv',delimiter=',')
    else:
        node_features = []

    print("Loading graph...")
    graph = np.loadtxt(data_path+'/Gsym.csv',delimiter=',')
    num_nodes = int(max(max(graph[:,0]),max(graph[:,1])) + 1)
    graph_sp = csr_matrix((np.ones((len(graph[:,0]),)), (graph[:,0], graph[:,1])), shape=(num_nodes, num_nodes))

    print('Done!')

    return true_labels_one_hot, edge_features, node_features, graph_sp


def prepare_data(labels,
                 is_labeled,
                 labeled_indices,
                 held_out_indices,
                 true_labels,
                 leave_k,
                 num_samples,
                 seed):
    """Construct masked data and validation data to be fed into the network
    Data will have its labeled nodes masked according to leave_k.
    Input:
        labels: one-hot encoded labels
        is_labeled: boolean indicating if the nodes are labeled
        labeled_indices: indices of labeled nodes
        true_labels: one-hot encoded true labels
        leave_k: number of labeled nodes to mask every epoch
        num_samples: number of samples used to train the network
        seed: seed for randomization
    Returns:
        train_data, validation_data
    """
    np.random.seed(seed)
    num_nodes,num_classes = labels.shape
    # set number of samples
    max_num_samples = sp.special.comb(len(labeled_indices), leave_k, exact=True)

    if max_num_samples < num_samples:
        num_samples = max_num_samples
    # true labels: y
    true_labeled = np.repeat(is_labeled.reshape(num_nodes,1),num_classes,axis=-1)
    true_labeled = true_labeled.reshape(num_nodes,1,num_classes)
    y = np.tile(true_labels,1).reshape(num_nodes,1,num_classes)
    # boolean indicating labeled/masked nodes
    labeled = np.repeat(true_labeled,num_samples,axis=1)
    masked = np.zeros((num_nodes,num_samples,num_classes))

    # TODO: change the shape of this
    if held_out_indices is not None:
        validation_labeled = np.zeros(num_nodes)
        validation_labeled.fill(False)
        validation_labeled.ravel()[held_out_indices] = True
        validation_labeled = validation_labeled.reshape(validation_labeled.shape[0],1)
        validation_labeled = np.repeat(validation_labeled,num_classes,axis=1)
        validation_labeled = validation_labeled.reshape((num_nodes,1,num_classes))
    else:
        validation_labeled = true_labeled


    # construct validation data
    validation_data = {
        'X': labels.reshape(num_nodes,1,num_classes),
        'y': y,
        'labeled': true_labeled,
        'true_labeled': true_labeled, # this will not be used
        'validation_labeled': validation_labeled,
        'masked': masked  # this will not be used
    }

    X = np.tile(labels,num_samples).reshape(num_nodes,num_samples,num_classes)
    if leave_k==1:
        # if leave-1-out sample without replacment
        # such that no indices will be selected twice
        for i,index_to_mask in enumerate(np.random.permutation(labeled_indices)[:num_samples]):
            X[index_to_mask,i,:] = 1/num_classes
            labeled[index_to_mask,i,:] = 0
            masked[index_to_mask,i,:] = 1
    else:
        # otherwise, it is unlikely that same index tuples are selected twice.
        # hence, sample with replacement
        for i in range(num_samples):
            indices_to_mask = np.random.permutation(labeled_indices)[:leave_k]
            X[index_to_mask,i,:] = 1/num_classes
            labeled[index_to_mask,i,:] = 0
            masked[index_to_mask,i,:] = 1

    # construct data used to train the model
    train_data = {
        'X': X,
        'y': y,
        'labeled': labeled,
        'true_labeled': true_labeled,
        'validation_labeled': validation_labeled,
        'masked': masked
    }

    return train_data, validation_data


def random_unlabel(true_labels,unlabel_prob,seed=None,confidence=False):
    """Randomly unlabel nodes based on unlabel probability
    Input:
        true_labels: one-hot encoded true labels
        unlabel_prob: fraction of nodes to be unlabeld
        num_nodes: total number of nodes
        seed: seed for randomization
    Returns:
        labels: one-hot encoded labels where unlabeled nodes are assigned
                uniform probability for each class
        is_labeled: boolean indicating whether each label is labeled/unlabeled
        labeled_indices: indices of labeled nodes
        unlabeled_indices: indices of unlabeled nodes
    """
    num_nodes = true_labels.shape[0]
    np.random.seed(seed)

    num_classes = true_labels.shape[1]
    labeled_indices_from_class = []
    for class_id in range(num_classes):
        labeled_indices_from_class.append(np.random.choice(np.where(true_labels[:,class_id])[0]))
    # sample indices to unlabel
    unlabeled_indices = np.array(sorted(np.random.choice([i for i in range(num_nodes) if i not in labeled_indices_from_class],
                                int(num_nodes * unlabel_prob), replace=False)))
    # unlabeled_indices = np.arange(int(num_nodes * (1-unlabel_prob)),num_nodes)
    labeled_indices = np.delete(np.arange(num_nodes),unlabeled_indices)

    return labeled_indices, unlabeled_indices


def calc_masks(true_labels, labeled_indices, unlabeled_indices, logistic=False, confidence=False):

    labels = true_labels.copy().astype(float)
    num_nodes = len(labeled_indices)+len(unlabeled_indices)
    is_labeled = np.zeros(num_nodes)
    is_labeled.fill(True)
    is_labeled.ravel()[unlabeled_indices] = False
    is_labeled = is_labeled.reshape(is_labeled.shape[0],1)

    # assign uniform probability to unlabled nodes

    if logistic:
        print("Logistic Regression")
        logreg = LogisticRegression(multi_class='multinomial',solver='lbfgs',class_weight='balanced')
        logreg.fit(features[labeled_indices], np.argmax(labels[labeled_indices],axis=1))
        y_pred = logreg.predict_proba(features[unlabeled_indices])
        labels[unlabeled_indices] = y_pred
    else:
        k = labels.shape[1]
        labels[unlabeled_indices] = 1/k


    return labels, is_labeled


def select_features(labeled_indices,features,labels,lasso=False,max_features=200):
    if lasso:
        X=features[labeled_indices]
        y=labels[labeled_indices]
        # Use lasso_path to compute a coefficient path
        _, coef_path, _ = lasso_path(X, y, max_iter=10000)

        coef_path[coef_path < 0.0001] = 0
        coef_path[coef_path >= 0.0001] = 1
        threshold_alpha_id = np.where(np.mean(np.sum(coef_path,axis=1),axis=0) < max_features)[0][-1]
        selected_feature_ids = np.where(coef_path[0][:,threshold_alpha_id])[0]
        print(str(len(selected_feature_ids)),'out of',str(features.shape[1]),'features selected with lasso')
        selected_features = features[:,selected_feature_ids]
    else:
        selected_features = features[:,np.argsort(np.sum(features>0,axis=0))[::-1]][:,:max_features]
        print(selected_features.shape[1],'out of',str(features.shape[1]),'features selected with sorting')
    return selected_features

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