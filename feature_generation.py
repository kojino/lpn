import pandas as pd
import numpy as np
import networkx as nx
import scipy as sp

from deeplp.models.data_prep import create_weighted_graph, load_data
from deeplp.models.data_prep import prepare_data, random_unlabel, calc_masks
from deeplp.models.utils import accuracy, indices_to_vec
from deeplp.models.data_prep import select_features

from datasets import utils
from datasets.utils import listify, lisify_links, rbf, one_hot, get_u_to_b_indices, get_d_to_b_indices, get_r_to_b_indices
from datasets.utils import node_feature_similarities, node_feature_reduction, node_centralities, node_partitions, random_walk_features, node_others, edge_centralities, link_predictions, community_features

data = 'cora'
true_labels, features, edge_features, node_features, \
graph, weights = load_data(data,directed=1)
U,D,B,R,node_features = utils.load_data(data,'datasets/')
edges = np.array(B.edges())
sources,sinks = edges[:,0],edges[:,1]
