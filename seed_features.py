import pandas as pd
import numpy as np
import networkx as nx
import scipy as sp

from deeplp.models.data_prep import create_weighted_graph, load_data
from deeplp.models.data_prep import prepare_data, random_unlabel, calc_masks
from deeplp.models.utils import accuracy, indices_to_vec
from deeplp.models.data_prep import select_features

from datasets import utils

import argparse

def main(args):
    data = args.data
    source = args.source
    seed = args.seed
    true_labels, features, edge_features, node_features, \
    graph, weights = load_data(data,directed=1)
    U,D,B,R,node_features = utils.load_data(data,'datasets/')
    edges = np.array(B.edges())
    sources,sinks = edges[:,0],edges[:,1]

    labeled_indices, unlabeled_indices = \
        random_unlabel(true_labels,0.99,features,
                       seed=seed)

    num_nodes, num_classes = true_labels.shape

    labels, is_labeled = calc_masks(true_labels, labeled_indices, unlabeled_indices, logistic=0)

    # print("Shortest Paths")
    # shortest_paths = np.ones(num_nodes) * np.inf
    # num_shortest_paths = np.ones(num_nodes) * np.inf
    # shortest_paths_seed = nx.shortest_path_length(B,source=source)
    # for target in range(num_nodes):
    #     shortest_path = shortest_paths_seed[target]
    #     shortest_paths[target] = shortest_path
    #     num_shortest_path = len(list(nx.all_shortest_paths(B, source, target, weight=None)))
    #     num_shortest_paths[target] = num_shortest_path

    print("Flows and Connectivity")
    for edge in B.edges:
        B.edges[edge[0],edge[1]]['capacity'] = 1
    maximum_flow = np.ones(num_nodes) * np.inf
    local_edge_connectivity = np.ones(num_nodes) * np.inf
    for target in range(num_nodes):
        if target % 100 == 0:
            print(target)
        if target == source:
            maximum_flow[target] = 0
            local_edge_connectivity[target] = 0
        else:
            maximum_flow[target] = nx.maximum_flow(B, source, target)[0]
            local_edge_connectivity[target] = nx.algorithms.connectivity.local_edge_connectivity(B, source, target)

    print("Conductance and Cut Size")
    conductance = np.ones(num_nodes) * np.inf
    cut_size = np.ones(num_nodes) * np.inf
    normalized_cut_size = np.ones(num_nodes) * np.inf
    for node in range(num_nodes):
        if node % 100 == 0:
            print(node)
        conductance[node] = nx.conductance(B, labeled_indices, [node])
        cut_size[node] = nx.cut_size(B, labeled_indices, [node])
        normalized_cut_size[node] = nx.normalized_cut_size(B, labeled_indices, [node])

    print("Saving...")
    neighbors = np.array(nx.to_numpy_matrix(B)[source])[0]
    conductance_sink_norm = conductance * neighbors / np.sum(conductance * neighbors)
    cut_size_sink_norm = cut_size * neighbors / np.sum(cut_size * neighbors)
    normalized_cut_size_sink_norm = normalized_cut_size * neighbors / np.sum(normalized_cut_size * neighbors)
    seed_features = np.vstack((shortest_paths,num_shortest_paths,maximum_flow,local_edge_connectivity,conductance_sink_norm,cut_size_sink_norm,normalized_cut_size_sink_norm)).T

    np.savetxt(f'seed_features/{data}/seed_features_{seed}_{source}.csv',seed_features,delimiter=',')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='cora', type=str)

    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--source', default=1, type=int)

    args = parser.parse_args()
    print(args)
    main(args)
