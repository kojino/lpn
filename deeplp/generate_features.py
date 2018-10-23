import argparse
import logging

import networkx as nx
import numpy as np

from deeplp.utils import (
    community_features, edge_centralities, get_d_to_b_indices,
    get_r_to_b_indices, get_u_to_b_indices, link_predictions, load_graph,
    node_centralities, node_feature_reduction, node_feature_similarities,
    node_partitions, pos_normalize)

logger = logging.getLogger("deeplp")
logging.basicConfig(
    format='%(asctime)s: %(message)s',
    level="INFO",
    datefmt='%m/%d/%Y %I:%M:%S %p')

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data', default='linqs_cora', help='data to run deeplp on')

parser.add_argument(
    '--raw_only',
    default=0,
    type=int,
    help='if True, only use the dimesion reduced version of the raw features')

args = parser.parse_args()

U, D, B, R, node_features = load_graph(args.data)

nx.write_edgelist(
    B, f'data/{args.data}/graph_symmetric.csv', delimiter=',', data=False)
# get list of edges
edges = np.array(B.edges())
sources, sinks = edges[:, 0], edges[:, 1]

# # np.savetxt(node_features[sources],

# if args.raw_only:
#     node_feature_reduction = node_feature_reduction(
#         node_features, sources, n_components=10)
#     features = node_feature_reduction
# else:
#     u_to_b_indices = get_u_to_b_indices(U, edges)
#     d_to_b_indices = get_d_to_b_indices(D, edges)
#     r_to_b_indices = get_r_to_b_indices(R, edges)

#     logger.info("Feature generation start.")
#     node_feature_similarities = node_feature_similarities(
#         node_features, sources, sinks)
#     node_feature_reduction = node_feature_reduction(node_features, sources)
#     node_centralities = node_centralities(B, D, R, U, sources, sinks)
#     node_partitions = node_partitions(U, sources, sinks)
#     edge_centralities = edge_centralities(B, D, R, U, edges, d_to_b_indices,
#                                           r_to_b_indices, u_to_b_indices)
#     link_predictions = link_predictions(U, edges)
#     community_features = community_features(U, node_features, edges, sinks,
#                                             sources)
#     features = np.hstack([
#         node_feature_similarities, node_feature_reduction, node_centralities,
#         node_partitions, edge_centralities, link_predictions,
#         community_features
#     ])

# normalized_features = pos_normalize(features)
# logger.info("Feature generation done.")
# if args.raw_only:
#     feature_fname = f'data/{args.data}/features_raw_reduced.csv'
# else:
#     feature_fname = f'data/{args.data}/features.csv'
# np.savetxt(feature_fname, normalized_features, delimiter=',')
