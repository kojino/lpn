import argparse
from collections import defaultdict as dd
import cPickle
import numpy as np
import os
from scipy import sparse as sp

def add_arguments():
    parser = argparse.ArgumentParser()
    # General settings
    parser.add_argument('--x', help = 'train features filename' , type = str)
    parser.add_argument('--tx', help = 'test features filename' , type = str)
    parser.add_argument('--y', help = 'train labels filename' , type = str)
    parser.add_argument('--ty', help = 'test labels filename' , type = str)
    parser.add_argument('--graph', help = 'graph filename' , type = str)
    parser.add_argument('--out_dir', help = 'dir where output files will be stored' , type = str)
    parser.add_argument('--dim', help = 'feature dimension' , type = int)
    parser.add_argument('--c', help = 'num classes' , type = int)
    return parser

def read_graph(path_to_grpah):
	graph_dict = dd(list)
	with open(path_to_grpah, "r") as graph_file:
		for line in graph_file:
			split_line = line.split('\t')
			if (len(split_line) != 2):
				print "Problematic line:", line
				continue
			src = int(split_line[0])
			dst = int(split_line[1])
			graph_dict[src].append(dst)
			graph_dict[dst].append(src)
	for node_id in graph_dict:
		graph_dict[node_id] = list(set(graph_dict[node_id]))
	return graph_dict

def read_labels(path_to_labels, num_classes):
	node_to_cluster = {}
	with open(path_to_labels) as f_labels:
		for line in f_labels:
			split_line = line.split('\t')
			node_id = int(split_line[0])
			cluster = int(split_line[1])
			node_to_cluster[node_id] = cluster
	max_node_id = max(node_to_cluster.keys())
	max_cluster = max(node_to_cluster.values())
	assert num_classes >= max_cluster+1
	label_matrix = np.zeros((max_node_id+1, num_classes))
	for node_id, cluster in node_to_cluster.iteritems():
		label_matrix[node_id, cluster] = 1.0
	label_matrix = label_matrix.astype(np.int32)
	return label_matrix

def read_features(path_to_features, dim):
	node_to_features = dd(list)
	max_feature_index = 0
	with open(path_to_features) as f_features:
		for line in f_features:
			split_line = line.split('\t')
			node_id = int(split_line[0])
			for features in split_line[1:]:
				split_features = features.split(":")
				feature_index = int(split_features[0])
				feature_value = float(split_features[1])
				max_feature_index = max(max_feature_index, feature_index)
				feature_details = (feature_index, feature_value)
				node_to_features[node_id].append(feature_details)
	max_node_id = max(node_to_features.keys())
	assert dim >= max_feature_index
	feature_matrix = np.zeros((max_node_id+1, dim))
	for node_id in node_to_features:
		for feature_detail in node_to_features[node_id]:
			(feature_index, feature_value) = feature_detail
			feature_matrix[node_id, feature_index] = feature_value
	feature_matrix = sp.csr_matrix(feature_matrix, dtype = np.float32)
	return feature_matrix

def main():
	parser 	=  add_arguments()
	args   	=  parser.parse_args()
	graph_dict = read_graph(args.graph)
	out_graph = os.path.join(args.out_dir, "graph")
	cPickle.dump(graph_dict, open(out_graph, "w"))

	x = read_features(args.x, args.dim)
	out_x = os.path.join(args.out_dir, "x")
	cPickle.dump(x, open(out_x, "w"))
	
	tx = read_features(args.tx, args.dim)
	out_tx = os.path.join(args.out_dir, "tx")
	cPickle.dump(tx, open(out_tx, "w"))

	y = read_labels(args.y, args.c)
	out_y = os.path.join(args.out_dir, "y")
	cPickle.dump(y, open(out_y, "w"))

	ty = read_labels(args.ty, args.c)
	out_ty = os.path.join(args.out_dir, "ty")
	cPickle.dump(ty, open(out_ty, "w"))	

if __name__ == '__main__':
	main()
