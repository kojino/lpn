from __future__ import print_function
from pathlib import Path
from random import shuffle
import random

import argparse
import copy
import networkx as nx
import numpy as np
import sys
import tensorflow as tf
import os

from deeplp.models.data_prep import create_weighted_graph, load_data
from deeplp.models.data_prep import prepare_data, random_unlabel, calc_masks
from deeplp.models.deeplp_att import DeepLP_ATT
from deeplp.models.deeplp_edge import DeepLP_Edge
from deeplp.models.deeplp_wrbf import DeepLP_WRBF
from deeplp.models.lp import LP
from deeplp.models.utils import accuracy, indices_to_vec
from deeplp.models.data_prep import select_features
from datasets import utils


def approx_chunk(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

def build_and_train_model(args,
                          edge_features,
                          graph,
                          held_out_indices,
                          is_labeled,
                          labels,
                          labeled_indices,
                          log_name,
                          node_features,
                          true_labels,
                          unlabeled_indices):
    tf.reset_default_graph()
    train_data, validation_data = prepare_data(labels,is_labeled,
                                           labeled_indices,held_out_indices,true_labels,
                                           args.leave_k,args.num_samples,
                                           args.split_seed)

    print('Data prepared!')
    sys.stdout.flush()

    lp = LP()
    unlabeled_pred = lp.iter_sp(labels,
                         graph,
                         is_labeled,
                         args.num_iter,
                         unlabeled_indices)
    y_pred = np.argmax(unlabeled_pred,axis=1)
    y_true = np.argmax(true_labels[unlabeled_indices],axis=1)
    accuracy = np.mean(y_pred == y_true)

    print("Baseline:",str(accuracy))


    if args.model == 'att':
        trained_model = DeepLP_ATT(# ATT Specific
                                   args.has_features,
                                   args.change_b,
                                   node_features,
                                   # Edge
                                   args.profile,
                                   'att',
                                   log_name,
                                   edge_features,
                                   graph,
                                   labels.shape[1],
                                   unlabeled_indices,
                                   loss_class_mass=args.loss_class_mass,
                                   confidence=args.confidence,
                                   clamp=args.clamp,
                                   lr=args.lr,
                                   num_iter=args.num_iter,
                                   regularize_theta=2**args.regularize_theta,
                                   regularize_weight=2**args.regularize_weight,
                                   seed=args.parameter_seed,
                                   sparse_edges=args.sparse_edges,
                                   clamp_scale=args.clamp_scale)

    elif args.model == 'edge':
        print(args.regularize_theta)
        trained_model = DeepLP_Edge(args.profile,
                                    'edge',
                                    log_name,
                                    edge_features, # [:,6:] for edge features only
                                    graph,
                                    labels.shape[1],
                                    unlabeled_indices,
                                    confidence=args.confidence,
                                    clamp=args.clamp,
                                    loss_class_mass=args.loss_class_mass,
                                    lr=args.lr,
                                    num_iter=args.num_iter,
                                    regularize_theta=2**args.regularize_theta,
                                    regularize_weight=2**args.regularize_weight,
                                    seed=args.parameter_seed,
                                    sparse_edges=args.sparse_edges,
                                    clamp_scale=args.clamp_scale)
    elif args.model == 'wrbf':
        trained_model = DeepLP_WRBF(args.profile,
                                'wrbf',
                                log_name,
                                edge_features, # [:,6:] for edge features only
                                graph,
                                labels.shape[1],
                                unlabeled_indices,
                                confidence=args.confidence,
                                clamp=args.clamp,
                                loss_class_mass=args.loss_class_mass,
                                lr=args.lr,
                                num_iter=args.num_iter,
                                regularize_theta=2**args.regularize_theta,
                                regularize_weight=2**args.regularize_weight,
                                seed=args.parameter_seed,
                                sparse_edges=args.sparse_edges,
                                clamp_scale=args.clamp_scale)       

    print('Model Built!')
    sys.stdout.flush()

    trained_model.train(train_data,validation_data,args.num_epoch)
    return trained_model


def main(args):
    if args.datatype == 'flip':
        if args.data == 'cora':
            args.num_iter = 20
            args.num_epoch = 5000
        elif args.data == 'dblp':
            args.num_iter = 20
            args.num_epoch = 1200
        elif args.data == 'flickr':
            args.num_iter = 10
            args.num_epoch = 500
        elif args.data == 'imdb':
            args.num_iter = 10
            args.num_epoch = 10000
        elif args.data == 'industry':
            args.num_iter = 10
            args.num_epoch = 10000

    if args.datatype == 'linqs':
        if args.data == 'cora':
            args.num_iter = 20
            args.num_epoch = 5000
        elif args.data == 'citeseer':
            args.num_iter = 50
            args.num_epoch = 5000
        elif args.data == 'pubmed':
            args.num_iter = 20
            args.num_epoch = 2000

    d = {
        0:-12,
        1:-6,
        2:-8,
        3:-6,
        4:-4,
        5:-12,
        6:4,
        7:-14,
        8:-30,
        9:-10
    }
    args.regularize_theta = d[args.split_seed]


    if args.crossval_k != 1:
        args.num_epoch = int(args.num_epoch / args.crossval_k)

    if not os.path.exists('accs'):
        os.makedirs('accs')

    print(args.datatype,args.data,args.num_iter,args.num_epoch)

    args_dict = vars(args)
    file_dict = { key: args_dict[key] for key in ['data',
                                                  'regularize_theta',
                                                  'split_seed']}

    # file_dict = { key: args_dict[key] for key in ['clamp',
    #                                               'data',
    #                                               'regularize',
    #                                               'split_seed',
    #                                               'confidence']}
    print("Paramaters changed:")
    print(file_dict)

    log_name = ':'.join(['{}={}'.format(k,v) for k,v in file_dict.items()])
    print("File Name")
    print(log_name)

    true_labels, edge_features, node_features, graph \
    = load_data(args.data,args.datatype,directed=args.asymmetric,confidence=args.confidence,model=args.model)
    U,D,B,R = utils.load_data(args.data,args.datatype,'datasets')
    edges = np.array(B.edges())
    sources,sinks = edges[:,0],edges[:,1]

    labeled_indices, unlabeled_indices = \
        random_unlabel(true_labels,args.unlabel_prob,
                       seed=args.split_seed,confidence=args.confidence)

    num_nodes, num_classes = true_labels.shape

    labels, is_labeled = calc_masks(true_labels, labeled_indices, unlabeled_indices, logistic=args.logistic, confidence=args.confidence)

    if args.model != 'wrbf':
        print("Generating Seed Dependent Features...")
        seed_to_node_lengths = []
        for i in labeled_indices:
            shortest_paths_seed = nx.shortest_path_length(B,source=int(i))
            path_lengths = [i[1] for i in sorted(shortest_paths_seed.items())]
            seed_to_node_lengths.append(path_lengths)
        seed_to_node_lengths = np.array(seed_to_node_lengths)
        labels_for_seeds = np.argmax(true_labels[labeled_indices],axis=1)
        labels_for_seeds_dict = {}
        for i,label in enumerate(labels_for_seeds):
            if label in labels_for_seeds_dict:
                labels_for_seeds_dict[label].append(i)
            else:
                labels_for_seeds_dict[label] = [i]

        seed_features = []
        for label in labels_for_seeds_dict:
            indices = labels_for_seeds_dict[label]
            label_seed_to_node_lengths = seed_to_node_lengths[indices]
            label_min_len_to_seed = np.min(label_seed_to_node_lengths,axis=0)[sources]
            label_mean_len_to_seed = np.mean(label_seed_to_node_lengths,axis=0)[sources]
            seed_features.append(label_min_len_to_seed)
            seed_features.append(label_mean_len_to_seed)

        min_len_to_seed = np.min(seed_to_node_lengths,axis=0)[sources]
        mean_len_to_seed = np.mean(seed_to_node_lengths,axis=0)[sources]
        seed_features.append(min_len_to_seed)
        seed_features.append(mean_len_to_seed)
        seed_features = np.array(seed_features).T
        seed_features = utils.pos_normalize(seed_features)


        print("Seed Dependent Features Done!")
        # print(edge_features.shape)


        if args.datatype == 'linqs':
            edge_fetures = np.hstack([edge_features,seed_features])
        else:
            edge_fetures = seed_features


    if args.crossval_k == 1:

        trained_model = build_and_train_model(args,
                                              edge_features,
                                              graph,
                                              None,
                                              is_labeled,
                                              labels,
                                              labeled_indices,
                                              log_name,
                                              node_features,
                                              true_labels,
                                              unlabeled_indices)


        # print outputs
        headers = [arg for arg in vars(args)]
        parameters = [getattr(args, arg) for arg in vars(args)]

        headers += ['org_leave_k_out_losses', 'opt_leave_k_out_losses',
                    'org_unlabeled_loss', 'opt_unlabeled_loss',
                    'org_accuracy', 'opt_accuracy', 'entropy']
        metrics = [trained_model.leave_k_out_losses[0],
                   trained_model.leave_k_out_losses[-1],
                   trained_model.unlabeled_losses[0],
                   trained_model.unlabeled_losses[-1],
                   trained_model.true_accuracies[0],
                   trained_model.true_accuracies[-1],
                   trained_model.entropies[-1]]

        outputs = np.hstack([parameters,metrics])
        for name, index in [('++initial',0),
                            ('++min_entropy:',np.argmin(trained_model.entropies)),
                            ('++min_leave_k_loss',np.argmin(trained_model.leave_k_out_losses)),
                            ('++max_accuracy',np.argmax(trained_model.accuracies)),
                            ('++last',-1)]:
            print(name,args.data,str(args.split_seed),str(args.regularize_theta),
                             trained_model.leave_k_out_losses[index],
                             trained_model.objectives[index],
                             trained_model.unlabeled_losses[index],
                             trained_model.true_accuracies[index],
                             trained_model.entropies[index],
                             trained_model.as_[index],
                             trained_model.bs_[index],
                             trained_model.regularize_vals[index])

    else:
        final_accs = []

        labeled_indices_copy = copy.copy(labeled_indices)
        random.seed(args.split_seed)
        shuffle(labeled_indices_copy)
        cv_held_out_indices_list = approx_chunk(labeled_indices_copy, args.crossval_k)


        for i, cv_held_out_indices in enumerate(cv_held_out_indices_list):
            print(f"{i}th cross validation")
            cv_labeled_indices = [index for index in labeled_indices if index not in cv_held_out_indices]
            cv_unlabeled_indices = np.delete(np.arange(true_labels.shape[0]),cv_labeled_indices)
            cv_labels, cv_is_labeled = calc_masks(true_labels, cv_labeled_indices, cv_unlabeled_indices,logistic=0)
            log_name_new = f"{log_name}:this_crossval_k={i}"
            print('---------------')
            print(log_name_new)

            trained_model = build_and_train_model(args,
                                                  edge_features,
                                                  graph,
                                                  cv_held_out_indices,
                                                  cv_is_labeled,
                                                  cv_labels,
                                                  cv_labeled_indices,
                                                  log_name_new,
                                                  node_features,
                                                  true_labels,
                                                  cv_unlabeled_indices)

            final_acc = trained_model.validation_accuracies[-1]
            final_accs.append(final_acc)

        average_acc = np.mean(final_accs)
        print('++average_acc',args.data,str(args.split_seed),str(args.regularize_theta),str(average_acc))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--change_b', default=0, type=float,
                        help='if 1, make b of att a variable')

    parser.add_argument('--clamp', default=0, type=int,
                        help='if 1, clamp the labels every layer')

    parser.add_argument('--crossval_k', default=5, type=int,
                        help='number of folds in crossvalidation for \
                              choosing the optimal regularization')

    parser.add_argument('--data', default='linqs/citeseer',
                        help='data to run deeplp')

    parser.add_argument('--has_features', default=0, type=int,
                        help='for att: hidden values have features')

    parser.add_argument('--leave_k', default=5, type=int,
                        help='number of labeled nodes to mask')

    parser.add_argument('--logistic', default=0, type=int,
                        help='if 1, start initial unlabel predictions as \
                              logisitc predictions')

    parser.add_argument('--loss_class_mass', default=1, type=int,
                        help='if 1, apply weighted loss; losses on minority labels \
                              are penalized more')

    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')

    parser.add_argument('--model', default='edge',
                        help='model for propagation')

    parser.add_argument('--num_epoch', default=10, type=int,
                        help='number of epochs to run the model')

    parser.add_argument('--num_iter', default=10, type=int,
                        help='number of hidden layers of the network')

    parser.add_argument('--num_samples', default=100, type=int,
                        help='number of samples to train the model')

    parser.add_argument('--parameter_seed', default=1, type=int,
                        help='random seed for parameter initialization')

    parser.add_argument('--regularize_theta', default=-3, type=float,
                        help='regularization parameter, 2**X')

    parser.add_argument('--regularize_type', default='l1',
                        help='norm of regularization')

    parser.add_argument('--split_seed', default=1, type=int,
                        help='random seed for labeled/unlabeled split')

    parser.add_argument('--asymmetric', default=1, type=int,
                        help='if 1, use asymmetric graph')

    parser.add_argument('--unlabel_prob', default=0.99, type=float,
                        help='fraction of unlabeled nodes')

    parser.add_argument('--weight_normalization', default='softmax', type=str,
                        help='method for normalizing the weight matrix')

    parser.add_argument('--confidence', type=str,
                        help='whether to use unknown class for att')

    parser.add_argument('--datatype', default='linqs', type=str,
                        help='linqs or flip')

    parser.add_argument('--profile', default=1, type=int,
                        help='profile the runtime')

    parser.add_argument('--sparse_edges', default=0, type=int,
                        help='sparsify edges')

    parser.add_argument('--regularize_weight', default=0, type=float,
                        help='sparsify edges')

    parser.add_argument('--clamp_scale', default=0.1, type=float,
                        help='sparsify edges')


    args = parser.parse_args()
    print(args)
    main(args)
