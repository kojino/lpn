import argparse
import copy
import datetime
import logging
import os
import random
import pandas as pd
import sys
from random import shuffle
import math
import networkx as nx
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from deeplp.models.deeplp_att import DeepLP_ATT
from deeplp.models.deeplp_edge import DeepLP_Edge
from deeplp.models.deeplp_wrbf import DeepLP_WRBF
from deeplp.models.lp import LP
from deeplp.utils import (calc_masks, create_seed_features, load_data,
                          num_layers_dict, prepare_data, random_unlabel,
                          create_weighted_graph)

# from deeplp.models.data_prep import select_features


def main():
    results_master = []
    for data in ['linqs_cora', 'linqs_citeseer', 'linqs_pubmed']:
        true_labels, features, weights_one, raw_features = load_data(
            'linqs_cora', model='edge', feature_type='all')
        weights, _, _ = create_weighted_graph(raw_features, weights_one)
        l = [0.99, 0.98, 0.97, 0.96, 0.95]
        if data == 'linqs_pubmed':
            l = [0.9975, 0.995, 0.9925, 0.9]
        for unlabel_prob in l:
            for split_seed in range(10):
                print(data, unlabel_prob, split_seed)

                # for seed in range(100):
                seed = split_seed
                labeled_indices, unlabeled_indices = \
                    random_unlabel(true_labels, unlabel_prob, seed)

                num_nodes, num_classes = true_labels.shape
                labels, is_labeled = calc_masks(
                    true_labels,
                    labeled_indices,
                    unlabeled_indices,
                    None,
                    logistic=0)

                y_true = np.argmax(true_labels[unlabeled_indices], axis=1)

                results = pd.DataFrame({
                    'data': [],
                    'model': [],
                    'split_seed': [],
                    'unlabel_prob': [],
                    'weights': [],
                    'iter': [],
                    'clamp': [],
                    'laplacian': [],
                    'accuracy': []
                })

                def save_result(results, unlabeled_pred, model, weight_id,
                                iter_id, clamp, laplacian):
                    # print('-------------------')
                    # print(np.sum(unlabeled_pred,axis=1))
                    y_pred = np.argmax(unlabeled_pred, axis=1)
                    y_true = np.argmax(true_labels[unlabeled_indices], axis=1)
                    accuracy = np.mean(y_pred == y_true)

                    results = results.append(
                        {
                            'data': data,
                            'model': model,
                            'split_seed': split_seed,
                            'unlabel_prob': unlabel_prob,
                            'weights': weight_id,
                            'iter': iter_id,
                            'clamp': clamp,
                            'laplacian': laplacian,
                            'accuracy': accuracy,
                        },
                        ignore_index=True)
                    return results

                print("LP closed form")
                lp = LP()
                for weight_id, weights_np in enumerate([weights, weights_one]):
                    unlabeled_pred = lp.closed_sp(
                        labels, weights_np, labeled_indices, unlabeled_indices)
                    results = save_result(results, unlabeled_pred, 'LP closed',
                                          weight_id, None, None, None)

                print("LP iterative form")
                for weight_id, weights_np in enumerate([weights, weights_one]):

                    unlabeled_pred = lp.iter_sp(labels, weights_np, is_labeled,
                                                split_seed, unlabeled_indices)
                    results = save_result(results, unlabeled_pred, 'LP Iter',
                                          weight_id, split_seed, None, None)
                    # print('clamp')
                    # for clamp in [num_classes/2,num_classes,2*num_classes]:
                    #             unlabeled_pred = lp.iter_sp(labels,
                    #                                  weights_np,
                    #                                  is_labeled,
                    #                                  iter_id,
                    #                                  unlabeled_indices,
                    #                                  clamp=clamp)
                    #             results = save_result(results,unlabeled_pred,'LP Iter Clamp',weight_id,iter_id,clamp,None)
                    # print('laplacian')
                    # for laplacian in [0.01, 0.05, 0.1, 0.2]:
                    #     unlabeled_pred = lp.iter_sp(labels,
                    #                              weights_np,
                    #                              is_labeled,
                    #                              iter_id,
                    #                              unlabeled_indices,
                    #                              laplacian=laplacian)
                    #     results = save_result(results,unlabeled_pred,'LP Iter Laplacian',weight_id,iter_id,None,laplacian)
                if data[:5] == 'linqs':
                    print("Logistic Regression")
                    from sklearn.linear_model import LogisticRegression
                    from sklearn import metrics

                    logreg = LogisticRegression(
                        multi_class='multinomial', solver='lbfgs')
                    logreg.fit(features[labeled_indices],
                               np.argmax(labels[labeled_indices], axis=1))
                    y_pred = logreg.predict(features[unlabeled_indices])
                    num_nodes = len(y_pred)
                    num_classes = labels.shape[1]
                    res = np.zeros((num_nodes, num_classes))
                    res[np.arange(num_nodes), y_pred.astype(int)] = 1
                    results = save_result(results, res, 'Logistic Regression',
                                          None, None, None, None)

                # print("Local Global Consistency")
                # import scipy as sp
                # W = weights.copy()
                # D = sp.sparse.diags([np.array(weights.sum(axis=1).T)[0]], [0])
                # Dinv = sp.sparse.linalg.inv(D.tocsc())
                # Dinv_sqrt = np.sqrt(Dinv)
                # S = Dinv_sqrt @ weights @ Dinv_sqrt

                # alpha = 0.99
                # h = labels
                # for i in range(100):
                #     h = alpha * S @ h + (1 - alpha) * labels

                # results = save_result(results, h[unlabeled_indices], 'LLGC', None, None,
                #                       None, None)

                print("Adsorption")
                h = labels
                Y = np.hstack([h, np.zeros((h.shape[0], 1))])
                r = np.hstack([np.zeros(h.shape[1]), np.ones(1)])
                entropy = weights.copy()
                entropy.data = (
                    -weights.data * np.log(weights.data + 0.000001))
                entropy = np.squeeze(np.array(entropy.sum(axis=0)))

                def f(x):
                    return np.log(2) / np.log(2 + np.exp(x))

                f = np.vectorize(f)
                c = f(entropy)
                d = (1 - c) * np.sqrt(entropy)
                d[unlabeled_indices] = 0
                z = np.maximum(c + d, np.ones(c.shape[0]))

                p_cont = np.expand_dims(c / z, axis=1)
                p_inj = np.expand_dims(d / z, axis=1)
                p_abnd = 1 - p_cont - p_inj

                Yhat = Y.copy()
                for i in range(100):
                    # propagate labels
                    D = np.array(weights @ Yhat / np.sum(weights, axis=1))
                    # don't update labeled nodes
                    Yhat = p_inj * Y + p_cont * D + p_abnd * r

                results = save_result(results, Yhat[unlabeled_indices][:, :-1],
                                      'Adsorption', None, None, None, None)
                results_master.append(results)
    pd.DataFrame(results_master).to_csv('baselines.csv')


if __name__ == '__main__':

    # parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     '--data', default='linqs/cora', help='data to run deeplp')

    # parser.add_argument(
    #     '--split_seed',
    #     default=1,
    #     type=int,
    #     help='random seed for labeled/unlabeled split')

    # parser.add_argument(
    #     '--unlabel_prob',
    #     default=0.9,
    #     type=float,
    #     help='fraction of unlabeled nodes')

    # args = parser.parse_args()
    # # print(args)
    main()
