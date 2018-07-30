from __future__ import print_function
from pathlib import Path
from sklearn.metrics import f1_score

import argparse
import csv
import numpy as np
import pandas as pd
import sys
import tensorflow as tf
import time

from deeplp.models.data_prep import create_weighted_graph
from deeplp.models.data_prep import load_data
from deeplp.models.data_prep import prepare_data, calc_masks
from deeplp.models.data_prep import random_unlabel
from deeplp.models.lp import LP
from deeplp.models.utils import accuracy
from deeplp.models.utils import indices_to_vec
from deeplp.models.utils import array_to_one_hot
from deeplp.models.data_prep import select_features


def main(args):
    true_labels, edge_features, node_features, weights \
    = load_data(args.data,'linqs',directed=1)

    weights_one = weights > 0

    labeled_indices, unlabeled_indices = \
        random_unlabel(true_labels,args.unlabel_prob,
                       seed=args.split_seed)

    num_nodes, num_classes = true_labels.shape

    labels, is_labeled = calc_masks(true_labels, labeled_indices,
                                    unlabeled_indices)

    weights_one = weights > 0

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
        'accuracy': [],
        'f1_micro': [],
        'f1_macro': []
    })

    def save_result(results, unlabeled_pred, model, weight_id, iter_id, clamp,
                    laplacian):
        # print('-------------------')
        # print(np.sum(unlabeled_pred,axis=1))
        y_pred = np.argmax(unlabeled_pred, axis=1)
        y_true = np.argmax(true_labels[unlabeled_indices], axis=1)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        accuracy = np.mean(y_pred == y_true)

        results = results.append(
            {
                'data': args.data,
                'model': model,
                'split_seed': args.split_seed,
                'unlabel_prob': args.unlabel_prob,
                'weights': weight_id,
                'iter': iter_id,
                'clamp': clamp,
                'laplacian': laplacian,
                'accuracy': accuracy,
                'f1_micro': f1_micro,
                'f1_macro': f1_macro
            },
            ignore_index=True)
        return results

    print("LP closed form")
    lp = LP()
    for weight_id, weights_np in enumerate([weights, weights_one]):
        unlabeled_pred = lp.closed_sp(labels, weights_np, labeled_indices,
                                      unlabeled_indices)
        results = save_result(results, unlabeled_pred, 'LP closed', weight_id,
                              None, None, None)

    print("LP iterative form")
    for iter_id in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        print(iter_id)
        for weight_id, weights_np in enumerate([weights, weights_one]):
            print(weight_id)

            unlabeled_pred = lp.iter_sp(labels, weights_np, is_labeled,
                                        iter_id, unlabeled_indices)
            results = save_result(results, unlabeled_pred, 'LP Iter',
                                  weight_id, iter_id, None, None)
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
    # if args.data[:5] == 'linqs':
    #     print("Logistic Regression")
    #     from sklearn.linear_model import LogisticRegression
    #     from sklearn import metrics
    #     print(features[labeled_indices].shape,
    #           np.argmax(labels[labeled_indices], axis=1).shape)

    #     logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    #     logreg.fit(features[labeled_indices],
    #                np.argmax(labels[labeled_indices], axis=1))
    #     y_pred = logreg.predict(features[unlabeled_indices])
    #     num_nodes = len(y_pred)
    #     num_classes = labels.shape[1]
    #     res = np.zeros((num_nodes, num_classes))
    #     res[np.arange(num_nodes), y_pred.astype(int)] = 1
    #     results = save_result(results, res, 'Logistic Regression', None, None,
    #                           None, None)

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

    # print("Adsorption")
    # h = labels
    # Y = np.hstack([h, np.zeros((h.shape[0], 1))])
    # r = np.hstack([np.zeros(h.shape[1]), np.ones(1)])
    # entropy = weights.copy()
    # entropy.data = (-weights.data * np.log(weights.data + 0.000001))
    # entropy = np.squeeze(np.array(entropy.sum(axis=0)))

    # def f(x):
    #     return np.log(2) / np.log(2 + np.exp(x))

    # f = np.vectorize(f)
    # c = f(entropy)
    # d = (1 - c) * np.sqrt(entropy)
    # d[unlabeled_indices] = 0
    # z = np.maximum(c + d, np.ones(c.shape[0]))

    # p_cont = np.expand_dims(c / z, axis=1)
    # p_inj = np.expand_dims(d / z, axis=1)
    # p_abnd = 1 - p_cont - p_inj

    # Yhat = Y.copy()
    # for i in range(100):
    #     # propagate labels
    #     D = np.array(weights @ Yhat / np.sum(weights, axis=1))
    #     # don't update labeled nodes
    #     Yhat = p_inj * Y + p_cont * D + p_abnd * r

    # results = save_result(results, Yhat[unlabeled_indices][:, :-1],
    #                       'Adsorption', None, None, None, None)

    for index, row in results.iterrows():
        print('++', ','.join(map(str, row)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data', default='linqs/cora', help='data to run deeplp')

    parser.add_argument(
        '--split_seed',
        default=1,
        type=int,
        help='random seed for labeled/unlabeled split')

    parser.add_argument(
        '--unlabel_prob',
        default=0.9,
        type=float,
        help='fraction of unlabeled nodes')

    args = parser.parse_args()
    print(args)
    main(args)
