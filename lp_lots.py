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


true_labels, features, edge_features, node_features, \
graph, weights = load_data('citeseer',1)
weights_one = weights > 0
for i in range(100):
    labeled_indices, unlabeled_indices = \
        random_unlabel(true_labels,0.99,features,seed=i)
    labels, is_labeled = calc_masks(true_labels, labeled_indices, unlabeled_indices, logistic=0)

    num_classes = labels.shape[1]

    weights_one = weights > 0

    lp = LP()
    accs = []
    for weight_id,weights_np in enumerate([weights, weights_one]):
        unlabeled_pred = lp.closed_sp(labels,
                                   weights_np,
                                   labeled_indices,
                                   unlabeled_indices)
        print(unlabeled_pred.shape,np.sum(unlabeled_pred,axis=1))
        accs.append(np.mean(np.argmax(true_labels[unlabeled_indices],axis=1) == np.argmax(unlabeled_pred,axis=1)))
        unlabeled_pred = lp.iter_sp(labels,
                             weights_np,
                             is_labeled,
                             100,
                             unlabeled_indices)
        print(unlabeled_pred.shape,np.sum(unlabeled_pred,axis=1))
        accs.append(np.mean(np.argmax(true_labels[unlabeled_indices],axis=1) == np.argmax(unlabeled_pred,axis=1)))
    print("Seed:", '%04d' % (i + 1),
          "closed_RBF=", "{:.5f}".format(accs[0]),
          "closed_one=", "{:.5f}".format(accs[2]),
          "iter_RBF=", "{:.5f}".format(accs[1]),
"iter_one=", "{:.5f}".format(accs[3]))
