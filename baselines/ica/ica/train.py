"""" This implementation is largely based on and adapted from:
 https://github.com/sskhandle/Iterative-Classification """
from ica.utils import load_data, load_data_2, pick_aggregator, create_map, build_graph
from ica.classifiers import LocalClassifier, RelationalClassifier, ICA

from scipy.stats import sem

from sklearn.metrics import accuracy_score
import numpy as np
import argparse
import time
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', default='cora', help='Dataset string.')
parser.add_argument(
    '-classifier',
    default='sklearn.linear_model.LogisticRegression',
    help='Underlying classifier.')
parser.add_argument('-seed', type=int, default=42, help='Random seed.')
parser.add_argument(
    '-num_trials', type=int, default=10, help='Number of trials.')
parser.add_argument(
    '-max_iteration',
    type=int,
    default=10,
    help='Number of iterations (iterative classification).')
parser.add_argument(
    '-aggregate',
    choices=['count', 'prop'],
    default='count',
    help='Aggregation operator.')
parser.add_argument(
    '-bootstrap',
    default=True,
    action='store_true',
    help=
    'Bootstrap relational classifier training with local classifier predictions.'
)
parser.add_argument(
    '-validation',
    default=False,
    action='store_true',
    help='Whether to test on validation set (True) or test set (False).')

args = parser.parse_args()
np.random.seed(args.seed)
results_master = []
# load data
# adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset)
for data_path in ['cora', 'citeseer', 'pubmed']:
    data_path = 'linqs_data/' + data_path
    l = [0.99, 0.98, 0.97, 0.96, 0.95]
    if data_path == 'linqs_data/pubmed':
        l = [0.9975, 0.995, 0.9925, 0.9]
    for unlabel_prob in l:
        for seed in range(10):
            print(data_path, unlabel_prob, seed)
            np.random.seed(seed)
            adj, features, labels, idx_train, idx_test = load_data_2(
                data_path, seed, unlabel_prob)

            graph, domain_labels = build_graph(adj, features, labels)

            # train / test splits
            train = idx_train
            test = idx_test
            eval_idx = np.setdiff1d(list(range(adj.shape[0])), idx_train)

            # run training
            ica_accuracies = list()
            for run in range(args.num_trials):

                t_begin = time.time()

                # random ordering
                np.random.shuffle(eval_idx)

                y_true = [graph.node_list[t].label for t in test]
                local_clf = LocalClassifier(args.classifier)
                agg = pick_aggregator(args.aggregate, domain_labels)
                relational_clf = RelationalClassifier(args.classifier, agg)
                ica = ICA(
                    local_clf,
                    relational_clf,
                    args.bootstrap,
                    max_iteration=args.max_iteration)
                ica.fit(graph, train)
                conditional_node_to_label_map = create_map(graph, train)
                ica_predict = ica.predict(graph, eval_idx, test,
                                          conditional_node_to_label_map)
                ica_accuracy = accuracy_score(y_true, ica_predict)
                ica_accuracies.append(ica_accuracy)
                # print('Run ' + str(run) + ': \t\t' + str(ica_accuracy) + ', Elapsed time: \t\t' + str(time.time() - t_begin))

            # print(("Final test results: {:.5f} +/- {:.5f} (sem)".format(np.mean(ica_accuracies), sem(ica_accuracies))))
            print(data_path, seed, np.mean(ica_accuracies))
            results = {
                'data': data_path,
                'split_seed': seed,
                'unlabel_prob': unlabel_prob,
                'accuracy': np.mean(ica_accuracies)
            }
            results_master.append(results)
            #     print('append----------------------')
            #     accs.append(test_acc)
            #     import csv

            # with open('gcn_' + data.replace('/', '_'), 'w') as myfile:
            #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            #     wr.writerow(accs)
            # sys.stdout.flush()
        pd.DataFrame(results_master).to_csv('ica.csv')
