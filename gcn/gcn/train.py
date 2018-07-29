from __future__ import division
from __future__ import print_function
import sys
import time
import tensorflow as tf

from gcn.utils import *
from gcn.models import GCN, MLP

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 100, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
data_path = 'linqs_data/cora'
seed = 10

validation = False

for data_path in ['linqs_data/pubmed']:

    for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        print(data_path,seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

        # Load data
        if validation:
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_2(data_path,seed)
        else:
            adj, features, y_train, y_test, train_mask, test_mask = load_data(data_path,seed)

        # Some preprocessing
        features = preprocess_features(features)
        if FLAGS.model == 'gcn':
            support = [preprocess_adj(adj)]
            num_supports = 1
            model_func = GCN

        # Define placeholders
        placeholders = {
            'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
            'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
            'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
            'labels_mask': tf.placeholder(tf.int32),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
        }

        # Create model
        model = model_func(placeholders, input_dim=features[2][1], logging=True)

        # Initialize session
        sess = tf.Session()


        # Define model evaluation function
        def evaluate(features, support, labels, mask, placeholders):
            t_test = time.time()
            feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
            outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
            return outs_val[0], outs_val[1], (time.time() - t_test)


        # Init variables
        sess.run(tf.global_variables_initializer())

        cost_val = []

        # Train model
        early_stop=0

        for epoch in range(FLAGS.epochs):

            t = time.time()
            # Construct feed dictionary
            feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

            # Validation
            if validation:
                cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
                cost_val.append(cost)
            # Testing
            test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)

            if validation:
                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
                      "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
                      "val_acc=", "{:.5f}".format(acc), 'test_loss=', "{:.5f}".format(test_cost),
                      'test_acc=', "{:.5f}".format(test_acc), "time=", "{:.5f}".format(time.time() - t),
                      "early_stopping=", early_stop)
            else:
                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
                      "train_acc=", "{:.5f}".format(outs[2]),'test_loss=', "{:.5f}".format(test_cost),
                      'test_acc=', "{:.5f}".format(test_acc), "time=", "{:.5f}".format(time.time() - t),
                      "early_stopping=", early_stop)

            sys.stdout.flush()
            if validation:
                if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
                    # print("Early stopping...")
                    early_stop=1
                # break
    sys.stdout.flush()
