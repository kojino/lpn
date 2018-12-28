from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn_original.utils import *
from gcn_original.models import GCN, MLP

# Set random seed

# np.random.seed(seed)
tf.set_random_seed(1234)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
# flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 2000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')



for setting in ['planetoid_balanced']:
    test_accs = np.zeros((24,11,3))
    unlabeled_accs = np.zeros((24,11,3))
    gcc_accs = np.zeros((24,11,3))
    nogcc_accs = np.zeros((24,11,3))
    for i,dataset in enumerate(['pubmed']):
        for j,seed in enumerate([0,1,2,3,4]):
            for k,num_ratio in enumerate([1/20,1/10,1/8,1/6,1/4,1/3,1/2,1,2,3,4,5]):
            # for num_val in [500]:
                print('---------------------',12 * i + k, j, setting,dataset,seed,num_ratio,'---------------------')
                # Load data
                tf.reset_default_graph()
                tf.set_random_seed(1234)
                adj, features, y_train, y_val, y_test, y_unlabeled, y_gcc, y_nogcc, train_mask, val_mask, test_mask, unlabeled_mask, gcc_mask, nogcc_mask = load_data(dataset,seed,setting,num_ratio)
                print('==========================')
                print(np.sum(y_val))
                # Some preprocessing
                features = preprocess_features(features)
                if FLAGS.model == 'gcn':
                    support = [preprocess_adj(adj)]
                    num_supports = 1
                    model_func = GCN
                elif FLAGS.model == 'gcn_cheby':
                    support = chebyshev_polynomials(adj, FLAGS.max_degree)
                    num_supports = 1 + FLAGS.max_degree
                    model_func = GCN
                elif FLAGS.model == 'dense':
                    support = [preprocess_adj(adj)]  # Not used
                    num_supports = 1
                    model_func = MLP
                else:
                    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

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
                with sess.as_default() as sess:

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
                    for epoch in range(FLAGS.epochs):

                        t = time.time()
                        # Construct feed dictionary
                        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
                        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

                        # Training step
                        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

                        # Validation
                        cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
                        cost_val.append(cost)

                        # # Print results
                        # print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
                        #     "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
                        #     "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

                        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
                            print("Early stopping...")
                            break

                    print("Optimization Finished!")

                    # Testing
                    test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
                    print("Test set results:", "acc=", "{:.5f}".format(test_acc),
                    "epoch=", "{:.5f}".format(epoch), "valacc=", "{:.5f}".format(acc))
                    test_accs[12 * i + k, j, 0] = test_acc
                    test_accs[12 * i + k, j, 1] = epoch
                    test_accs[12 * i + k, j, 2] = acc
                    

                    unlabeled_cost, unlabeled_acc, unlabeled_duration = evaluate(features, support, y_unlabeled, unlabeled_mask, placeholders)
                    print("Unlabeled set results:", "acc=", "{:.5f}".format(test_acc),
                    "epoch=", "{:.5f}".format(epoch), "valacc=", "{:.5f}".format(acc))
                    unlabeled_accs[12 * i + k, j, 0] = unlabeled_acc
                    unlabeled_accs[12 * i + k, j, 1] = epoch
                    unlabeled_accs[12 * i + k, j, 2] = acc
                    

                    gcc_cost, gcc_acc, gcc_duration = evaluate(features, support, y_gcc, gcc_mask, placeholders)
                    print("gcc set results:", "acc=", "{:.5f}".format(test_acc),
                    "epoch=", "{:.5f}".format(epoch), "valacc=", "{:.5f}".format(acc))
                    gcc_accs[12 * i + k, j, 0] = gcc_acc
                    gcc_accs[12 * i + k, j, 1] = epoch
                    gcc_accs[12 * i + k, j, 2] = acc

                    nogcc_cost, nogcc_acc, nogcc_duration = evaluate(features, support, y_nogcc, nogcc_mask, placeholders)
                    print("nogcc set results:", "acc=", "{:.5f}".format(test_acc),
                    "epoch=", "{:.5f}".format(epoch), "valacc=", "{:.5f}".format(acc))
                    nogcc_accs[12 * i + k, j, 0] = nogcc_acc
                    nogcc_accs[12 * i + k, j, 1] = epoch
                    nogcc_accs[12 * i + k, j, 2] = acc                
        
    test_accs.dump(f'gcn-planetoid-test-{setting}.p')
    unlabeled_accs.dump(f'gcn-planetoid-unlabeled-{setting}.p')
    gcc_accs.dump(f'gcn-planetoid-gcc-{setting}.p')
    nogcc_accs.dump(f'gcn-planetoid-nogcc-{setting}.p')
    # np.savetxt(f'gcn-planetoid-test-{setting}.csv',test_accs,delimiter=',')
    # np.savetxt(f'gcn-planetoid-unlabeled-{setting}.csv',unlabeled_accs,delimiter=',')
    # np.savetxt(f'gcn-planetoid-gcc-{setting}.csv',gcc_accs,delimiter=',')
    # np.savetxt(f'gcn-planetoid-nogcc-{setting}.csv',nogcc_accs,delimiter=',')