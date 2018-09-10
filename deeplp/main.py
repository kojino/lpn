import argparse
import copy
import datetime
import logging
import os
import random
import sys
from random import shuffle
import math
import networkx as nx
import numpy as np
import tensorflow as tf

from deeplp.models.deeplp_att import DeepLP_ATT
from deeplp.models.deeplp_edge import DeepLP_Edge
from deeplp.models.deeplp_wrbf import DeepLP_WRBF
from deeplp.models.lp import LP
from deeplp.utils import (calc_masks, create_seed_features, load_data,
                          num_layers_dict, prepare_data, random_unlabel)

logger = logging.getLogger("deeplp")

def approx_chunk(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

def main(args):
 
    if args.crossval_k != 1:
        args.num_epoch = int(args.num_epoch / args.crossval_k)

    if args.ckpt == "0":
        args.ckpt = 0

    if not os.path.exists('accs'):
        os.makedirs('accs')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    if args.num_layers == -1:
        args.num_layers = num_layers_dict[args.data]

    # change exp_name to include any varying parameters
    date = datetime.datetime.now()
    exp_name = (
        f"deeplp_{args.batch_size}_{args.decay}_{args.lamda}_{args.lr}_{args.split_seed}_{args.unlabel_prob}_")

    # create directory and file for saving log
    exp_dir = 'experiment_results/' + exp_name
    os.makedirs(exp_dir, exist_ok=True)
    log_file = exp_name + ".log"
    logging.basicConfig(
        format='%(asctime)s: %(message)s',
        level=args.log,
        datefmt='%m/%d/%Y %I:%M:%S %p')
    file_handler = logging.FileHandler(exp_dir + "/" + log_file)
    logging.getLogger().addHandler(file_handler)

    # print all params
    param_str = '\n'.join([f"{k}: {v}" for k, v in sorted(vars(args).items())])
    logger.debug("--- All settings used: ---")
    for k, v in sorted(vars(args).items()):
        logger.debug("{0}: {1}".format(k, v))
    logger.debug("------")

    true_labels, features, graph = load_data(
        args.data, model=args.model, feature_type=args.feature_type)

    labeled_indices, unlabeled_indices = \
        random_unlabel(true_labels, args.unlabel_prob, args.split_seed)

    num_nodes, num_classes = true_labels.shape
    labels, is_labeled = calc_masks(true_labels, labeled_indices,
                                    unlabeled_indices)
    if args.feature_type == 'all':
        seed_features = create_seed_features(graph, labeled_indices, true_labels)
        if len(features) == 0:
            features = seed_features
        else:
            features = np.hstack([features, seed_features])
    _, num_features = features.shape



    final_accs = []

    labeled_indices_copy = copy.copy(labeled_indices)
    random.seed(args.split_seed)
    shuffle(labeled_indices_copy)
    cv_held_out_indices_list = approx_chunk(labeled_indices_copy, args.crossval_k)

    if args.crossval_k == 1:
        cv_held_out_indices_list = [[]]

    for i, cv_held_out_indices in enumerate(cv_held_out_indices_list):
        logger.info(f"{i}th cross validation")
        cv_labeled_indices = [index for index in labeled_indices if index not in cv_held_out_indices]
        cv_unlabeled_indices = np.delete(np.arange(true_labels.shape[0]),cv_labeled_indices)
        cv_labels, cv_is_labeled = calc_masks(true_labels, cv_labeled_indices, cv_unlabeled_indices)
        
        tf.reset_default_graph()

        logger.info('Applied leave-k-out.')

        if args.model == 'att':
            Model = DeepLP_ATT
        elif args.model == 'edge':
            Model = DeepLP_Edge
        else:
            Model = DeepLP_WRBF

        model = Model(
            graph,
            features,
            num_classes,
            num_features,
            num_nodes,
            weighted_loss=args.weighted_loss,
            lr=args.lr,
            num_layers=args.num_layers,
            lamda=2**args.lamda,
            seed=args.split_seed,
            bifurcation=args.bifurcation,
            decay=args.decay)

        train_data, validation_data, num_samples = prepare_data(model,
            cv_labels, cv_is_labeled, cv_labeled_indices, None, true_labels,
            args.leave_k, args.num_samples, args.split_seed)

        logger.info('Model built.')

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(f'log/{exp_name}', sess.graph)
        saver = tf.train.Saver()
        ckpt_dir = f"ckpt/{exp_name}"
        os.makedirs(ckpt_dir, exist_ok=True)
        if args.ckpt:
            saver.restore(sess, f"{args.ckpt}/model.ckpt")
        if args.save_params:
            os.makedirs(f"params/summary", exist_ok=True)
            os.makedirs(f"params/theta", exist_ok=True)

        thetas = []
        as_ = []
        bs_ = []
        accs = []
        losses = []
        objectives = []
        valaccs = []

        for epoch in range(args.num_epoch):
            if args.batch_size > num_samples:
                batch_size = num_samples
            else:
                batch_size = args.batch_size
            batch_data = {model.global_step: epoch}
            batch_indices = np.random.choice(num_samples, batch_size, replace=False)
            for key in train_data:
                if train_data[key].shape[1] == num_samples:
                    batch_data[key] = train_data[key][:,batch_indices,:]
                else:
                    batch_data[key] = train_data[key]
            _, summary, l_o_loss, objective, validation_accuracy = sess.run(
                [model.opt_op, model.summary_op, model.l_o_loss, model.objective, model.validation_accuracy], feed_dict=batch_data)
            loss, accuracy = sess.run(
                [model.loss, model.accuracy], feed_dict=validation_data)
            if math.isnan(loss) or math.isnan(l_o_loss):
                break
            logger.info(f"Epoch: {epoch}, L-k-Loss: "
                        f"{l_o_loss:.5f}, Loss: {loss:.5f}, Accuracy: {accuracy:.5f}")
            if args.save_params:
                a, b = sess.run([model.a, model.b])
                if args.model == 'edge':
                    theta = sess.run([model.theta])
                    thetas.append(np.array(theta)[0,0,:])
                as_.append(a)
                bs_.append(b)
                accs.append(accuracy)
                losses.append(loss)
                objectives.append(objective)
                valaccs.append(validation_accuracy)
            writer.add_summary(summary, global_step=epoch)
            if epoch != 0 and (epoch + 1) % 10 == 0:
                logger.info('saving checkpoint')
                save_path = saver.save(sess, f"{ckpt_dir}/model.ckpt")
                if args.save_params:
                    if args.model == 'edge':
                        np.savetxt(f'params/theta/theta_{exp_name}.csv', np.array(thetas),delimiter=',',fmt="%.6f")                    
                    np.savetxt(f'params/summary/summary_{exp_name}.csv', np.array([accs,valaccs,losses,as_,bs_,objectives]).T,delimiter=',',fmt="%.6f")
        writer.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--batch_size', default=10, type=int, help='batch size')

    parser.add_argument(
        '--bifurcation', default=0.01, type=float, help='')

    parser.add_argument(
        '--ckpt',
        default=None,
        type=str,
        help='ckpt file name. If provided, restore variables from that file.')

    parser.add_argument('--crossval_k', default=1, type=int,
                        help='number of folds in crossvalidation for \
                              choosing the optimal regularization')

    parser.add_argument(
        '--data', default='linqs_cora', help='data to run deeplp on')

    parser.add_argument(
        '--decay', default=0, help='decay lr', type=int)

    parser.add_argument(
        '--feature_type',
        default='all',
        help='types of features to use',
        choices=['all', 'raw', 'raw_reduced'])

    parser.add_argument(
        '--lamda', default=-14, type=float, help='regularization parameter, 2**(x)')

    parser.add_argument('--leave_k', default=1, type=int,
                        help='number of labeled nodes to mask')

    parser.add_argument(
        '--log',
        default="DEBUG",
        type=str,
        help='logging level',
        choices=["DEBUG", "INFO"])

    parser.add_argument('--lr', default=0.01, type=float,
                        help='learning rate')

    parser.add_argument(
        '--model',
        default='edge',
        help='model for propagation',
        choices=['edge', 'att', 'wrbf'])

    parser.add_argument('--num_epoch', default=10, type=int,
                        help='number of epochs to run the model')

    parser.add_argument(
        '--num_layers',
        default=30,  # 
        type=int,
        help='number of hidden layers of the network, \
        if -1, use precalculated optimal num_layers for each dataset, \
        otherwise use the given number')

    parser.add_argument('--num_samples', default=100, type=int,
                        help='number of samples to train the model')

    parser.add_argument(
        '--save_params',
        default=1,
        type=int,
        help='if 1, save theta and bif a, b to csv every epoch')

    parser.add_argument('--split_seed', default=1, type=int,
                        help='random seed for labeled/unlabeled split')

    parser.add_argument('--unlabel_prob', default=0.99, type=float,
                        help='fraction of unlabeled nodes')

    parser.add_argument(
        '--weighted_loss',
        default=1,
        type=int,
        help='if 1, apply weighted loss; losses on minority labels \
                            are penalized more')

    args = parser.parse_args()
    main(args)
