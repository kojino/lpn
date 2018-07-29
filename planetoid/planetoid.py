
from scipy import sparse as sp
from trans_model import trans_model as model
import argparse
import cPickle
import sys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--steps', help = 'number of overall learning steps', type = int, default = 1000)
parser.add_argument('--init_iter_label', help = 'init_iter_label', type = int, default = 2000)
parser.add_argument('--init_iter_graph', help = 'init_iter_graph', type = int, default = 70)
parser.add_argument('--learning_rate', help = 'learning rate for supervised loss', type = float, default = 0.1)
parser.add_argument('--embedding_size', help = 'embedding dimensions', type = int, default = 50)
parser.add_argument('--window_size', help = 'window size in random walk sequences', type = int, default = 3)
parser.add_argument('--path_size', help = 'length of random walk sequences', type = int, default = 10)
parser.add_argument('--batch_size', help = 'batch size for supervised loss', type = int, default = 200)
parser.add_argument('--g_batch_size', help = 'batch size for graph context loss', type = int, default = 200)
parser.add_argument('--g_sample_size', help = 'batch size for label context loss', type = int, default = 100)
parser.add_argument('--neg_samp', help = 'negative sampling rate; zero means using softmax', type = int, default = 0)
parser.add_argument('--g_learning_rate', help = 'learning rate for unsupervised loss', type = float, default = 1e-2)
parser.add_argument('--model_file', help = 'filename for saving models', type = str, default = 'trans.model')
parser.add_argument('--use_feature', help = 'whether use input features', type = bool, default = True)
parser.add_argument('--update_emb', help = 'whether update embedding when optimizing supervised loss', type = bool, default = True)
parser.add_argument('--layer_loss', help = 'whether incur loss on hidden layers', type = bool, default = True)
parser.add_argument('--path', help = 'path of data files', type = str, default = '.')
args = parser.parse_args()

def comp_accu(tpy, ty):
    import numpy as np
    return (np.argmax(tpy, axis = 1) == np.argmax(ty, axis = 1)).sum() * 1.0 / tpy.shape[0]

# load the data: x, y, tx, ty, graph
NAMES = ['x', 'y', 'tx', 'ty', 'graph']
OBJECTS = []
for i in range(len(NAMES)):
    OBJECTS.append(cPickle.load(open("{}/{}".format(args.path, NAMES[i]))))
x, y, tx, ty, graph = tuple(OBJECTS)

m = model(args)                                             # initialize the model
m.add_data(x, y, graph)                                     # add data
m.build()                                                   # build the model
m.init_train(args.init_iter_label, args.init_iter_graph)  # pre-training  2000,70
iter_cnt, max_accu = 0, 0
for i in range(args.steps):
    print 'train step=', i
    m.step_train(max_iter = 1, iter_graph = 0, iter_inst = 1, iter_label = 0)   # perform a training step

tpy = m.predict(tx)                                                         # predict the dev set
np.savetxt("{}/out".format(args.path), tpy, delimiter = '\t', fmt =  '%.5f')
