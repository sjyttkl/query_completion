"""Script for training language models."""
import argparse
import logging
import os
import pandas
import time
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import helper
from dataset import Dataset, LoadData
from model import Model
from metrics import MovingAvg
from vocab import Vocab

# ./ trainer.py  path / to / expdir - -data / path / to / data.tsv - -valdata / path / to / valdata.tsv

parser = argparse.ArgumentParser()

data_dir2 = "/Users/songdongdong/workSpace/datas/aol_search_query_logs/process/"
data_dir = "/Users/songdongdong/workSpace/datas/query_completion_data/"
dir = "/Users/songdongdong/PycharmProjects/query_completion/model"

parser.add_argument('--expdir', help='experiment directory',
                    default=dir)
parser.add_argument('--params', type=str, default='default_params.json',
                    help='json file with hyperparameters')
# parser.add_argument('--data', type=str, action='append', dest='data',default=[data_dir+"user_query_time_1.txt",data_dir+"user_query_time_2.txt",data_dir+"user_query_time_3.txt",data_dir+"user_query_time_4.txt",data_dir+"user_query_time_5.txt",data_dir+"user_query_time_6.txt"],

parser.add_argument('--data', type=str, action='append', dest='data',
                    default=[data_dir + "queries01.train.txt.gz", data_dir + "queries02.train.txt.gz",
                             data_dir + "queries03.train.txt.gz"],
                    help='where to load the data from')
parser.add_argument('--valdata', type=str, action='append', dest='valdata',
                    # help='where to load validation data', default=[data_dir+"user_query_time_7.txt",data_dir+"user_query_time_8.txt",data_dir+"user_query_time_9.txt",data_dir+"user_query_time_10.txt"])
                    help='where to load validation data',
                    default=[data_dir + "queries01.dev.txt.gz", data_dir + "queries02.dev.txt.gz",
                             data_dir + "queries03.dev.txt.gz"])
parser.add_argument('--threads', type=int, default=12,
                    help='how many threads to use in tensorflow')
args = parser.parse_args()


# expdir = args.expdir
expdir = dir
if not os.path.exists(expdir):
    os.mkdir(expdir)
else:
    for file in os.listdir(expdir):
        file = expdir+"/"+file
        os.remove(file)
    os.removedirs(expdir)

    print('ERROR: expdir already exists')
    # exit(-1)

    # tf.set_random_seed(int(time.time() * 1000))
    tf.compat.v1.set_random_seed(int(time.time() * 1000))
params = helper.GetParams(args.params, 'train', args.expdir)

logging.basicConfig(filename=os.path.join(expdir, 'logfile.txt'),
                    level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())

df = LoadData(args.data)
char_vocab = Vocab.MakeFromData(df.query_, min_count=10)
char_vocab.Save(os.path.join(args.expdir, 'char_vocab.pickle'))
params.vocab_size = len(char_vocab)
user_vocab = Vocab.MakeFromData([[u] for u in df.user], min_count=15)
user_vocab.Save(os.path.join(args.expdir, 'user_vocab.pickle'))
params.user_vocab_size = len(user_vocab)
dataset = Dataset(df, char_vocab, user_vocab, max_len=params.max_len,
                  batch_size=params.batch_size)

val_df = LoadData(args.valdata)
valdata = Dataset(val_df, char_vocab, user_vocab, max_len=params.max_len,
                  batch_size=params.batch_size)

model = Model(params)
saver = tf.train.Saver(tf.global_variables())
config = tf.ConfigProto(inter_op_parallelism_threads=args.threads,
                        intra_op_parallelism_threads=args.threads)
session = tf.Session(config=config)
session.run(tf.global_variables_initializer())

avg_loss = MovingAvg(0.97)  # exponential moving average of the training loss
for idx in range(params.iters):
    feed_dict = dataset.GetFeedDict(model)
    feed_dict[model.dropout_keep_prob] = params.dropout

    c, _ = session.run([model.avg_loss, model.train_op], feed_dict)
    cc = avg_loss.Update(c)
    if idx % 50 == 0 and idx > 0:
        # test one batch from the validation set
        val_c = session.run(model.avg_loss, valdata.GetFeedDict(model))
        logging.info({'iter': idx, 'cost': cc, 'rawcost': c,
                      'valcost': val_c})
    if idx % 2000 == 0:  # save a model file every 2,000 minibatches
        saver.save(session, os.path.join(expdir, 'model.bin'),
                   write_meta_graph=True, global_step=idx)
        # gd = tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), ['add'])
        # with tf.gfile.GFile('./tmodel/model.pb', 'wb') as f:
        #     f.write(gd.SerializeToString())
        print(" 模型保持成功")

# if __name__ == "__main__":
# ./ trainer.py / path / to / expdir - -data / path / to / data.tsv - -valdata / path / to / valdata.tsv
