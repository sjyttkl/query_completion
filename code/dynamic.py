from __future__ import print_function
import argparse
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf

import time
import sys
from beam import GetCompletions
from dataset import LoadData, Dataset
from helper import GetPrefixLen
from metrics import GetRankInList, MovingAvg
from model import MetaModel

# data_dir = "/Users/songdongdong/workSpace/datas/aol_search_query_logs/process/"
data_dir = "/Users/songdongdong/workSpace/datas/query_completion_data/"

parser = argparse.ArgumentParser()
parser.add_argument('--expdir', help='experiment directory',
                    default="/Users/songdongdong/PycharmProjects/query_completion/project")
parser.add_argument('--data', type=str, action='append', dest='data', default=[data_dir + "queries07.train.txt.gz"],
                    # ,data_dir+"queries07.dev.txt.gz",data_dir+"queries07.test.txt.gz"
                    help='where to load the data')
parser.add_argument('--optimizer', default='ada',
                    choices=['sgd', 'adam', 'ada', 'adadelta'],
                    help='which optimizer to use to learn user embeddings')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--threads', type=int, default=12,
                    help='how many threads to use in tensorflow')
parser.add_argument('--tuning', action='store_true', dest='tuning',
                    help='when tuning don\'t do beam search decoding',
                    default=False)
parser.add_argument('--partial', action='store_true', dest='partial',
                    help='do partial matching rank', default=False)
parser.add_argument('--limit', type=int, default=385540,
                    help='how many queries to evaluate')
args = parser.parse_args()


class DynamicModel(MetaModel):

    def __init__(self, expdir, learning_rate=None, threads=8,
                 optimizer=tf.train.AdagradOptimizer):
        super(DynamicModel, self).__init__(expdir)

        if learning_rate is None:  # set the default learning rates
            if self.params.use_lowrank_adaptation:
                learning_rate = 0.15
            else:
                learning_rate = 0.925

        self.MakeSessionAndRestore(threads)
        # tf.reset_default_graph()
        with self.graph.as_default():  # add some nodes to the tensorflow graph
            unk_embed = self.model.user_embed_mat.eval(session=self.session)[
                self.user_vocab['<UNK>']]  # 恢复 <UNK> embedding中的 unk向量

            self.reset_user_embed = tf.scatter_update(self.model.user_embed_mat, [0],
                                                      np.expand_dims(unk_embed, 0))  # 添加一个embding (86580, 32)

            self.session.run(self.reset_user_embed)  # 上面的步骤运行了一次。

            with tf.variable_scope('optimizer'):
                self.train_op = tf.no_op()  # 专门对 user_embed进行更新？ no_op()什么都不做，仅做为点位符使用控制边界
                if (self.params.use_lowrank_adaptation or self.params.use_mikolov_adaptation):
                    self.train_op = optimizer(learning_rate).minimize(self.model.avg_loss,
                                                                      var_list=[self.model.user_embed_mat])

            opt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "optimizer")
            if len(opt_vars):  # check if optimzer state needs resetting
                self.reset_user_embed = tf.group(self.reset_user_embed,
                                                 tf.variables_initializer(opt_vars))  # 这里的操作将会依次执行，没有返回值，因为这里是操作

    def Train(self, query, userid, hourofday, dayofweek, train=True):
        # If train is false then it will just do the forward pass
        qIds = np.zeros((1, self.params.max_len))
        for i in range(min(self.params.max_len, len(query))):
            qIds[0, i] = self.char_vocab[query[i]]

        feed_dict = {
            self.model.user_ids: np.array([userid]),
            self.model.query_lengths: np.array([len(query)]),
            self.model.queries: qIds,
            self.model.hourofday: np.array([hourofday]),
            self.model.dayofweek: np.array([dayofweek])
        }

        if train:
            c, words_in_batch, _ = self.session.run(
                [self.model.avg_loss, self.model.words_in_batch, self.train_op],
                feed_dict)
            return c, words_in_batch
        else:
            # just compute the forward pass
            return self.session.run([self.model.avg_loss, self.model.words_in_batch],
                                    feed_dict)


if __name__ == '__main__':
    optimizer = {'sgd': tf.train.GradientDescentOptimizer,
                 'adam': tf.train.AdamOptimizer,
                 'ada': tf.train.AdagradOptimizer,
                 'adadelta': tf.train.AdadeltaOptimizer}[args.optimizer]
    # 这里其实包括了，对用户id进行初始化操作l
    mLow = DynamicModel(args.expdir, learning_rate=args.learning_rate,
                        threads=args.threads, optimizer=optimizer)
    # print(mLow.user_vocab['<UNK>'])
    # print(mLow.user_vocab.word_to_idx)
    # print(mLow.user_vocab.word_to_idx["s10015521"])
    df = LoadData(args.data)
    users = df.groupby('user')
    avg_time = MovingAvg(0.95)

    stop = '</S>'  # decide if we stop at first space or not
    if args.partial:
        stop = ' '

    counter = 0
    for user, grp in users:
        grp = grp.sort_values('date')
        mLow.session.run(mLow.reset_user_embed)  # 更新 userembedding

        for i in range(len(grp)):
            row = grp.iloc[i]
            query = ''.join(row.query_[1:-1])  # query截取 第二位 到 倒数第二位，
            if len(query) < 3:
                continue

            start_time = time.time()
            result = {'query': query, 'user': row.user, 'idx': i}
            # print(result)
            # run the beam search decoding
            if not args.tuning:
                prefix_len = GetPrefixLen(row.user, query, i)  # query的前缀长度
                prefix = row.query_[:prefix_len + 1]  # 前缀
                # print(prefix)
                # print(user)
                if user in mLow.user_vocab.word_to_idx:
                    user2 = mLow.user_vocab.word_to_idx[user]
                else:
                    user2 = 0
                b = GetCompletions(prefix, user2, mLow, branching_factor=4,
                                   beam_size=100, stop=stop)  # always use userid=0
                # print(b.size)
                qlist = [''.join(q.words[1:-1]) for q in reversed(list(b))]

                if args.partial and ' ' in query[prefix_len:]:
                    word_boundary = query[prefix_len:].index(' ')
                    query = query[:word_boundary + prefix_len]
                score = GetRankInList(query, qlist)
                result['user'] = user2
                result['score'] = score
                result['top_completion'] = qlist[0]
                result['hourofday'] = grp.hourofday
                result['dayofweek'] = grp.dayofweek
                result['prefix_len'] = int(prefix_len)
            # print(i,len(grp)-1)

            result['cost'], result['length'] = mLow.Train(row.query_, grp.hourofday, grp.dayofweek,
                                                          train=i != len(grp) - 1)
            print(result)
            counter += 1
            t = avg_time.Update(time.time() - start_time)

            if i % 25 == 0:
                sys.stdout.flush()  # flush every so often
                sys.stderr.write('{0}\n'.format(t))
        if counter > args.limit:
            break
