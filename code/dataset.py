"""Holds the Dataset class used for managing training and test data."""
import datetime
import pandas
import numpy as np
import os


# 数据说明：https://github.com/sjyttkl/AOL-Exploratory-Data-Analysis
# 数据来源于：http://www.cim.mcgill.ca/~dudek/206/Logs/AOL-user-ct-collection/


def LoadData(filenames, split=True):
    """Load a bunch of files as a pandas dataframe.

    Input files should have three columns for userid, query, and date.
    """

    # filenames = []
    # for file in os.listdir(datapath):
    #     filenames.append(datapath+"/"+file)

    def Prepare(s):
        s = str(s)
        return ['<S>'] + list(s) + ['</S>']

    dfs = []
    for filename in filenames:
        df = pandas.read_csv(filename, compression='gzip', sep='\t', header=None)  # gzip
        df = df[:1000]
        df.columns = ['user', 'query_', 'date']
        if split:
            df['query_'] = df.query_.apply(Prepare)
        df['user'] = df.user.apply(lambda x: 's' + str(x))

        dates = df.date.apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        df['hourofday'] = [d.hour for d in dates]
        df['dayofweek'] = [d.dayofweek for d in dates]
        dfs.append(df)
        print(df[0:20])
    return pandas.concat(dfs)


class Dataset(object):

    def __init__(self, df, char_vocab, user_vocab, batch_size=24, max_len=60):
        self.max_len = max_len
        self.char_vocab = char_vocab
        self.user_vocab = user_vocab
        self.df = df.sample(frac=1)  # 表示 按照多少比例进行 采样
        self.batch_size = batch_size
        self.current_idx = 0

        self.df['lengths'] = self.df.query_.apply(
            lambda x: min(self.max_len, len(x)))

    #
    def GetFeedDict(self, model):
        if self.current_idx + self.batch_size > len(self.df):
            self.current_idx = 0

        idx = range(self.current_idx, self.current_idx + self.batch_size)  # 需要迭代的范围 batch_size
        self.current_idx += self.batch_size  # 目前的位置

        grp = self.df.iloc[idx]  # 提取需要迭代数据

        f1 = np.zeros((self.batch_size, self.max_len))  # 需要迭代数据，矩阵
        user_ids = np.zeros(self.batch_size)  # 需要迭代数据，user_id
        dayofweek = np.zeros(self.batch_size)  # 需要迭代数据，user_id
        hourofday = np.zeros(self.batch_size)  # 需要迭代数据，user_id

        lockedW = np.zeros([536, 3 * 512])
        lockedBias = np.zeros([3 * 513])

        feed_dict = {
            model.queries: f1,
            model.query_lengths: grp.lengths.values,
            model.user_ids: user_ids,
            model.decoder_cell.lockedW: lockedW

        }
        if model.use_time_features:
            # print(grp.dayofweek.values)
            # print(grp.dayofweek)
            # print(np.array(grp.dayofweek.values).reshape(-1))
            feed_dict[model.dayofweek] = dayofweek  # np.array(grp.dayofweek.values).reshape(-1),
            feed_dict[model.hourofday] = hourofday  # grp.hourofday.values.reshape(-1)

        # 这里是上面先定义，后面再输入数据，感觉有点像老外的被动语法
        for i in range(0, len(grp)):
            row = grp.iloc[i]
            user_ids[i] = self.user_vocab[row.user]
            if model.use_time_features:
                dayofweek[i] = grp.dayofweek.values[i]
                hourofday[i] = grp.hourofday.values[i]
            for j in range(row.lengths):
                f1[i, j] = self.char_vocab[row.query_[j]]

        return feed_dict
