import argparse
import numpy as np

from dataset import Dataset, LoadData
from model import MetaModel
from metrics import MovingAvg

data_dir = "/Users/songdongdong/workSpace/datas/query_completion_data/"

parser = argparse.ArgumentParser()
parser.add_argument('--expdir', help='experiment directory',
                    default="/Users/songdongdong/PycharmProjects/query_completion/model/1605774995")
parser.add_argument('--data', type=str, action='append', dest='data', default=[data_dir + "queries07.test.txt.gz"],
                    help='where to load the data')
parser.add_argument('--threads', type=int, default=12,
                    help='how many threads to use in tensorflow')
args = parser.parse_args()
expdir = args.expdir

# 模型加载
metamodel = MetaModel(expdir)
model = metamodel.model
metamodel.MakeSessionAndRestore(args.threads)
# 数据加载
df = LoadData(args.data)
dataset = Dataset(df, metamodel.char_vocab, metamodel.user_vocab,
                  max_len=metamodel.params.max_len)

total_word_count = 0
total_log_prob = 0
print(len(dataset.df), dataset.batch_size)  # 20999    24
for idx in range(0, int(len(dataset.df) / dataset.batch_size)):
    feed_dict = dataset.GetFeedDict(model)
    # 这里的session 是 获取的是 保存后的模型
    c, words_in_batch = metamodel.session.run([model.avg_loss, model.words_in_batch],
                                              feed_dict)
    # c是 total_loss, words_in_batch 一个batch里字数
    total_word_count += words_in_batch  # 整个字数
    total_log_prob += float(c * words_in_batch)
    print('整体损失值： {0}\t{1:.3f}'.format(idx, np.exp(total_log_prob / total_word_count)))
