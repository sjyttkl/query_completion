from __future__ import print_function
import gzip
import os
import pandas
import pygtrie
import re
import sys
import numpy as np
from dataset import LoadData
from helper import GetPrefixLen

# 针对 log日志做 前缀树的 baseline

query_trie = pygtrie.CharTrie()
dirname = "/Users/songdongdong/workSpace/datas/query completion data/"

# dirname = '../data'
# filenames = ['queries01.train.txt.gz', 'queries02.train.txt.gz',
#              'queries03.train.txt.gz', 'queries04.train.txt.gz',
#              'queries05.train.txt.gz', 'queries06.train.txt.gz'
# ]

filenames = ['queries01.train.txt.gz'
             ]
df = LoadData([os.path.join(dirname, f) for f in filenames], split=False)
z = df.query_.value_counts()
z = z[z > 2]
# 前缀树 建立完成
for q, count in zip(z.index.values, z):
    query_trie[q] = count

cache = {}


def GetTopK(prefix, k=100):  # 提取topk个，并且按照词频排序
    if prefix in cache:
        return cache[prefix]
    results = query_trie.items(prefix)  # 找出 该前缀树中 以 prefix 前缀 的搜有字符串 以及 词频
    queries, counts = zip(*sorted(results, key=lambda x: -x[-1]))  # 按照词频排序，提取
    cache[prefix] = queries[:k]
    return queries[:k]


# 根据query，查找候选词下面的query
def GetRankInList(query, qlist):
    if query not in qlist:
        return 0
    return 1.0 / (1.0 + qlist.index(query))


regex_eval = re.compile(r"'(\w*)': '?([^,]+)'?[,}]")


def FastLoadDynamic(filename):
    rows = []
    with gzip.open(filename, 'r') as f:
        for line in f:
            if type(line) != str:
                line = line.decode('utf8')  # for python3 compatibility
            matches = regex_eval.finditer(line)
            d = dict([m.groups() for m in matches])
            if len(d) > 0:
                rows.append(d)
            else:
                print('bad line')
                print(line)
    dynamic_df = pandas.DataFrame(rows)
    if len(dynamic_df) > 0:
        if 'cost' in dynamic_df.columns:
            dynamic_df['cost'] = dynamic_df.cost.astype(float)
        if 'length' in dynamic_df.columns:
            dynamic_df['length'] = dynamic_df['length'].astype(float)
        dynamic_df['score'] = dynamic_df['score'].astype(float)
    return dynamic_df


rank_data = FastLoadDynamic(dirname + '/predictions.log.gz')

for i in range(len(rank_data)):
    row = rank_data.iloc[i]  # 提取一行的数据，并且每个元素 以元组形式展现
    if sys.version_info.major == 2:
        query = row['query'][:-1].decode('string_escape')
    else:  # python3 compatability
        query = row['query'][:-1].encode('utf8').decode('unicode_escape')
    query_len = len(query)  # 最终query长度

    prefix_len = int(row.prefix_len)  # 前缀出现长度

    prefix_not_found = False
    prefix = query[:prefix_len]  # 提取前准
    if not query_trie.has_subtrie(prefix):  # 查看 前缀树里是否存在该前缀
        prefix_not_found = True  # 表示该前缀树不存在前缀
        score = 0.0
    else:
        qlist = GetTopK(prefix)  # 存在前缀树，并查找出该前缀的搜有query
        score = GetRankInList(query, qlist)  # 根据该query中

    result = {'query': query, 'prefix_len': int(prefix_len),
              'score': score, 'user': row.user,
              'prefix_not_found': prefix_not_found}
    print(result)
    if i % 100 == 0:
        sys.stdout.flush()
