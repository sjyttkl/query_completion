# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     Load_Model.py
   email:         songdongdong@weidian.com
   Author :       songdongdong
   date：          2020/12/9 01:34
   Description :  
==================================================
"""

# import tensorflow.compat.v1 as tf
import tensorflow as tf
from helper import GetPrefixLen
from metrics import GetRankInList, MovingAvg
from beam import BeamItem, BeamQueue
import pickle
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import os

import numpy as np



class Load_pb:

    def __init__(self, pb_path_model,pb_model_name,vocab_char,user_vocab):
        self.pb_path_model_name = pb_path_model+pb_model_name
        self.pb_path_model = pb_path_model
        self.vocab_char = vocab_char
        self.user_vocab = user_vocab

    def initBeam(self):
        phrase = self.feed_dict[self.querys]  # 前缀
        prev_hidden = np.zeros((1, 2 * 512))
        for word in phrase[:-1]:
            feed_dict = {
                self.prev_hidden_state: prev_hidden,
                self.prev_word: [self.vocab_char.word_to_idx[word]],
                self.beam_size: 4
            }

            prev_hidden = self.sess.run(self.next_hidden_state, feed_dict)
        nodes = [BeamItem(phrase, prev_hidden)]
        return nodes
    def getCompletion_beam(self, stop='</S>', beam_size=100, branching_factor=8):
        nodes = self.initBeam()
        for i in range(36):
            new_nodes = BeamQueue(max_size=beam_size)
            current_nodes = []
            for node in nodes:
                # print("node.words",node.words)
                if i > 0 and node.words[-1] == stop:  # don't extend past the stop token
                    new_nodes.Insert(node)  # copy over finished beams
                else:
                    current_nodes.append(node)  # these ones will get extended
            if len(current_nodes) == 0:
                return new_nodes  # all beams have finished

            # group together all the nodes in the queue for efficient computation
            prev_hidden = np.vstack([item.prev_hidden for item in current_nodes])
            prev_words = np.array([self.vocab_char.word_to_idx[item.words[-1]] for item in current_nodes])
            # print(prev_words)
            # print("prev_words  ",prev_words)
            # 喂到解码器里
            self.feed_dict = {
                self.prev_word: prev_words,
                self.prev_hidden_state: prev_hidden,
                self.beam_size: branching_factor
            }
            current_char, current_char_p, prev_hidden = self.sess.run(
                [self.beam_chars, self.top_k, self.next_hidden_state],
                self.feed_dict)
            # print(current_char)
            # print(-current_char_p)
            # print(prev_hidden)
            current_char_p = -current_char_p
            for i, node in enumerate(current_nodes):
                for new_word, top_value in zip(current_char[i, :], current_char_p[i, :]):
                    new_cost = top_value + node.log_probs
                    if new_nodes.CheckBound(new_cost):  # only create a new object if it fits in beam
                        new_beam = BeamItem(node.words + [new_word], prev_hidden[i, :],
                                            log_prob=new_cost)
                        new_nodes.Insert(new_beam)
            # print("node.words end. ")
            nodes = new_nodes
        return nodes


    def run_model_pb2(self, feed_dict_pre):
        # We parse the graph_def file
        with tf.gfile.GFile(self.pb_path_model_name, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # We load the graph_def in the default graph
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name="",
                op_dict=None,
                producer_op_list=None
            )
        return graph

    def run_model_pb(self, feed_dict_pre):
        self.feed_dict_pre = feed_dict_pre
        self.sess = tf.Session()
        # sess.run([tf.local_variables_initializer(),tf.global_variables_initializer()])
        ## read pb into graph_def
        with gfile.GFile(self.pb_path_model_name, 'rb') as f: #gfile.FastGFile
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            self.sess.graph.as_default()
            # import graph into session
            tf.import_graph_def(graph_def, name="")

        # tf.train.write_graph(graph_def, '/Users/songdongdong/PycharmProjects/query_completion/pb_model/', 'good_frozen.pb', as_text=False)
        # tf.train.write_graph(graph_def, '/Users/songdongdong/PycharmProjects/query_completion/pb_model/', 'good_frozen.pbtxt', as_text=True)

        # # import graph_def
        # with sess.graph.as_default() as graph:
        #     tf.import_graph_def(graph_def, name='')# 导入计算图
        #     # sess.graph.as_default()

        # 需要有一个初始化的过程
        # 需要有一个初始化的过程
        # self.sess.run([tf.local_variables_initializer(), tf.global_variables_initializer()])

        # 需要先复原变量
        # print(sess.run('b:0'))
        # 1

        # 输入
        self.querys = self.sess.graph.get_tensor_by_name("queries:0")
        self.query_length = self.sess.graph.get_tensor_by_name("query_lengths:0")
        self.user_ids = self.sess.graph.get_tensor_by_name("user_ids:0")
        self.hourofday = self.sess.graph.get_tensor_by_name("hourofday:0")
        self.dayofweek = self.sess.graph.get_tensor_by_name("dayofweek:0")
        self.keep_prob = self.sess.graph.get_tensor_by_name("keep_prob:0")
        self.prev_word = self.sess.graph.get_tensor_by_name("prev_word:0")
        self.beam_size = self.sess.graph.get_tensor_by_name("beam_size:0")
        self.beam_chars = self.sess.graph.get_tensor_by_name("beam_chars:0")
        self.next_hidden_state = self.sess.graph.get_tensor_by_name("next_hidden_state:0")  # output
        self.top_k = self.sess.graph.get_tensor_by_name("top_k:0")
        self.lock_op = self.sess.graph.get_operation_by_name("rnn/factor_cell/lock_op")
        # self.reset_user_embed = self.sess.graph.get_operation_by_name("reset_user_embed")
        self.train_op = self.sess.graph.get_operation_by_name("optimizer/NoOp")  # train_op
        self.avg_loss = self.sess.graph.get_tensor_by_name("avg_loss:0")
        self.prev_hidden_state = self.sess.graph.get_tensor_by_name("prev_hidden_state:0")
        # for op in self.sess.graph.get_operations():
        #     print(op.name, op.values())
        # self.sess.run(self.reset_user_embed)

        self.feed_dict = {self.querys: feed_dict_pre['prefix'],
                     self.user_ids: [feed_dict_pre['user']],
                     self.hourofday: feed_dict_pre['hourofday'],
                     self.dayofweek: feed_dict_pre['dayofweek'],
                     self.keep_prob: 1.0,
                     self.beam_size: 8.0}

        # --------   Lock  --------
        if self.feed_dict[self.user_ids][0] in self.user_vocab.word_to_idx:
            user = self.user_vocab.word_to_idx[self.feed_dict[self.user_ids][0]]
        else:
            user = 0

        self.sess.run(self.lock_op, {self.user_ids: [user],
                           self.hourofday: self.feed_dict[self.hourofday],
                           self.dayofweek: self.feed_dict[self.dayofweek]})

        queue_item2 = self.getCompletion_beam(beam_size=100, branching_factor=4, stop="</S>")
        qlist = ["".join(q.words[1:-1]) for q in reversed(list(queue_item2))]
        print(qlist)
        # partial = False
        # if partial and ' ' in prefix[prefix_len:]:
        #     word_boundary = feed_dict_pre["query"][prefix_len:].index(' ')
        #     query = feed_dict_pre["query"][:word_boundary + prefix_len]
        # score = GetRankInList(feed_dict_pre["query"], qlist)
        # feed_dict_pre['user'] = feed_dict_pre["user"]
        # feed_dict_pre['score'] = score
        # feed_dict_pre['top_completion'] = qlist[0]
        # feed_dict_pre['hourofday'] = feed_dict_pre["hourofday"]
        # feed_dict_pre['dayofweek'] = feed_dict_pre["dayofweek"]
        # feed_dict_pre['prefix_len'] = int(prefix_len)
        #
        print(self.train(True))

    def run_model_pb_serving2(self,feed_dict_pre,pb_model_serving):
        self.sess = tf.Session()
        self.graph = self.sess.graph.as_default()
        MetaGraphDef = tf.saved_model.loader.load(self.sess, tags=[tf.saved_model.tag_constants.SERVING], export_dir=self.pb_path_model + pb_model_serving)
        # 解析得到 SignatureDef protobuf
        SignatureDef_d = MetaGraphDef.signature_def
        SignatureDef = SignatureDef_d[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        # print(SignatureDef)
        # 解析得到 3 个变量对应的 TensorInfo protobuf
        queries = SignatureDef.inputs['querys'].name
        query_lengths = SignatureDef.inputs['query_lengths']
        user_ids = SignatureDef.outputs['user_ids']
        print(queries)
        # 解析得到具体 Tensor
        # .get_tensor_from_tensor_info() 函数中可以不传入 graph 参数，TensorFlow 自动使用默认图
        queries = self.graph.get_tensor_from_tensor_info(queries)
        query_lengths = tf.saved_model.utils.get_tensor_from_tensor_info(query_lengths, self.graph)
        user_ids = tf.saved_model.utils.get_tensor_from_tensor_info(user_ids, self.graph)

        print(self.sess.run(queries))
        # print(sess.run(y, feed_dict={X: [3., 2., 1.]}))


    def run_model_pb_serving (self, feed_dict_pre,pb_model_serving):

        self.sess = tf.Session()

        tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING], self.pb_path_model + pb_model_serving)
        # self.sess.run(tf.global_variables_initializer())

        self.feed_dict_pre = feed_dict_pre


        # 输入
        self.querys = self.sess.graph.get_tensor_by_name("queries:0")
        self.query_length = self.sess.graph.get_tensor_by_name("query_lengths:0")
        self.user_ids = self.sess.graph.get_tensor_by_name("user_ids:0")
        self.hourofday = self.sess.graph.get_tensor_by_name("hourofday:0")
        self.dayofweek = self.sess.graph.get_tensor_by_name("dayofweek:0")
        self.keep_prob = self.sess.graph.get_tensor_by_name("keep_prob:0")
        self.prev_word = self.sess.graph.get_tensor_by_name("prev_word:0")
        self.beam_size = self.sess.graph.get_tensor_by_name("beam_size:0")
        self.beam_chars = self.sess.graph.get_tensor_by_name("beam_chars:0")
        self.next_hidden_state = self.sess.graph.get_tensor_by_name("next_hidden_state:0")  # output
        self.top_k = self.sess.graph.get_tensor_by_name("top_k:0")
        # self.lock_op = self.sess.graph.get_operation_by_name("rnn/factor_cell/lock_op")
        self.lock_op = self.sess.graph.get_tensor_by_name("rnn/factor_cell/lock_op/control_dependency:0")
        # self.reset_user_embed = self.sess.graph.get_operation_by_name("reset_user_embed")
        self.reset_user_embed = self.sess.graph.get_tensor_by_name("reset_user_embed/control_dependency:0")
        self.train_op = self.sess.graph.get_operation_by_name("optimizer/NoOp")  # train_op
        self.avg_loss = self.sess.graph.get_tensor_by_name("avg_loss:0")
        self.prev_hidden_state = self.sess.graph.get_tensor_by_name("prev_hidden_state:0")
        # for op in self.sess.graph.get_operations():
            # print(op.name, op.values())

        self.sess.run(self.reset_user_embed)

        self.feed_dict = {self.querys: feed_dict_pre['prefix'],
                          self.user_ids: [feed_dict_pre['user']],
                          self.hourofday: feed_dict_pre['hourofday'],
                          self.dayofweek: feed_dict_pre['dayofweek'],
                          self.keep_prob: 1.0,
                          self.beam_size: 8.0}

        # --------   Lock  --------
        if self.feed_dict[self.user_ids][0] in self.user_vocab.word_to_idx:
            user = self.user_vocab.word_to_idx[self.feed_dict[self.user_ids][0]]
        else:
            user = 0

        self.sess.run(self.lock_op, {self.user_ids: [user],
                                     self.hourofday: self.feed_dict[self.hourofday],
                                     self.dayofweek: self.feed_dict[self.dayofweek]})

        queue_item2 = self.getCompletion_beam(beam_size=100, branching_factor=4, stop="</S>")
        qlist = ["".join(q.words[1:-1]) for q in reversed(list(queue_item2))]
        print(qlist)
        # partial = False
        # if partial and ' ' in prefix[prefix_len:]:
        #     word_boundary = feed_dict_pre["query"][prefix_len:].index(' ')
        #     query = feed_dict_pre["query"][:word_boundary + prefix_len]
        # score = GetRankInList(feed_dict_pre["query"], qlist)
        # feed_dict_pre['user'] = feed_dict_pre["user"]
        # feed_dict_pre['score'] = score
        feed_dict_pre['top_completion'] = qlist[0]
        # feed_dict_pre['hourofday'] = feed_dict_pre["hourofday"]
        # feed_dict_pre['dayofweek'] = feed_dict_pre["dayofweek"]
        # feed_dict_pre['prefix_len'] = int(prefix_len)
        #
        print(self.train(True))
    def train(self, train=True):
        qIds = np.zeros((1, 12))
        for i in range(min(12, len(self.feed_dict_pre["query_"]))):
            qIds[0, i] = self.vocab_char.word_to_idx[query_[i]]

        feed_dict2 = {
            self.user_ids: np.array([self.user_vocab.word_to_idx[feed_dict_pre["user"]]]),
            self.query_length: np.array([len(self.feed_dict_pre["query_"])]),
            self.querys: qIds,
            self.hourofday: self.feed_dict_pre["hourofday"],
            self.dayofweek: self.feed_dict_pre["dayofweek"]
        }
        if train:
            self.feed_dict_pre['cost'], _ = self.sess.run([self.avg_loss, self.train_op], feed_dict2)
        else:
            feed_dict_pre['cost'] = self.sess.run([self.avg_loss], feed_dict2)

        return feed_dict_pre

    def load_save_model(self,pb_model_path):
        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, ['cpu_1'], pb_model_path + '/savemodel')
            sess.run(tf.global_variables_initializer())

            input_x = sess.graph.get_tensor_by_name('x:0')
            input_y = sess.graph.get_tensor_by_name('y:0')

            op = sess.graph.get_tensor_by_name('op_to_store:0')

            ret = sess.run(op, feed_dict={input_x: 5, input_y: 5})
            print(ret)
        # 只需要指定要恢复模型的 session，模型的 tag，模型的保存路径即可,使用起来更加简单


def restore_vocab(expdir, vocab_name):
    with open(expdir + vocab_name, 'rb') as f:
        v = pickle.load(f)
    # print(v.idx_to_word)
    # print(v.word_to_idx)
    # print(v.unk_symbol)
    return v

if __name__ == "__main__":
    expdir = "/home/jovyan/project/query_completion/model/dynamic_1609935328/"
    model_name = "model.bin-2000.meta"
    # # 字典恢复
    vocab_char = restore_vocab(expdir, "char_vocab.pickle")
    print(vocab_char.word_to_idx['<S>'])
    print(vocab_char.word_to_idx['连'])
    print(vocab_char.word_to_idx['帽'])

    print(vocab_char.idx_to_word[2515])
    print(vocab_char.unk_symbol)

    print("userid")
    user_vocab = restore_vocab(expdir, "user_vocab.pickle")
    print(user_vocab.word_to_idx['sw996641'])
    print(user_vocab.idx_to_word[442101])
    print(user_vocab.unk_symbol)



    query_ = ["<S>", "连", "帽", "卫", "衣", "</S>"]
    query = ''.join(query_[1:-1])
    # prefix_len = GetPrefixLen("sw996641", query, 5)  # query的前缀长度
    prefix_len = 2  # query的前缀长度
    prefix = query_[:prefix_len + 1]  # 前缀

    feed_dict_pre = {'prefix': prefix, 'user': "sw996641", "hourofday": [14], "dayofweek": [3], "query_": query_,
                     "query": query, "prefix_len": prefix_len, }

    pb_model_path = "/home/jovyan/project/query_completion/pb_model/"
    pb_model_name = "saved_model.pb"
    pb_model_serving = "saved_model_serving2"
    load_pb = Load_pb(pb_model_path, pb_model_name,vocab_char, user_vocab)
    # load_pb.run_model_pb(feed_dict_pre)
    load_pb.run_model_pb_serving(feed_dict_pre,pb_model_serving)
    # load_pb.run_model_pb_serving2(feed_dict_pre,pb_model_serving)
