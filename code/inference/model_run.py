# -*- coding: utf-8 -*-


"""
==================================================
   File Name：     model_save.py
   email:         songdongdong@weidian.com
   Author :       songdongdong
   date：          2020/12/1 15:31
   Description :  
==================================================
"""
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()  ## 启用动态图机制
from helper import GetPrefixLen
from metrics import GetRankInList, MovingAvg
from beam import BeamItem, BeamQueue
import pickle
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import os

import numpy as np


def InitBeam(phrase, user_id, m):
    # Need to find the hidden state for the last char in the prefix.
    prev_hidden = np.zeros((1, 2 * m.params.num_units))
    for word in phrase[:-1]:
        feed_dict = {
            m.model.prev_hidden_state: prev_hidden,
            m.model.prev_word: [m.char_vocab[word]],
            m.model.beam_size: 4
        }
        prev_hidden = m.session.run(m.model.next_hidden_state, feed_dict)

    return prev_hidden


class Restore_Model:
    def __init__(self, expdir, model_name, vocab_char, user_vocab, ):
        self.expdir = expdir

        self.model_name = model_name
        self.vocab_char = vocab_char
        self.user_vocab = user_vocab
        self.sess = tf.Session()

    def restore_tensor(self, feed_dict_pre):
        graph = tf.get_default_graph()
        saver = tf.train.import_meta_graph(self.expdir + self.model_name, clear_devices=True)
        # saver = tf.compat.v1.train.Saver(tf.trainable_variables())
        self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])  # 这里需要初始化全局变量和局部变量
        saver.restore(self.sess, tf.train.latest_checkpoint(self.expdir))  # 输入路径即可
        # saver.restore(self.sess, os.path.join(self.expdir, model_name))
        # tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING], '../export_dir/0/')

        self.querys = graph.get_tensor_by_name("queries:0")
        self.query_length = graph.get_tensor_by_name("query_lengths:0")
        self.user_ids = graph.get_tensor_by_name("user_ids:0")
        self.hourofday = graph.get_tensor_by_name("hourofday:0")
        self.dayofweek = graph.get_tensor_by_name("dayofweek:0")
        self.keep_prob = graph.get_tensor_by_name("keep_prob:0")
        self.prev_word = graph.get_tensor_by_name("prev_word:0")
        self.beam_size = graph.get_tensor_by_name("beam_size:0")
        self.beam_chars = graph.get_tensor_by_name("beam_chars:0")
        self.next_hidden_state = graph.get_tensor_by_name("next_hidden_state:0")  # output
        self.top_k = graph.get_tensor_by_name("top_k:0")
        self.lock_op = graph.get_operation_by_name("rnn/factor_cell/lock_op")
        self.reset_user_embed = graph.get_operation_by_name("reset_user_embed")
        self.train_op = graph.get_operation_by_name("optimizer/NoOp")  # train_op
        self.avg_loss = graph.get_tensor_by_name("avg_loss:0")
        self.prev_hidden_state = graph.get_tensor_by_name("prev_hidden_state:0")
        self.feed_dict = {self.querys: feed_dict_pre['prefix'],
                          self.user_ids: [feed_dict_pre['user']],
                          self.hourofday: feed_dict_pre['hourofday'],
                          self.dayofweek: feed_dict_pre['dayofweek'],
                          self.keep_prob: 1.0,
                          self.beam_size: 8.0}
        self.feed_dict_pre = feed_dict_pre
        # for oper in graph.get_operations():
        #     print(oper)

    def save_model_pb(self, pb_model_path, pb_model_name):

        # for fixing the bug of batch norm
        gd = self.sess.graph.as_graph_def()
        # convert_variables_to_constants 需要指定output_node_names，list()，可以多个
        # graph_util.convert_variables_to_constants
        # fix nodes
        #
        # self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])  # 这里需要初始化全局变量和局部变量

        for node in gd.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            # elif node.op == 'ScatterUpdate':
            #     node.op = 'Update'
            #     if 'use_locking' in node.attr: del node.attr['use_locking']
            elif node.op == 'Assign':
                node.op = 'Identity'
                if 'use_locking' in node.attr: del node.attr['use_locking']
                if 'validate_shape' in node.attr: del node.attr['validate_shape']
                if len(node.input) == 2:
                    # input0: ref: Should be from a Variable node. May be uninitialized.
                    # input1: value: The value to be assigned to the variable.
                    node.input[0] = node.input[1]
                    del node.input[1]
            elif node.op == 'AssignAdd':
                node.op = 'Add'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            # else:
            # print(node.op, "->" ,node.name)
        # output_node_names = [n.name for n in gd.node]
        # print(output_node_names)

        # elif node.op == '':
        constant_graph = graph_util.convert_variables_to_constants(self.sess, gd,
                                                                   ["rnn/factor_cell/lock_op", 'next_hidden_state',
                                                                    "avg_loss", "beam_chars", "top_k",
                                                                    "optimizer/NoOp"]) #"reset_user_embed"
        # self.run_reset_user_embed()
        # self.Lock()
        # queue_item2 = self.getCompletion_beam(beam_size=100, branching_factor=4, stop="</S>")
        # qlist = ["".join(q.words[1:-1]) for q in reversed(list(queue_item2))]

        # print(qlist)
        # tf.train.write_graph(constant_graph, "/Users/songdongdong/PycharmProjects/query_completion/pb_model",
        #                      "model.pb", as_text=False)
        # tf.train.write_graph(constant_graph, "/Users/songdongdong/PycharmProjects/query_completion/pb_model",
        #                      "model.pb.txt", as_text=True)

        # 写入序列化的 PB 文件
        with tf.gfile.GFile(pb_model_path + pb_model_name, mode='wb') as f:
            f.write(constant_graph.SerializeToString())
        print("%d ops in the final graph." % len(gd.node))  # 2040 ops in the final graph.

        inputs = {'querys': tf.saved_model.utils.build_tensor_info(self.querys),
                  'query_length': tf.saved_model.utils.build_tensor_info(self.query_length),
                  'user_ids': tf.saved_model.utils.build_tensor_info(self.user_ids),
                  'hourofday': tf.saved_model.utils.build_tensor_info(self.hourofday),
                  'dayofweek': tf.saved_model.utils.build_tensor_info(self.dayofweek),
                  'prev_word': tf.saved_model.utils.build_tensor_info(self.prev_word),
                  'keep_prob': tf.saved_model.utils.build_tensor_info(self.keep_prob),
                  'beam_size': tf.saved_model.utils.build_tensor_info(self.beam_size),
                  # 'beam_chars': tf.saved_model.utils.build_tensor_info(self.beam_chars),
                  # 'next_hidden_state': tf.saved_model.utils.build_tensor_info(self.next_hidden_state),
                  # 'top_k': tf.saved_model.utils.build_tensor_info(self.top_k),
                  # 'lock_op': tf.saved_model.utils.build_tensor_info(self.lock_op),
                  # 'train_op': tf.saved_model.utils.build_tensor_info(self.train_op),
                  # 'avg_loss': tf.saved_model.utils.build_tensor_info(self.avg_loss),
                  # 'reset_user_embed': tf.saved_model.utils.build_tensor_info(self.reset_user_embed)

                  }

        outputs = {
            'avg_loss': tf.saved_model.utils.build_tensor_info(self.avg_loss),
            'next_hidden_state': tf.saved_model.utils.build_tensor_info(self.next_hidden_state),
            # 'reset_user_embed': tf.saved_model.build_tensor_info(self.reset_user_embed),
            # 'train_op': tf.saved_model.build_tensor_info(self.train_op),
            # 'lock_op': tf.saved_model.build_tensor_info(self.lock_op),
            'beam_chars': tf.saved_model.utils.build_tensor_info(self.beam_chars),
            'top_k': tf.saved_model.utils.build_tensor_info(self.top_k),
        }


        builder = tf.saved_model.builder.SavedModelBuilder(pb_model_path + "/saved_model_serving")
        queryCompletion_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs, tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        # 官网有误，写成了 saved_model_builder
        # 构造模型保存的内容，指定要保存的 session，特定的 tag,
        # 输入输出信息字典，额外的信息
        builder.add_meta_graph_and_variables(self.sess, tags=[tf.saved_model.tag_constants.SERVING],signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: queryCompletion_signature})

        builder.save()

    def run_reset_user_embed(self):
        self.sess.run(self.reset_user_embed,) #这里传不传值都是i无所谓的{self.user_ids: [0]}

    def Lock(self):
        # --------   Lock  --------
        if self.feed_dict[self.user_ids][0] in self.user_vocab.word_to_idx:
            user = self.user_vocab.word_to_idx[self.feed_dict[self.user_ids][0]]
        else:
            user = 0

        self.sess.run(self.lock_op, {self.user_ids: [user],
                                     self.hourofday: self.feed_dict[self.hourofday],
                                     self.dayofweek: self.feed_dict[self.dayofweek]})

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
            # print("prev_words  ",prev_words)
            print("prev_hidden shape   ", prev_hidden.shape, "  prev_hidden :    ")
            print("prev_words shape   ", prev_words.shape, " prev_words :   ",prev_words)
            # 喂到解码器里
            feed_dict = {
                self.prev_word: prev_words,
                self.prev_hidden_state: prev_hidden,
                self.beam_size: branching_factor
            }
            current_char, current_char_p, prev_hidden = self.sess.run(
                [self.beam_chars, self.top_k, self.next_hidden_state],
                feed_dict)
            current_char_p = -current_char_p
            print(current_char[0][1])
            print(current_char,current_char.shape) #(1, 4),(4, 4) ==>(?,4)
            print(current_char_p,current_char_p.shape) # (1, 4),(4, 4)==>(?,4)
            print(prev_hidden,prev_hidden.shape) #(1,1024),(4, 1024)==>(?,1024)
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

    # for oper in graph.get_operations():
    #     print(oper)


def save_vocab(filename,save_path,dic_path):
    with open(save_path+filename + '.json', 'a') as outfile:
        json.dump(dic_path, outfile, ensure_ascii=True)
        outfile.write('\n')

import json
def restore_vocab(expdir, vocab_name):
    with open(expdir + vocab_name, 'rb') as f:
        v = pickle.load(f)
    print()
    # save_vocab("char_idx_to_word","/Users/songdongdong/PycharmProjects/query_completion/pb_model/",v.idx_to_word)
    # # print(v.word_to_idx)
    # save_vocab("user_word_to_idx","/Users/songdongdong/PycharmProjects/query_completion/pb_model/",v.word_to_idx)

    print(v.unk_symbol)
    return v


class Load_pb:

    def __init__(self, pb_path_model_name, vocab_char, user_vocab):
        self.pb_path_model_name = pb_path_model_name
        self.vocab_char = vocab_char
        self.user_vocab = user_vocab

    def initBeam(self):
        phrase = self.feed_dict[self.querys]  # 前缀
        prev_hidden = np.zeros((1, 2 * 512))
        for word in phrase[:]:
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
            # print("prev_words shape   ",prev_words.shape,"  prev_words :    ",prev_words)
            # print("prev_words shape   ",prev_hidden.shape," prev_hidden :   ",prev_hidden)
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
        self.sess = tf.Session()
        # sess.run([tf.local_variables_initializer(),tf.global_variables_initializer()])
        ## read pb into graph_def
        with gfile.GFile(self.pb_path_model_name, 'rb') as f:  # gfile.FastGFile
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
        self.sess.run([tf.local_variables_initializer(), tf.global_variables_initializer()])

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
        self.reset_user_embed = self.sess.graph.get_operation_by_name("reset_user_embed")
        self.train_op = self.sess.graph.get_operation_by_name("optimizer/NoOp")  # train_op
        self.avg_loss = self.sess.graph.get_tensor_by_name("avg_loss:0")
        self.prev_hidden_state = self.sess.graph.get_tensor_by_name("prev_hidden_state:0")
        for op in self.sess.graph.get_operations():
            print(op.name, op.values())
        self.sess.run(self.reset_user_embed)

        self.feed_dict = {self.querys: feed_dict_pre['prefix'],
                          self.user_ids: [feed_dict_pre['user']],
                          self.hourofday: feed_dict_pre['hourofday'],
                          self.dayofweek: feed_dict_pre['dayofweek'],
                          self.keep_prob: 1.0,
                          self.beam_size: 8.0}

        # --------   Lock  --------
        if self.feed_dict[self.user_ids][0] in self.user_vocab.word_to_idx:
            user = user_vocab.word_to_idx[self.feed_dict[self.user_ids][0]]
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
        # print(model.train(True))


if __name__ == "__main__":
    expdir = "/Users/songdongdong/PycharmProjects/query_completion/model/dynamic_1607270485/"
    model_name = "model.bin-318000.meta"

    # 字典恢复
    vocab_char = restore_vocab(expdir, "char_vocab.pickle")
    print(vocab_char.word_to_idx['<S>'])
    print(vocab_char.word_to_idx['史'])
    print(vocab_char.word_to_idx[vocab_char.unk_symbol])
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

    # 模型恢复
    model = Restore_Model(expdir, model_name, vocab_char, user_vocab)
    model.restore_tensor(feed_dict_pre)

    model.run_reset_user_embed()
    model.Lock()
    queue_item2 = model.getCompletion_beam(beam_size=100, branching_factor=4, stop="</S>")
    qlist = ["".join(q.words[1:-1]) for q in reversed(list(queue_item2))]
    print(qlist)
    print(len(qlist))
    partial = False
    if partial and ' ' in prefix[prefix_len:]:
        word_boundary = feed_dict_pre["query"][prefix_len:].index(' ')
        query = feed_dict_pre["query"][:word_boundary + prefix_len]
    score = GetRankInList(feed_dict_pre["query"], qlist)
    feed_dict_pre['user'] = feed_dict_pre["user"]
    feed_dict_pre['score'] = score
    feed_dict_pre['top_completion'] = qlist[0]
    feed_dict_pre['hourofday'] = feed_dict_pre["hourofday"]
    feed_dict_pre['dayofweek'] = feed_dict_pre["dayofweek"]
    feed_dict_pre['prefix_len'] = int(prefix_len)

    print(model.train(True))

    pb_model_path = "/Users/songdongdong/PycharmProjects/query_completion/pb_model/"
    pb_model_name = "saved_model.pb"
    model.save_model_pb(pb_model_path, pb_model_name)
    load_pb = Load_pb(pb_model_path + pb_model_name, vocab_char, user_vocab)
    # load_pb.run_model_pb(feed_dict_pre)

    # graph = load_pb.run_model_pb2(feed_dict_pre)
    # for op in graph.get_operations():
    #     print(op.name, op.values())
    #
    #     # 输入
    # querys =graph.get_tensor_by_name("queries:0")
    # query_length =graph.get_tensor_by_name("query_lengths:0")
    # user_ids = graph.get_tensor_by_name("user_ids:0")
    # hourofday =graph.get_tensor_by_name("hourofday:0")
    # dayofweek = graph.get_tensor_by_name("dayofweek:0")
    # keep_prob = graph.get_tensor_by_name("keep_prob:0")
    # prev_word = graph.get_tensor_by_name("prev_word:0")
    # beam_size = graph.get_tensor_by_name("beam_size:0")
    # beam_chars = graph.get_tensor_by_name("beam_chars:0")
    # next_hidden_state = graph.get_tensor_by_name("next_hidden_state:0")  # output
    # top_k = graph.get_tensor_by_name("top_k:0")
    # lock_op = graph.get_operation_by_name("rnn/factor_cell/lock_op")
    # # reset_user_embed = graph.get_operation_by_name("reset_user_embed")
    # train_op = graph.get_operation_by_name("optimizer/NoOp")  # train_op
    # avg_loss = graph.get_tensor_by_name("avg_loss:0")
    # prev_hidden_state = graph.get_tensor_by_name("prev_hidden_state:0")
    #
    # feed_dict = {querys: feed_dict_pre['prefix'],
    #              user_ids: [feed_dict_pre['user']],
    #              hourofday: feed_dict_pre['hourofday'],
    #              dayofweek: feed_dict_pre['dayofweek'],
    #              keep_prob: 1.0,
    #              beam_size: 8.0}
    #
    # with tf.Session(graph=graph) as sess:
    #     # sess.run(reset_user_embed)
    #
    #
    #     # --------   Lock  --------
    #     if feed_dict[user_ids][0] in user_vocab.word_to_idx:
    #         user = user_vocab.word_to_idx[feed_dict[user_ids][0]]
    #     else:
    #         user = 0
    #
    #     sess.run(lock_op, {user_ids: [user],
    #                                  hourofday: feed_dict[hourofday],
    #                                  dayofweek: feed_dict[dayofweek]})
    #
    #     # queue_item2 = getCompletion_beam(beam_size=100, branching_factor=4, stop="</S>")
    #     # qlist = ["".join(q.words[1:-1]) for q in reversed(list(queue_item2))]
    #     # print(qlist)
    #     # partial = False
    #     # if partial and ' ' in prefix[prefix_len:]:
    #     #     word_boundary = feed_dict_pre["query"][prefix_len:].index(' ')
    #     #     query = feed_dict_pre["query"][:word_boundary + prefix_len]
    #     # score = GetRankInList(feed_dict_pre["query"], qlist)
    #     # feed_dict_pre['user'] = feed_dict_pre["user"]
    #     # feed_dict_pre['score'] = score
    #     # feed_dict_pre['top_completion'] = qlist[0]
    #     # feed_dict_pre['hourofday'] = feed_dict_pre["hourofday"]
    #     # feed_dict_pre['dayofweek'] = feed_dict_pre["dayofweek"]
    #     # feed_dict_pre['prefix_len'] = int(prefix_len)
    #     #
    #     # print(model.train(True))
    # # load_pb.run_model_pb(feed_dict_pre)
