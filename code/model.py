import os
# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
from factorcell import FactorCell
from vocab import Vocab
import helper


class MetaModel:
    """Helper class for loading models. and eval"""

    def __init__(self, expdir):
        # load params ......
        self.expdir = expdir
        self.params = helper.GetParams(os.path.join(expdir, 'char_vocab.pickle'), 'eval',
                                       expdir)
        # mapping of characters to indices
        self.char_vocab = Vocab.Load(os.path.join(expdir, 'char_vocab.pickle'))
        # mapping of user ids to indices
        self.user_vocab = Vocab.Load(os.path.join(expdir, 'user_vocab.pickle'))
        self.params.vocab_size = len(self.char_vocab)
        self.params.user_vocab_size = len(self.user_vocab)

        # construct the tensorflow graph
        # self.graph = tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.model = Model(self.params, training_mode=False)
            self.char_tensor = tf.constant(self.char_vocab.GetWords(), name='char_tensor')  # 获得一批vocabulary (75,?)
            self.beam_chars = tf.nn.embedding_lookup(self.char_tensor,
                                                     self.model.selected)  # 通过beam 搜索超出的 结果selected，然后又再次 查表获取chars

    def Lock(self, user_id=0):
        """Locking precomputes the adaptation for a given user."""
        self.session.run(self.model.decoder_cell.lock_op,
                         {self.model.user_ids: [user_id],
                          self.model.hourofday: [3],
                          self.model.dayofweek: [4]})

    def MakeSession(self, threads=8):
        """Create the session with the given number of threads."""
        config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=threads,
                                          intra_op_parallelism_threads=threads)
        with self.graph.as_default():
            self.session = tf.compat.v1.Session(config=config)

    # 模型恢复
    def Restore(self):
        """Initialize all variables and restore model from disk."""
        with self.graph.as_default():
            saver = tf.compat.v1.train.Saver(tf.trainable_variables())
            self.session.run(tf.compat.v1.global_variables_initializer())
            saver.restore(self.session, os.path.join(self.expdir, 'model.bin-0'))

    def MakeSessionAndRestore(self, threads=8):
        self.MakeSession(threads)
        self.Restore()


class Model(object):
    """Defines the Tensorflow graph for training and testing a model."""

    def __init__(self, params, training_mode=True, optimizer=tf.compat.v1.train.AdamOptimizer,
                 learning_rate=0.001):  # tf.train.AdamOptimizer
        self.params = params
        opt = optimizer(learning_rate)
        self.BuildGraph(params, training_mode=training_mode, optimizer=opt)
        if not training_mode:
            # pass;
            self.BuildDecoderGraph()

    def BuildGraph(self, params, training_mode=True, optimizer=None):
        self.queries = tf.compat.v1.placeholder(tf.int32, [None, params.max_len], name='queries')
        self.query_lengths = tf.compat.v1.placeholder(tf.int32, [None], name='query_lengths')
        self.user_ids = tf.compat.v1.placeholder(tf.int32, [None], name='user_ids')

        x = self.queries[:, :-1]  # strip off the end of query token,剔除最后一个字符
        y = self.queries[:, 1:]  # need to predict y from x ，剔除第一个字符

        self.char_embeddings = tf.compat.v1.get_variable(
            'char_embeddings',
            [params.vocab_size, params.char_embed_size])  # 初始化参数 字符embedding   [vocab_size，char_embed_size=24]
        self.char_bias = tf.compat.v1.get_variable('char_bias', [params.vocab_size])  # 初始化参数   vocab_size
        self.user_embed_mat = tf.compat.v1.get_variable(  # this defines the user embeddings
            'user_embed_mat',
            [params.user_vocab_size, params.user_embed_size])  # 初始化 用户 user_vocab_size,user_embed_size:32

        inputs = tf.nn.embedding_lookup(self.char_embeddings, x)  # 查询lookup

        # create a mask to zero out the loss for the padding tokens
        indicator = tf.sequence_mask(self.query_lengths - 1, params.max_len - 1)  # mask
        _mask = tf.where(indicator, tf.ones_like(x, dtype=tf.float32),
                         tf.zeros_like(x, dtype=tf.float32))  # 实际长度为1， 超过实际长度为0

        # drop_out
        self.dropout_keep_prob = tf.compat.v1.placeholder_with_default(1.0, (), name='keep_prob')
        # 用户embed矩阵
        user_embeddings = tf.nn.embedding_lookup(self.user_embed_mat, self.user_ids)

        self.use_time_features = False
        if hasattr(params, 'use_time_features') and params.use_time_features:  # 判断是否开启 时间序列
            self.use_time_features = True
            self.dayofweek = tf.compat.v1.placeholder(tf.int32, [None], name='dayofweek')  # 星期几 1-7
            self.hourofday = tf.compat.v1.placeholder(tf.int32, [None], name='hourofday')  # 小时 0-24
            self.day_embed_mat = tf.compat.v1.get_variable('day_embed_mat', [7, 2])  # 初始化 7行 2列 没天用 2维的数据进行表示
            self.hour_embed_mat = tf.compat.v1.get_variable('hour_embed_mat', [24, 3])  ##初始化 24行 3列

            hour_embeds = tf.nn.embedding_lookup(self.hour_embed_mat, self.hourofday)
            day_embeds = tf.nn.embedding_lookup(self.day_embed_mat, self.dayofweek)

            user_embeddings = tf.concat(axis=1, values=[user_embeddings, hour_embeds,
                                                        day_embeds])  # 制作 用户embedding数据，在user_embeddings纬度 后面直接加入hour_embeds ,day_embeds []
            # [user_vocab_size,user_embed_size:] 加入：[7, 2]，[24, 3] ->> [user_vocab_size, 37]
        with tf.compat.v1.variable_scope('rnn'):
            self.decoder_cell = FactorCell(params.num_units, params.char_embed_size,  # 512 ,24,
                                           user_embeddings,  # 32
                                           bias_adaptation=params.use_mikolov_adaptation,
                                           # 这个是mikolov adaptation model ConcatCell，
                                           lowrank_adaptation=params.use_lowrank_adaptation,  # 这个是  FactorCell
                                           rank=params.rank,  # 24
                                           layer_norm=params.use_layer_norm,  # 层归一化
                                           dropout_keep_prob=self.dropout_keep_prob)  # drop out
            # [batch_size, max_time, cell_state_size]
            outputs, _ = tf.compat.v1.nn.dynamic_rnn(self.decoder_cell, inputs,
                                                     sequence_length=self.query_lengths,
                                                     dtype=tf.float32)
            # reshape outputs to 2d before passing to the output layer
            reshaped_outputs = tf.reshape(outputs, [-1,
                                                    params.num_units])  # 直接转换成 [-1,num_units] 其实就是[batch_size, max_time, cell_state_size] --> [batch_size*max_time, cell_state_size]
            projected_outputs = tf.layers.dense(reshaped_outputs, params.char_embed_size,
                                                name='proj')  # 这里会改变最后 reshape_outputs 最后一纬度。比如：[batch_size*max_time, cell_state_size] ->[batch_size*max_time, char_embed_size:24]

            reshaped_logits = tf.matmul(projected_outputs, self.char_embeddings,
                                        transpose_b=True) + self.char_bias  # 最后继续做个matmul ，由于 transponse_b=True 则b 在乘法之前转置
            #  [batch_size*max_time, char_embed_size] * [char_embed_size=24,vocab_size]   +  vocab_size ,  =

        reshaped_labels = tf.reshape(y, [-1])  # 变成 一维向量
        reshaped_mask = tf.reshape(_mask, [-1])  # 变成一维向量 ，已经mask 的效果：实际长度为1， 超过实际长度为0

        reshaped_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=reshaped_logits, labels=reshaped_labels)  # 计算logits 和 labels 之间的稀疏softmax 交叉熵
        masked_loss = tf.multiply(reshaped_loss, reshaped_mask)  # 真实的长度保留，超过的部分都变成 0

        # reshape the loss back to the input size in order to compute
        # the per sentence loss
        self.per_word_loss = tf.reshape(masked_loss, tf.shape(x))  # 纬度恢复，每个字的loss
        self.per_sentence_loss = tf.compat.v1.div(tf.reduce_sum(self.per_word_loss, 1),
                                                  tf.reduce_sum(_mask, 1))  # 每个预测结果进行相加，在除以 ，mask真实结果/每个句子损失值
        self.per_sentence_loss = tf.reduce_sum(self.per_word_loss, 1)

        total_loss = tf.reduce_sum(masked_loss)  # 整体损失
        self.words_in_batch = tf.compat.v1.to_float(tf.reduce_sum(self.query_lengths - 1))
        self.avg_loss = total_loss / self.words_in_batch

        if training_mode:
            self.train_op = optimizer.minimize(self.avg_loss)  # 还是使用均值loss进行迭代

    def BuildDecoderGraph(self):
        """This part of the graph is only used for evaluation.

        It computes just a single step of the LSTM.
        """
        self.prev_word = tf.compat.v1.placeholder(tf.int32, [None], name='prev_word')  # 前一个字
        self.prev_hidden_state = tf.compat.v1.placeholder(tf.float32, [None, 2 * self.params.num_units],
                                                          name='prev_hidden_state')  # 隐藏单元 [None,512 + 512 ]

        prev_c = self.prev_hidden_state[:, :self.params.num_units]  # [None,: 512] 分成 cell
        prev_h = self.prev_hidden_state[:, self.params.num_units:]  # [None ,512:] 分成 output

        # temperature can be used to tune diversity of the decoding
        self.temperature = tf.compat.v1.placeholder_with_default([1.0], [1])

        prev_embed = tf.nn.embedding_lookup(self.char_embeddings, self.prev_word)  # 前一个字的embedding

        state = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(prev_c,
                                                        prev_h)  # LSTMStateTuple 于存储LSTM单元的cell_state,output 变成 lstm，
        result, (next_c, next_h) = self.decoder_cell(prev_embed, state,
                                                     use_locked=True)  # 一个输入和一个state ，输出是一个 lstm的结果一样
        self.next_hidden_state = tf.concat([next_c, next_h], 1)  # 按照第二位 进行拼接

        with tf.compat.v1.variable_scope('rnn', reuse=True):
            proj_result = tf.compat.v1.layers.dense(result, self.params.char_embed_size, reuse=True,
                                                    name='proj')  # result 的最后一个纬度变成 char_embed_size (None,24)
        logits = tf.matmul(proj_result, self.char_embeddings, transpose_b=True) + self.char_bias  # (?,75)

        prevent_unk = tf.one_hot([0], self.params.vocab_size, -30.0)  # (1,75)

        self.next_prob = tf.nn.softmax(prevent_unk + logits / self.temperature)  # (None,75)

        self.next_log_prob = tf.nn.log_softmax(logits / self.temperature)  # (None ,75)

        # return the top `beam_size` number of characters for use in decoding
        self.beam_size = tf.compat.v1.placeholder_with_default(1, (), name='beam_size')

        log_probs, self.selected = tf.nn.top_k(self.next_log_prob, self.beam_size)  # (None,None)  (None,None)

        self.selected_p = -log_probs  # cost is the negative log likelihood
