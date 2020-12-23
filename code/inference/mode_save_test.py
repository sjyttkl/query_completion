# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     mode_save_test.py
   email:         songdongdong@weidian.com
   Author :       songdongdong
   date：          2020/12/1 16:09
   Description :  
==================================================
"""
import tensorflow.compat.v1 as tf

#
expdir = "/Users/songdongdong/PycharmProjects/query_completion/model/1605774995/"
# metamodel = MetaModel("/Users/songdongdong/PycharmProjects/query_completion/model/1605774995")
# model = metamodel.model
# metamodel.MakeSessionAndRestore(1)
#
# graph = tf.Graph()
# config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=1,
#                                   intra_op_parallelism_threads=1)
# with graph.as_default():
#     session = tf.compat.v1.Session(config=config)
#     # saver = tf.compat.v1.train.Saver(tf.trainable_variables())
#     session.run(tf.compat.v1.global_variables_initializer())
#     W = tf.get_variable('W', [536, 3 * 512],
#                         initializer=tf.constant_initializer(0.0, tf.float32))  # 536 ,3*512
#     lockedW = tf.Variable(tf.zeros_like(W), name='lockedW',
#                           collections=[tf.GraphKeys.LOCAL_VARIABLES],  # 存储在 局部变量中，不能进行分布式共享
#                           trainable=False)
#
#     print(W)
#     print(lockedW)


print("beam {0:.3f}: ".format(0.323232) +"<S> 史")

from tensorflow.python.ops.control_flow_ops import cond
def add():
    print("add")
    x = a + b
    return x

def last():
    print("last")
    return x

with tf.Session() as s:
    a = tf.Variable(tf.constant(0.),name="a")
    b = tf.Variable(tf.constant(0.),name="b")
    x = tf.constant(-1.)
    calculate= cond(x.eval()==-1.,add,last)
    val = s.run([calculate], {a: 1., b: 2.})
    print(val) # 3
    print(s.run([calculate],{a:3.,b:4.})) # 7
    print(val) # 3