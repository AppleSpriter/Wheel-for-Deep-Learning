'''
Author: Senior

查看pb文件
'''

import tensorflow as tf
from tensorflow.python.platform import gfile

model = 'terrace_mobilenetv2_96_3000_002fin.pb'
graph = tf.get_default_graph()
graph_def = graph.as_graph_def()
graph_def.ParseFromString(gfile.FastGFile(model, 'rb').read())
tf.import_graph_def(graph_def, name='graph')
summaryWriter = tf.summary.FileWriter('log/', graph)
