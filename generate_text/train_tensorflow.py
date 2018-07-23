import os
import numpy as np
import tensorflow as tf
from . import Data_process
from .rnn import model_tensorflow
import pickle

DIR = os.path.dirname(os.path.abspath(__file__))


def train_tensorflow(maxlen=40,
                     batchsize=64,
                     num_words=4000,
                     num_units=128,
                     num_layers=2,
                     epochs=1,
                     mode='length',
                     file='poem',
                     len_min=0,
                     len_max=10e8,
                     cut=False,
                     process_path=DIR + '/model/poem/data_process.pkl',
                     model_path=DIR + '/model/poem/train'
                     ):
    '''

    :param maxlen: 每句话的长度,拆分为x和y后长度-1
    :param batchsize: 训练时候每个batch的大小
    :param num_words: 词典大小
    :param num_units: rnn神经元数量
    :param num_layers: rnn层数
    :param epochs: epoch数量
    :param mode: 拆分数据集的方式,length按固定长度,sample按样本
    :param file: 导入的数据
    :param len_min: 导入数据后句子被保留的最小长度
    :param len_max: 导入数据后句子被保留的最大长度
    :param cut: 是否分词,诗歌不要分词,小说建议分词
    :return:
    '''
    data_process = Data_process()
    x, y, word_index = data_process.data_transform(num_words=num_words,
                                                   mode=mode,
                                                   len_min=len_min,
                                                   len_max=len_max,
                                                   maxlen=maxlen,
                                                   one_hot=False,
                                                   file=file,
                                                   cut=cut)
    # 偷懒直接加这里面
    data_process.num_units = num_units
    data_process.num_layers = num_layers

    with open(process_path, mode='wb') as f:
        pickle.dump(data_process, f)
    input_data = tf.placeholder(tf.int32, [None, None])
    output_targets = tf.placeholder(tf.int32, [None, None])

    tensors = model_tensorflow(input_data=input_data,
                               output_targets=output_targets,
                               num_words=data_process.num_words,
                               num_units=num_units,
                               num_layers=num_layers,
                               batchsize=batchsize)
    saver = tf.train.Saver(tf.global_variables())
    initializer = tf.global_variables_initializer()
    print('start training')
    with tf.Session() as sess:
        sess.run(initializer)
        for epoch in range(epochs):
            for batch in range(len(x) // batchsize * 2):
                index_all = np.arange(len(x))
                index_batch = np.random.choice(index_all, batchsize)
                x_batch = x[index_batch]
                y_batch = y[index_batch]
                print(x_batch.shape)
                loss, _, _ = sess.run([
                    tensors['total_loss'],
                    tensors['last_state'],
                    tensors['train_op']
                ], feed_dict={input_data: x_batch, output_targets: y_batch})
                print('Epoch: %d, batch: %d, training loss: %.6f' % (epoch + 1, batch + 1, loss))
            if epoch % 1 == 0:
                saver.save(sess, model_path, global_step=epoch)

if __name__ == '__main__':
    train_tensorflow(mode='length', file='poem',
                     len_min=10, len_max=50,
                     maxlen=50, num_words=20000,
                     num_units=128, num_layers=2,
                     batchsize=64, epochs=1,
                     process_path=DIR + '/model/poem/poem.pkl',
                     model_path=DIR + '/model/poem/train')