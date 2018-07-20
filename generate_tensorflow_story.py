import os
import numpy as np
import tensorflow as tf
from Data_process import Data_process
from rnn import model_tensorflow
import re
import pickle

DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


def generate(batchsize=1,
             num_units=128,
             num_layers=2,
             file='story',
             process_path=DIR + '/model/story/story.pkl'
             ):
    '''
    
    :param batchsize: 训练时候每个batch的大小
    :param num_units: rnn神经元数量
    :param num_layers: rnn层数
    :param file: 导入的数据
    :return:
    '''
    with open(process_path, mode='rb') as f:
        data_process = pickle.load(f)
    word_index = data_process.word_index

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
    while True:
        with tf.Session() as sess:
            sess.run(initializer)
            checkpoint = tf.train.latest_checkpoint(DIR + '/model/%s/' % file)
            saver.restore(sess, checkpoint)

            try:
                print('生成小说，写作前请确保有模型。输入开头或随机，quit=离开；\n请输入命令：')

                start_word = input()
                if start_word == 'quit':
                    break
                if start_word == '':
                    start_word = np.random.choice(list(word_index.keys()), 1)
                print('开始创作')
                input_index = []
                for i in start_word:
                    index_next = word_index[i]
                    input_index.append(index_next)
                input_index = input_index[:-1]

                # 用于修正标点位置
                punctuation = [word_index['，'], word_index['。'], word_index['？']]
                punctuation_index = len(start_word)

                # while index_next not in [0, word_index['E']]:
                while index_next not in [0, word_index['E']]:
                    input_index.append(index_next)
                    [y_predict, last_state] = sess.run([tensors['prediction'], tensors['last_state']],
                                                       feed_dict={input_data: np.array([input_index])})
                    y_predict = y_predict[-1]
                    index_next = np.random.choice(np.arange(len(y_predict)), p=y_predict)
                    punctuation_index += 1
                    if len(input_index) > 200:
                        break
                index_word = {word_index[i]: i for i in word_index}
                text = [index_word[i] for i in input_index]
                text = ''.join(text)
            except Exception as e:
                print(e)
                text = '不能识别%s' % start_word
            finally:
                # text_list = re.findall(pattern='[^。？！]*[。？！]', string=text)
                print('小说完成：\n')
                print(text)
                print('\n------------我是分隔符------------\n')


if __name__ == '__main__':
    generate(num_units=128, num_layers=2,
             file='story', batchsize=1)
