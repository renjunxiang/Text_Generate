import os
import numpy as np
import tensorflow as tf
from .rnn import model_tensorflow
import re
import pickle

DIR = os.path.dirname(os.path.abspath(__file__))


def generate_tensorflow(process_path=DIR + '/model/poem/poem.pkl',
                        model_path=DIR + '/model/poem/train',
                        maxlen=80,
                        newline=True
                        ):
    '''

    :param process_path: 训预处理模型路径
    :param model_path: 网络参数路径
    :param maxlen: maxlen创作最大长度
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
                               num_units=data_process.num_units,
                               num_layers=data_process.num_layers,
                               batchsize=1)
    saver = tf.train.Saver(tf.global_variables())
    initializer = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(initializer)
        checkpoint = tf.train.latest_checkpoint(model_path)
        saver.restore(sess, checkpoint)
        while True:
            print('创作前请确保有模型。输入开头，quit=离开；\n请输入命令：')

            start_word = input()
            if start_word == 'quit':
                print('\n再见！')
                break

            if start_word == '':
                words = list(word_index.keys())
                # 随机初始不能是标点和终止符
                for i in ['。', '？', '！', 'E']:
                    words.remove(i)
                start_word = np.random.choice(words, 1)

            try:
                print('开始创作')
                input_index = []
                for i in start_word:
                    index_next = word_index[i]
                    input_index.append(index_next)
                input_index = input_index[:-1]

                # 原则上不会出现0,保险起见还是加上去
                while index_next not in [0, word_index['E']]:
                    input_index.append(index_next)
                    y_predict = sess.run(tensors['prediction'],
                                         feed_dict={input_data: np.array([input_index])})
                    y_predict = y_predict[-1]
                    index_next = np.random.choice(np.arange(len(y_predict)), p=y_predict)

                    if len(input_index) > maxlen:
                        break
                index_word = {word_index[i]: i for i in word_index}
                text = [index_word[i] for i in input_index]
                text = ''.join(text)
            except Exception as e:
                print(e)
                text = '不能识别%s' % start_word
            finally:
                print('创作完成：\n')
                if newline:
                    text_list = re.findall(pattern='[^。？！]*[。？！]', string=text)
                    for i in text_list:
                        print(i)
                else:
                    print(text)
                print('\n------------我是分隔符------------\n')


if __name__ == '__main__':
    generate_tensorflow(process_path=DIR + '/model/poem/poem.pkl',
                        model_path=DIR + '/model/poem/train',
                        maxlen=100)
