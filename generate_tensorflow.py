import os
import numpy as np
import tensorflow as tf
from Data_process import Data_process
from rnn import model_tensorflow
import re

DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


def generate(maxlen=60,
             batchsize=1,
             num_words=3000,
             num_units=128,
             num_layers=2,
             correct=True):
    '''
    
    :param maxlen: 
    :param batchsize: 
    :param num_words: 
    :param num_units: 
    :param num_layers: 
    :param correct: 用于修正长时间不出现标点，会强制从候选中选择标点
    :return: 
    '''
    data_process = Data_process()
    x, y, word_index = data_process.data_transform(num_words=num_words,
                                                   mode='length',
                                                   len_min=0,
                                                   len_max=100,
                                                   maxlen=maxlen,
                                                   one_hot=False)

    input_data = tf.placeholder(tf.int32, [None, None])
    output_targets = tf.placeholder(tf.int32, [None, None])

    tensors = model_tensorflow(input_data=input_data,
                               output_targets=output_targets,
                               num_words=num_words,
                               num_units=num_units,
                               num_layers=num_layers,
                               batchsize=batchsize)
    saver = tf.train.Saver(tf.global_variables())
    initializer = tf.global_variables_initializer()
    while True:
        with tf.Session() as sess:
            sess.run(initializer)
            checkpoint = tf.train.latest_checkpoint(DIR + '/model/')
            saver.restore(sess, checkpoint)

            try:
                print('中文作诗，作诗前请确保有模型。输入开头，quit=离开；\n请输入命令：')

                start_word = input()
                if start_word == 'quit':
                    break

                print('开始创作')
                input_index = []
                for i in start_word:
                    index_next = word_index[i]
                    input_index.append(index_next)
                input_index = input_index[:-1]

                # 用于修正标点位置
                punctuation = [word_index['，'], word_index['。'], word_index['？']]
                punctuation_index = len(start_word)

                while index_next not in [0, word_index['E']]:
                    input_index.append(index_next)
                    [y_predict, last_state] = sess.run([tensors['prediction'], tensors['last_state']],
                                                       feed_dict={input_data: np.array([input_index])})
                    y_predict = y_predict[-1]
                    y_predict = {num: i for num, i in enumerate(y_predict)}
                    index_max = sorted(y_predict, key=lambda x: y_predict[x], reverse=True)[:8]
                    index_next = np.random.choice(index_max)
                    punctuation_index += 1
                    if correct:
                        # [3,7]之间个字符出现标点正常，重置索引
                        if index_next in punctuation and punctuation_index > 3 and punctuation_index < 8:
                            punctuation_index = 0
                        # 当超过7个字符没有出现标点，且标点出现在候选中，选择标点
                        elif punctuation_index >= 8:
                            punctuation_index = 0
                            while (set(punctuation) & set(index_max)) and (index_next not in punctuation):
                                index_next = np.random.choice(index_max)
                        # 当少于3个字符出现标点，选择文字
                        elif punctuation_index <= 3:
                            while index_next in punctuation:
                                index_next = np.random.choice(index_max)
                        else:
                            pass

                    if len(input_index) > 100:
                        break
                index_word = {word_index[i]: i for i in word_index}
                text = [index_word[i] for i in input_index]
                text = ''.join(text)
            except Exception as e:
                print(e)
                text = '不能识别%s' % start_word
            finally:
                text_list = re.findall(pattern='[^。？！]*[。？！]', string=text)
                print('作诗完成：\n')
                for i in text_list:
                    print(i)
                print('\n------------我是分隔符------------\n')


if __name__ == '__main__':
    generate(maxlen=100, batchsize=1, num_words=20000,
             num_units=128, num_layers=2, correct=True)
