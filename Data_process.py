from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import os
import pickle

DIR = os.path.dirname(__file__)


class Data_process():
    def __init__(self):
        self.texts = None
        self.len_max = None
        self.word_index = None
        self.num_words = None
        self.texts_seq = None
        self.x_pad_seq = None
        self.y_pad_seq = None
        self.y_one_hot = None

    def load_data(self, len_min=0, len_max=200):
        '''
        导入数据，长度控制适用于text2seq的mode='poem'
        :param len_min: 最短长度
        :param len_max: 最长长度
        :return:
        '''
        with open(DIR + '/data/Tang_Poetry.pkl', mode='rb') as f:
            texts = pickle.load(f)
        texts = [i for i in texts if len(i) >= len_min and len(i) <= len_max]
        self.texts = texts
        self.len_min = len_min
        self.len_max = len_max

    def text2seq(self, mode='length', num_words=None, maxlen=40):
        '''
        文本转编码
        :param mode: 文本reshape方式，length以长度重新分割，poem不处理
        :param num_words: 保留词语数量
        :param maxlen: 保留文本长度
        :return:
        '''
        self.mode = mode
        texts = self.texts
        tokenizer = Tokenizer(num_words=num_words, char_level=True)
        if mode == 'poem':
            pass
        # 让样本长度保持一致提高准确率
        elif mode == 'length':
            texts_new = []
            for i in texts:
                mod = len(i) % maxlen
                i += ('E' * mod)
                for j in range(len(i) // maxlen):
                    texts_new.append(i[j * maxlen:(j * maxlen + maxlen)])
            texts = texts_new
        else:
            raise ValueError('mode should be length or poem')

        # 训练词典
        tokenizer.fit_on_texts(texts)
        word_index = tokenizer.word_index
        self.word_index = word_index
        # print('word num=', len(word_index.keys()))
        num_words = min(num_words, len(word_index.keys()) + 1)
        self.num_words = num_words

        # 转编码
        texts_seq = tokenizer.texts_to_sequences(texts=texts)
        self.texts_seq = texts_seq

        del self.texts

    def creat_one_hot(self, y, num_words):
        y_one_hot = np.zeros(shape=[len(y), num_words])
        for num, i in enumerate(y):
            y_one_hot[num, i] = 1
        return y_one_hot

    def creat_x_y(self, maxlen=40, one_hot=False):
        '''

        :param maxlen: 保留文本长度
        :return:
        '''
        self.one_hot = one_hot
        texts_seq = self.texts_seq
        x = []
        y = []
        for i in texts_seq:
            x.append(i[:-1])
            y.append(i[1:])
        # self.x = x
        # self.y = y

        n = 0
        pad_seq = []
        # 分批执行pad_sequences
        while n < len(texts_seq):
            pad_seq += list(pad_sequences(x[n:n + 5000], maxlen=maxlen,
                                          padding='post', value=0, dtype='int'))
            n += 5000
            # if n < len(texts_seq):
            #     print('finish pad_sequences %d samples(%f)' % (n, n / len(texts_seq)))
            # else:
            #     print('finish pad_sequences %d samples(1.0)' % len(texts_seq))

        pad_seq = pad_sequences(x, maxlen, padding='post', truncating='post')
        y_pad_seq = pad_sequences(y, maxlen - 1, padding='post', truncating='post')

        # 生成x和y
        self.x_pad_seq = np.array([i[:-1] for i in pad_seq])
        self.y_pad_seq = np.array([i[1:] for i in pad_seq])

        if one_hot:
            # y转one-hot
            y_one_hot = [self.creat_one_hot(i, self.num_words) for i in y_pad_seq]
            self.y_one_hot = y_one_hot

    def data_transform(self,
                       num_words=6000,
                       mode='poem',
                       len_min=0,
                       len_max=5,
                       maxlen=79,
                       one_hot=False):
        '''
        整合前面的步骤
        :param num_words:
        :param mode: poem/length
        :param len_min:
        :param len_max:
        :param maxlen:
        :return:
        '''
        self.load_data(len_min=len_min, len_max=len_max)
        self.text2seq(mode=mode, num_words=num_words)
        self.creat_x_y(maxlen=maxlen, one_hot=one_hot)
        x = np.array(self.x_pad_seq)
        if one_hot:
            y = np.array(self.y_one_hot)
        else:
            y = np.array(self.y_pad_seq)
        return x, y, self.word_index

if __name__ == '__main__':
    pass
