import os
import numpy as np
import pickle
from keras.models import load_model
import re


def generate_keras(process_path='./model/poem/data_process.pkl',
                   model_path='./model/poem/',
                   maxlen=100
                   ):
    '''
    生成文本
    :param process_path: 训预处理模型路径
    :param model_path: 网络参数路径
    :param maxlen: maxlen创作最大长度
    :return:
    '''
    with open(process_path, mode='rb') as f:
        data_process = pickle.load(f)
    word_index = data_process.word_index
    model = load_model(model_path)

    while True:
        print('中文作诗，作诗前请确保有模型。输入开头，quit=离开；\n请输入命令：')

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
                y_predict = model.predict(np.array([input_index]))
                y_predict = y_predict[0][-1]
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
            text_list = re.findall(pattern='[^。？！]*[。？！]', string=text)
            print('作诗完成：\n')
            for i in text_list:
                print(i)
            print('\n------------我是分隔符------------\n')


