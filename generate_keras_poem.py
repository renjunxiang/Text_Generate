import os
import numpy as np
import pickle
from keras.models import load_model
import re

DIR = os.path.dirname(os.path.abspath(__file__))


def generate_text(model=None,
                  word_index=None,
                  # correct=True
                  ):
    '''
    生成文本
    :param model: 训练好的模型
    :param start_word: 起始文字
    :param word_index: 词典索引
    :return:
    '''
    while True:
        try:
            print('中文作诗，作诗前请确保有模型。输入开头，quit=离开；\n请输入命令：')

            start_word = input()
            if start_word == 'quit':
                break
            if start_word == '':
                words = list(word_index.keys())
                # 随机初始不能是标点和终止符
                for i in ['。', '？', '！', 'E']:
                    words.remove(i)
                start_word = np.random.choice(words, 1)

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
    file='poem'
    with open(DIR + '/model/%s/word_index.pkl'%file, mode='rb') as f:
        word_index = pickle.load(f)
    model = load_model(DIR + '/model/%s/model_keras.h5'%file)
    generate_text(model=model, word_index=word_index)
