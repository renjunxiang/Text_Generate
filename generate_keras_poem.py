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
                y_predict = model.predict(np.array([input_index]))
                y_predict = y_predict[0][-1]
                # y_predict = {num: i for num, i in enumerate(y_predict)}
                # index_max = sorted(y_predict, key=lambda x: y_predict[x], reverse=True)[:10]
                index_next = np.random.choice(np.arange(len(y_predict)), p=y_predict)
                punctuation_index += 1
                # if correct:
                #     # [3,7]之间个字符出现标点正常，重置索引
                #     if index_next in punctuation and punctuation_index > 3 and punctuation_index < 8:
                #         punctuation_index = 0
                #     # 当超过7个字符没有出现标点，且标点出现在候选中，选择标点
                #     elif punctuation_index >= 8:
                #         punctuation_index = 0
                #         while (set(punctuation) & set(index_max)) and (index_next not in punctuation):
                #             index_next = np.random.choice(index_max)
                #     # 当少于3个字符出现标点，选择文字
                #     elif punctuation_index <= 3:
                #         while index_next in punctuation:
                #             index_next = np.random.choice(index_max)
                #     else:
                #         pass

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
