from Data_process import Data_process
from rnn import model_keras
import os
import pickle

DIR = os.path.dirname(os.path.abspath(__file__))


def train(maxlen=40,
          batchsize=64,
          num_words=3000,
          num_units=128,
          epochs=1):
    data_process = Data_process()
    x, y, word_index = data_process.data_transform(num_words=num_words,
                                                   mode='length',
                                                   len_min=5,
                                                   len_max=100,
                                                   maxlen=maxlen,
                                                   one_hot=True)
    with open(DIR + '/model/word_index.pkl', mode='wb') as f:
        pickle.dump(word_index, f)
    model = model_keras(num_words=num_words, num_units=num_units)
    model.fit(x=x, y=y, epochs=epochs, batch_size=batchsize, verbose=1)
    model.save(DIR + '/model/model_keras.h5')


if __name__ == '__main__':
    train(maxlen=40, batchsize=64, num_words=3000,
          num_units=128, epochs=2)
