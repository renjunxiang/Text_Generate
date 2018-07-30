from . import Data_process
from .rnn import model_keras
import os
import pickle
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))


def train_keras(maxlen=40,
                batchsize=64,
                num_words=3000,
                num_units=128,
                epochs=1,
                mode='length',
                file='poem',
                len_min=0,
                len_max=100,
                one_hot=False,
                process_path=DIR + '/model/poem/data_process.pkl',
                model_path=DIR + '/model/poem/keras.h5'):
    data_process = Data_process()
    x, y, word_index = data_process.data_transform(num_words=num_words,
                                                   mode=mode,
                                                   file=file,
                                                   len_min=len_min,
                                                   len_max=len_max,
                                                   maxlen=maxlen,
                                                   one_hot=one_hot)
    y = np.expand_dims(y, -1)
    with open(process_path, mode='wb') as f:
        pickle.dump(data_process, f)
    model = model_keras(num_words=data_process.num_words, num_units=num_units)
    for epoch in range(epochs):
        model.fit(x=x, y=y, epochs=1, batch_size=batchsize, verbose=1)
        model.save(model_path)
