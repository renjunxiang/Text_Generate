from generate_text import train_keras, generate_keras
import os

DIR = os.path.dirname(os.path.abspath(__file__))

train_keras(maxlen=40,
            batchsize=64,
            num_words=5000,
            num_units=128,
            epochs=1,
            mode='length',
            file='poem',
            len_min=0,
            len_max=100,
            one_hot=False,
            process_path=DIR + '/model/poem/keras_process.pkl',
            model_path=DIR + '/model/poem/keras.h5')

generate_keras(process_path=DIR + '/model/poem/keras_process.pkl',
               model_path=DIR + '/model/poem/keras.h5',
               maxlen=100)
