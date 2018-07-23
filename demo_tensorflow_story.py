from generate_text import train_tensorflow, generate_tensorflow
import os
import tensorflow as tf

DIR = os.path.dirname(os.path.abspath(__file__))

# train_tensorflow(mode='length', file='story',
#                  len_min=10, len_max=200,
#                  maxlen=200, num_words=20000,
#                  num_units=128, num_layers=2,
#                  batchsize=64, epochs=15,
#                  cut=True,
#                  process_path=DIR + '/model/story/story.pkl',
#                  model_path=DIR + '/model/story/')

# 不加这句会出现VarScope冲突
tf.reset_default_graph()

generate_tensorflow(process_path=DIR + '/model/story/story.pkl',
                    model_path=DIR + '/model/story/',
                    maxlen=300,
                    newline=False)
