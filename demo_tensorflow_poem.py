from generate_text import train_tensorflow, generate_tensorflow
import os
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
DIR = os.path.dirname(os.path.abspath(__file__))

# train_tensorflow(mode='length', file='poem',
#                  len_min=10, len_max=50,
#                  maxlen=50, num_words=20000,
#                  num_units=128, num_layers=2,
#                  batchsize=64, epochs=5,
#                  process_path=DIR + '/model/poem/poem.pkl',
#                  model_path=DIR + '/model/poem/')

# 不加这句会出现VarScope冲突
tf.reset_default_graph()

generate_tensorflow(process_path=DIR + '/model/poem/poem.pkl',
                    model_path=DIR + '/model/poem/',
                    maxlen=100)
