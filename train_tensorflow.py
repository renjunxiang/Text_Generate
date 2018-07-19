import os
import numpy as np
import tensorflow as tf
from Data_process import Data_process
from rnn import model_tensorflow

DIR = os.path.dirname(os.path.abspath(__file__))


def train(maxlen=40,
          batchsize=64,
          num_words=4000,
          num_units=128,
          num_layers=2,
          epochs=1):
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
    print('start training')
    with tf.Session() as sess:
        sess.run(initializer)
        for epoch in range(epochs):
            for batch in range(1000):
                index_all = np.arange(len(x))
                index_batch = np.random.choice(index_all, batchsize)
                x_batch = x[index_batch]
                y_batch = y[index_batch]
                print(x_batch.shape)
                loss, _, _ = sess.run([
                    tensors['total_loss'],
                    tensors['last_state'],
                    tensors['train_op']
                ], feed_dict={input_data: x_batch, output_targets: y_batch})
                print('Epoch: %d, batch: %d, training loss: %.6f' % (epoch + 1, batch + 1, loss))
            if epoch % 1 == 0:
                saver.save(sess, DIR+'/model/train', global_step=epoch)


if __name__ == '__main__':
    train(maxlen=24, batchsize=64, num_words=5000,
          num_units=128, num_layers=2, epochs=1)
