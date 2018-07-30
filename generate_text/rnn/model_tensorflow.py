import tensorflow as tf


def model_tensorflow(input_data=None,
                     output_targets=None,
                     num_words=3000,
                     num_units=128,
                     num_layers=2,
                     batchsize=64):
    '''

    :param input_data:
    :param output_targets:
    :param num_words:
    :param num_units:
    :param num_layers:
    :param batchsize: 1代表生成，大于1代表训练
    :return:
    '''
    tensors = {}

    with tf.name_scope('embedding'):
        w = tf.Variable(tf.random_uniform([num_words, num_units], -1.0, 1.0), name="W")
        # 词向量shape [?,?,num_units]
        inputs = tf.nn.embedding_lookup(w, input_data)

    with tf.name_scope('lstm'):
        lstmcell = tf.nn.rnn_cell.BasicLSTMCell
        cell_list = [lstmcell(num_units, state_is_tuple=True) for i in range(num_layers)]
        cell_mul = tf.nn.rnn_cell.MultiRNNCell(cell_list, state_is_tuple=True)

        initial_state = cell_mul.zero_state(batch_size=batchsize, dtype=tf.float32)
        # 序列输出shape [?,?,num_units]
        outputs, last_state = tf.nn.dynamic_rnn(cell_mul, inputs, initial_state=initial_state)

    with tf.name_scope('softmax'):
        output = tf.reshape(outputs, [-1, num_units])
        weights = tf.Variable(tf.truncated_normal([num_units, num_words]))
        bias = tf.Variable(tf.zeros(shape=[num_words]))
        logits = tf.nn.bias_add(tf.matmul(output, weights), bias=bias)

    # 训练的时候计算loss,target用独热编码;生成的时候只需要计算logits
    if batchsize > 1:
        with tf.name_scope('loss'):
            # labels = tf.one_hot(tf.reshape(output_targets, [-1]), depth=num_words)
            # # 每个词语的loss,shape=[?, num_words]
            # loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            # total_loss = tf.reduce_mean(loss)

            #和上面等价，感觉好像这个的训练快一些？
            labels = tf.reshape(output_targets, [-1])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            total_loss = tf.reduce_mean(loss)

        train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

        tensors['initial_state'] = initial_state
        tensors['output'] = output
        tensors['train_op'] = train_op
        tensors['total_loss'] = total_loss
        tensors['loss'] = loss
        tensors['last_state'] = last_state
    else:
        prediction = tf.nn.softmax(logits)

        tensors['initial_state'] = initial_state
        tensors['last_state'] = last_state
        tensors['prediction'] = prediction

    return tensors
