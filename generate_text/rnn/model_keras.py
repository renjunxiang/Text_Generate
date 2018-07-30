from keras.layers import Input, Embedding, LSTM, Dense, Reshape
from keras.models import Model
from keras import optimizers


def model_keras(num_words=3000, num_units=128):
    '''
    生成RNN模型
    :param num_words:词汇数量
    :param num_units:词向量维度,lstm神经元数量默认一样
    :return:
    '''
    data_input = Input(shape=[None])
    embedding = Embedding(input_dim=num_words, output_dim=num_units, mask_zero=True)(data_input)
    lstm = LSTM(units=num_units, return_sequences=True)(embedding)
    x = LSTM(units=num_units, return_sequences=True)(lstm)
    # keras好像不支持内部对y操作,不能像tensorflow那样用reshape
    # x = Reshape(target_shape=[-1, num_units])(x)
    outputs = Dense(units=num_words, activation='softmax')(x)

    model = Model(inputs=data_input, outputs=outputs)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.adam(lr=0.01),
                  metrics=['accuracy'])
    return model
