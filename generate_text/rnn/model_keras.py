from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model


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
    # x = Dense(units=1000, activation='relu')(x)
    dense2 = Dense(units=num_words, activation='softmax')(x)
    model = Model(inputs=data_input, outputs=dense2)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
