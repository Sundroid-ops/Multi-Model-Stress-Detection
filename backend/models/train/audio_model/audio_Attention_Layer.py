from tensorflow.keras.layers import Layer, Dense
import tensorflow.keras.backend as K

class AttentionLayer(Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()
        self.dense = Dense(1, activation='tanh')

    def call(self, inputs):
        score = self.dense(inputs)
        score = K.squeeze(score, axis=-1)

        weights = K.softmax(score)
        weights = K.expand_dims(weights, axis=-1)

        context = inputs * weights
        context = K.sum(context, axis=1)

        return context