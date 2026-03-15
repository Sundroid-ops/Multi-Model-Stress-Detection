from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

from backend.models.evaluate_model import evaluate_model
from backend.models.preprocess.audio_preprocess import load_audio_data

l2_reg = 0.001

# build audio model
def build_audio_model(input_shape, num_classes):
    model = Sequential()

    # layer - 1
    model.add(Dense(units = 256,
                    kernel_initializer = 'he_normal',
                    kernel_regularizer = l2(l2_reg),
                    input_shape = (input_shape, )))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    # layer - 2
    model.add(Dense(units = 128,
                    kernel_initializer = 'he_normal',
                    kernel_regularizer = l2(l2_reg)))
    model.add(Activation('relu'))
    model.add(Dropout(0.35))

    # layer - 3
    model.add(Dense(units = 64,
                    kernel_initializer = 'he_normal',
                    kernel_regularizer = l2(l2_reg)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    # layer - 4
    model.add(Dense(units = 32,
                    kernel_initializer = 'he_normal',
                    kernel_regularizer = l2(l2_reg)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    # output
    model.add(Dense(num_classes, activation='softmax'))

    return model

def train_audio_model():
    X_train, X_test, y_train, y_test = load_audio_data()

    # 57 columns of audio data
    input_dim = X_train.shape[1]

    # num of classes of emotions
    num_classes = y_train.shape[1]

    model = build_audio_model(input_dim, num_classes)
    model.summary()

    callback = EarlyStopping(
        monitor = "val_loss",
        min_delta = 0.00001,
        patience = 20,
        verbose = 1,
        mode = "auto",
        restore_best_weights = False
    )

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 100, batch_size = 32, callbacks = callback)

    evaluate_model(history)


train_audio_model()

