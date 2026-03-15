from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Activation, Dropout, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

from ..preprocess.data_augmentation import image_generator

l2_reg = 0.001

pool_size = (2, 2)
kernel_size = (3, 3)

# build image model
def build_image_model(input_shape, num_classes):
    model = Sequential()

    # layer - 1
    model.add(Conv2D(filters = 32,
                     kernel_size = kernel_size,
                     padding = 'same',
                     input_shape = input_shape,
                     kernel_initializer = 'he_normal',
                     kernel_regularizer = l2(l2_reg)))
    model.add(Activation('relu'))

    model.add(Conv2D(filters = 32,
                     kernel_size = kernel_size,
                     padding = 'same',
                     kernel_initializer = 'he_normal',
                     kernel_regularizer = l2(l2_reg)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size = pool_size))
    model.add(Dropout(0.25))

    # layer - 2
    model.add(Conv2D(filters = 64,
                     kernel_size=kernel_size,
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg)))
    model.add(Activation('relu'))

    model.add(Conv2D(filters = 64,
                     kernel_size=kernel_size,
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    # layer - 3
    model.add(Conv2D(filters = 128,
                     kernel_size=kernel_size,
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg)))
    model.add(Activation('relu'))

    model.add(Conv2D(filters = 128,
                     kernel_size=kernel_size,
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.35))

    # layer - 4
    model.add(Conv2D(filters = 256,
                     kernel_size=kernel_size,
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.35))

    # fully connected layer
    model.add(Flatten())

    model.add(Dense(units = 512,
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(units = 256,
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg)))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    # output
    model.add(Dense(units = num_classes, activation='softmax', kernel_initializer='glorot_uniform'))

    return model

# train image model
def train_image_model():
    model = build_image_model(input_shape=(256, 256, 3), num_classes = 5)
    model.summary()

    callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00001,
        patience=20,
        verbose=1,
        mode="auto",
        restore_best_weights = False
    )

    # retrieving train and test generators
    train_gen, test_gen = image_generator()

    model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy', metrics = ['accuracy'])
    history = model.fit(train_gen, validation_data = test_gen, epochs=10, callbacks = callback)

train_image_model()



