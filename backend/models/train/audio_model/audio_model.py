import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dropout,
                            Dense, Bidirectional, LSTM,
                            Reshape, BatchNormalization)
from tensorflow.keras.regularizers import l2

from backend.config import audio_dir
from backend.models.evaluate_model import evaluate_model
from backend.models.preprocess.train.audio_preprocess.audio_generator import AudioDataGenerator
from backend.models.preprocess.train.audio_preprocess.audio_train_preprocess import load_dataset
from backend.models.train.audio_model.audio_Attention_Layer import AttentionLayer

# build audio model
def build_model(input_shape, num_classes):
    l2_reg = 0.0001
    pool_size = (2, 2)
    kernel_size = (3, 3)

    model = Sequential()

    # -------- CNN -------- #
    model.add(Conv2D(32, kernel_size, padding='same',
                     activation='relu',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg),
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, kernel_size, padding='same',
                     activation='relu',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, kernel_size, padding='same',
                     activation='relu',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size))
    model.add(Dropout(0.4))

    # -------- RESHAPE -------- #
    model.add(Reshape((-1, model.output_shape[-1])))

    # -------- LSTM -------- #
    model.add(Bidirectional(
        LSTM(128,
             return_sequences=True,
             kernel_regularizer=l2(l2_reg),
             recurrent_regularizer=l2(l2_reg))
    ))

    # -------- ATTENTION -------- #
    model.add(AttentionLayer())

    # -------- DENSE -------- #
    model.add(Dense(128,
                    activation='relu',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(0.5))

    # -------- OUTPUT -------- #
    model.add(Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=l2(l2_reg)))

    return model

def train_model():
    print('Starting Training Audio Model ....')

    # load data
    file_paths, y_onehot, y_encoded = load_dataset(audio_dir)

    # split
    X_train_paths, X_test_paths, y_train, y_test = train_test_split(
        file_paths, y_onehot,
        test_size=0.2,
        stratify=y_encoded,
        random_state=42
    )

    # generators
    train_gen = AudioDataGenerator(X_train_paths, y_train)
    test_gen = AudioDataGenerator(X_test_paths, y_test)

    # class weights
    y_labels = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_labels),
        y=y_labels
    )
    class_weights = dict(enumerate(class_weights))

    # model
    input_shape = (128, 128, 1)
    num_classes = y_train.shape[1]

    model = build_model(input_shape, num_classes)
    model.summary()

    # callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-5
    )

    print('Training audio model ....')

    # compile
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # train
    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=50,
        class_weight=class_weights,
        callbacks=[early_stop, lr_scheduler]
    )

    evaluate_model(history)