import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.python.keras.utils.np_utils import to_categorical

from backend.config import audio_dir, audio_scaler_path, audio_encoder_path

# processing csv data to training and test data
def audio_preprocess():
    df = pd.read_csv(audio_dir)

    # splitting 57 cols into 3 groups of 19
    mfcc = df.iloc[:, 1:20].values
    delta = df.iloc[:, 20:39].values
    delta2 = df.iloc[:, 39:58].values

    # last column is emotion label
    emotion_label = df.iloc[:, -1].values

    # stacking (samples, 57) flat for dense model
    X = np.concatenate((mfcc, delta, delta2), axis = 1)

    # normalizing features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # saving trained scaler for using during inference
    joblib.dump(scaler, audio_scaler_path)
    print('Scaler saved for model inference')

    # encoding labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(emotion_label)
    y_onehot = to_categorical(y_encoded)

    # saving trained encoder for using during inference
    joblib.dump(encoder, audio_encoder_path)
    print('Encoder saved for model inference')

    return X, y_encoded, y_onehot

# loading audio data for audio model training
def load_audio_data():
    X, y_encoded, y_onehot = audio_preprocess()

    # splitting training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    return X_train, X_test, y_train, y_test