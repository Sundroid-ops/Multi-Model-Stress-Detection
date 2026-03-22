import joblib

from backend.config import audio_scaler_path, audio_encoder_path

# audio preprocessing for model inference
def audio_inference_preprocess(audio_features):
    try:
        print('Preprocessing audio for inference ....')

        # load scalar
        scalar = joblib.load(audio_scaler_path)

        # scale features using training scalar
        audio_features = scalar.transform(audio_features)

        print('Audio preprocessed shape:', audio_features.shape)
        print('Completed Preprocessing audio for inference ....')

        return audio_features

    except FileNotFoundError as err:
        print('error : ', err)
        raise
    except Exception as ex:
        print('Unexpected Error while preprocessing audio for inference : ', ex)
        raise