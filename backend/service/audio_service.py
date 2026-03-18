import librosa
import numpy as np

from backend.utils.audio_util import extract_audio, split_audio

# extract mfcc, delta, delta2 per segment
def extract_audio_features(video_path):
    try:
        audio_path = extract_audio(video_path)
        segments, sr = split_audio(audio_path)

        print('extracting audio features ....')

        if not segments:
            return ValueError('No audio segments found')

        audio_features = []

        for segment in segments:
            # mfcc = (19, time_frames)
            mfcc = librosa.feature.mfcc(y = segment, sr = sr, n_mfcc = 19)

            # delta = rate of change
            delta = librosa.feature.delta(mfcc)

            # delta2 = acceleration
            delta2 = librosa.feature.delta(mfcc, order=2)

            # mean across time = (19,) each
            mfcc_mean = np.mean(mfcc.T, axis=0)
            delta_mean = np.mean(delta.T, axis=0)
            delta2_mean = np.mean(delta2.T, axis=0)

            # combine → (57,)
            feature_vector = np.hstack([mfcc_mean, delta_mean, delta2_mean])

            audio_features.append(feature_vector)

        audio_features = np.array(audio_features)
        print('audio features shape: ', audio_features.shape)

        return audio_features

    except ValueError as err:
        print('error : ', err)
    except Exception as ex:
        print('Unexpected Error while extracting audio features', ex)
        raise