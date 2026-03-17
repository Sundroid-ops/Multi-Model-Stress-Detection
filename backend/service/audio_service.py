import ffmpeg
import librosa
from pathlib import Path

import numpy as np

from backend.config import audio_store_dir

# extract audio from video using ffmpeg
def extract_audio(video_path):
    try:
        audio_file = Path(video_path)
        audio_path = audio_store_dir/ f"{audio_file.stem}.mp4"

        # video -> audio (audio rate = 22050, audio channel = mono (1))
        ffmpeg.input(str(video_path)) \
            .audio \
            .output(str(audio_path), ar=22050, ac=1) \
            .overwrite_output() \
            .run(quiet=True) # prevents ffmpeg logs

        print('Audio extracted : ', audio_path)

        return str(audio_path)

    except FileNotFoundError as err:
        print('audio path not found ', err)
        raise
    except ffmpeg.Error as err:
        print(f"[extract_audio] ffmpeg error: {err.stderr.decode()}")
        raise
    except Exception as ex:
        print('Unexpected Error while extracting audio', ex)
        raise

# split audio into 3 second segments
def split_audio(audio_path, segment_duration = 3):
    try:
        audio, sr = librosa.load(audio_path)
        segment_length = segment_duration * sr

        segments = []

        for start in range(0, len(audio), segment_length):
            end = start + segment_length
            segment = audio[start:end]

            # segment = 3 secs data
            if len(segment) == segment_length:
                segments.append(segment)

        return segments, sr

    except Exception as ex:
        print('Unexpected Error while splitting audio', ex)
        raise

# extract mfcc, delta, delta2 per segment
def extract_audio_features(video_path):
    try:
        audio_path = extract_audio(video_path)
        segments, sr = split_audio(audio_path)

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

            segments.append(feature_vector)

        audio_features = np.array(audio_features)
        print('audio features shape: ', audio_features.shape)

        return audio_features

    except ValueError as err:
        print('error : ', err)
    except Exception as ex:
        print('Unexpected Error while extracting audio features', ex)
        raise