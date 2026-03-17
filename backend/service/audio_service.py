import ffmpeg
from pathlib import Path

import librosa

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