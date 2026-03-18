import ffmpeg
import librosa

from pathlib import Path

from backend.config import audio_store_dir

# extract audio from video using ffmpeg
def extract_audio(video_path):
    try:
        print('extracting audio ....')

        audio_file = Path(video_path)
        audio_path = Path(audio_store_dir) / f"{audio_file.stem}.wav"

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
        print('splitting audio ....')

        audio, sr = librosa.load(audio_path)

        audio_length = len(audio)
        duration = audio_length / sr

        print(f"samples  : {audio_length}")
        print(f"duration : {duration / sr:.2f}s")

        if duration < segment_duration:
            raise ValueError(f'Audio too short , minimum {segment_duration} seconds required for at least 1 segment')

        segment_length = int(segment_duration * sr)

        segments = []

        for start in range(0, len(audio), segment_length):
            end = start + segment_length
            segment = audio[start:end]

            # segment = 3 secs data
            if len(segment) == segment_length:
                segments.append(segment)

        print('splitting done')

        return segments, sr
    except ValueError as err:
        print('error : ', err)
        raise
    except Exception as ex:
        print('Unexpected Error while splitting audio', ex)
        raise
