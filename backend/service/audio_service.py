import ffmpeg
from pathlib import Path

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
        print('Unexpected Error while extracting audio features', ex)
        raise