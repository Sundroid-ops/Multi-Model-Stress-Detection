from backend.service.audio_service import extract_audio_features
from backend.utils.video_util import allowed_file

# extract video features (images + audio)
def extract_video_features(video_path):
    try:
        if video_path == '':
            raise FileNotFoundError('No file provided')

        if not allowed_file(video_path):
            raise ValueError('Invalid file format')

        # audio_features
        audio_features = extract_audio_features(video_path)

        # TODO: image_features

    except FileNotFoundError as err:
        print('error: ', err)

    except ValueError as err:
        print('error: ', err)

    except Exception as ex:
        print('Unexpected error: ', ex)