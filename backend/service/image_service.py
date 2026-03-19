from backend.utils.image_util import extract_images

# extract and preprocess images for inference
def extract_image_features(video_path):
    try:
        image_paths = extract_images(video_path)

        # TODO: preprocess images

    except Exception as ex:
        print('Unexpected error while extracting image features: ', ex)
        raise