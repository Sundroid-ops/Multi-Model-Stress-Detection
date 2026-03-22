from backend.models.preprocess.inference.image_inference import image_preprocess_inference
from backend.utils.image_util import extract_images

# extract and preprocess images for model inference
def extract_image_features(video_path):
    try:
        print('Extracting image features ....')

        # extract images from video
        image_paths = extract_images(video_path)

        # preprocess images
        images_data = image_preprocess_inference(image_paths)

        print('Extracted image features ....')

        return images_data

    except Exception as ex:
        print('Unexpected error while extracting image features: ', ex)
        raise