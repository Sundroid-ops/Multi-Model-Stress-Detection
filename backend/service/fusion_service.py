import numpy as np
from sklearn.metrics.cluster import entropy

# calculate entropy of model to know its certainty
def model_certainty(vector):
     e = entropy(vector)
     max_entropy = np.log(len(vector))

     return float(1 - (e / max_entropy))

# aggregate image vectors in window
def aggregate_image_vectors(image_vectors):
    try:
        vectors = np.array(image_vectors)

        # confidence weighted average
        certainties = np.array([model_certainty(v) for v in vectors])

        # normalize
        weights = certainties/ np.sum(certainties)

        # weighted average across images
        aggregated = np.average(vectors, axis = 0, weights = weights)
        aggregated = aggregated/ np.sum(aggregated)

        print(f'Aggregated {len(vectors)} frames -> {aggregated.round(4)}')

        return aggregated

    except Exception as ex:
        print('Unexpected error while aggregating image vectors: ', ex)
        raise

# align audio and image vectors based by time window
def windowed_fusion(audio_vectors, image_vectors, fps = 1, segment_duration = 2):
    try:
        audio_vectors = np.array(audio_vectors)
        image_vectors = np.array(image_vectors)

        len_audio = len(audio_vectors)
        len_image = len(image_vectors)

        # 1 fps * 2 segment duration = 2 images per window
        images_per_window = fps * segment_duration

        stress_vectors = []
        emotion_vectors = []

        for window in range(len_audio):
            audio_vector = audio_vectors[window]

            # get image vectors for this window
            image_start = window * images_per_window
            image_end = image_start + images_per_window

            window_images = image_vectors[image_start: image_end] # (3, 7)

            # checks for 0 image vectors and skips it
            if len(window_images) == 0:
                print(f'No images found for window {window}, skipping')
                continue

            # skip aggregation if image vector length = 1
            if len(window_images) == 1:
                image_vector = window_images[0]
            else: # average frames
                image_vector = aggregate_image_vectors(window_images)

            # TODO: fuse window based on audio and image vector

            # TODO: calculate stress from emotion fused vector


    except Exception as ex:
        print('Unexpected error while executing windowed_fusion: ', ex)
        raise