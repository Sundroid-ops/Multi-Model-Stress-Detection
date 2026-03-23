import numpy as np

from backend.utils.emotions import emotions

# fusion logic based on audio and image vectors
def fusion(audio_vector, image_vector, audio_weight = 0.4, image_weight = 0.6):
    try:
        print('Started fusion logic ....')

        audio_vector = np.array(audio_vector)
        image_vector = np.array(image_vector)

        # fusion strategies

        # strategy - 1 -> weighted average
        # trusting images more than audio
        weighted = (audio_weight * audio_vector) + (image_weight * image_vector)

        # strategy - 2 -> max fusion
        # taking highest confidence per emotion across both models
        max_fusion = np.maximum(audio_vector, image_vector)
        max_fusion = max_fusion / max_fusion.sum() # normalize

        # strategy - 3 -> agreement
        # if both models agree on same emotion, boost its confidence
        audio_idx  = np.argmax(audio_weight)
        image_idx = np.argmax(image_vector)

        agreement_fusion = weighted.copy()
        emotions_lst = emotions() # list containing types of emotions

        # both models agree on same emotion
        if audio_idx == image_idx:
            # boosting confidence by 20%
            agreement_fusion[audio_idx] *= 1.2
            agreement_fusion = agreement_fusion/ agreement_fusion.sum() # renormalize
            print(f'Models agree on : {emotions_lst[audio_idx]} - boosting confidence')

        else:
            # disagree - trust images more
            agreement_fusion = (0.3 * audio_vector) + (0.7 * image_vector)
            agreement_fusion = agreement_fusion/ agreement_fusion.sum() # renormalize
            print(f'Models disagree — audio:{emotions_lst[audio_idx]} image:{emotions_lst[image_idx]} — trusting image more')

        print('Finished fusion logic ....')

        # final vector
        final_vector = (weighted + agreement_fusion) / 2 # combine weight + agreement
        final_vector = final_vector/ final_vector.sum() # normalize

        # final emotion
        final_idx = np.argmax(final_vector)
        final_emotion = emotions_lst[final_idx]
        confidence = final_vector[final_idx]

        print(f"\nWeighted fusion   : {np.round(weighted, 3)}")
        print(f"Max fusion        : {np.round(max_fusion, 3)}")
        print(f"Agreement fusion  : {np.round(agreement_fusion, 3)}")
        print(f"Final vector      : {np.round(final_vector, 3)}")
        print(f"Final emotion     : {final_emotion}")
        print(f"Confidence        : {confidence:.2%}")

    except Exception as ex:
        print('Unexpected error while applying fusion :', ex)
        raise