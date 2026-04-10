# (valence, arousal) → (-1 to 1)
def get_EMOTION_VA_MAP():
    EMOTION_VA_MAP = {
        "angry"   : (-0.9,  0.8),
        "disgust" : (-0.7,  0.6),
        "fear"    : (-0.8,  0.9),
        "happy"   : ( 0.8,  0.4),
        "neutral" : ( 0.0,  0.3),
        "sad"     : (-0.7,  0.3)}

    return EMOTION_VA_MAP

# list of emotions
def get_emotions():
    emotion_list =  ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
    return emotion_list