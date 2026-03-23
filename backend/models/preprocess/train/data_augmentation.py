from keras.src.legacy.preprocessing.image import ImageDataGenerator

from backend.config import image_train_dir, image_test_dir
from backend.utils.emotions import emotions

img_size  = (256, 256)
batch     = 32

# augmentation for TRAIN only
def data_augmentation():
    train_datagen = ImageDataGenerator(
        rescale=1./255,            # scale pixels 0-1
        rotation_range=15,         # rotate up to 15 degrees
        width_shift_range=0.1,     # shift horizontally
        height_shift_range=0.1,    # shift vertically
        horizontal_flip=True,      # flip horizontally
        zoom_range=0.1,            # slight zoom in
        shear_range=0.1,           # shear transformation
        fill_mode="nearest"        # fill empty pixels
    )


    # scaling ONLY for TEST
    test_datagen = ImageDataGenerator(
        rescale=1./255
    )

    return train_datagen, test_datagen

# load images
def image_generator():
    # image augmentation
    train_datagen, test_datagen = data_augmentation()

    emotion_class = emotions()

    # load from train dir
    train_gen = train_datagen.flow_from_directory(
        image_train_dir,
        target_size = img_size,
        batch_size = batch,
        class_mode = "categorical",
        classes = emotion_class
    )

    # load from test dir
    test_gen = test_datagen.flow_from_directory(
        image_test_dir,
        target_size = img_size,
        batch_size = batch,
        class_mode = "categorical",
        classes = emotion_class
    )

    return train_gen, test_gen