from keras.preprocessing.image import ImageDataGenerator


"""
# Generates image data for training CNN models
# Uses datagen from flow_from_directory.
# Use in case if data hierarchy is:
#
#...train/
#.....hot_dogs
#........a_image_1.jpg
#........a_image_2.jpg
#.....not_hot_dogs
#........b_image_3.jpg
#........b_image_4.jpg
#
#...validation/
#.....hot_dogs
#........a_image_5.jpg
#........a_image_6.jpg
#.....not_hot_dogs
#........b_image_7.jpg
#........b_image_8.jpg
"""

def get_train_generator(train_dir, target_size, batch_size):
    """
    Validation Data generator for binary classiffication
    for CNN models
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary')
    return train_generator

def get_validation_generator(train_dir, target_size, batch_size):
    """
    Validation Data generator for binary classiffication
    for CNN models
    Note: no augmentation for validation data
    """
    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary')
    return validation_generator
