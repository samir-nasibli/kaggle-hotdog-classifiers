from keras.callbacks import TensorBoard
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.models import Model


def define_CNN(width, height):
    """
    args:
       width:
       height:
    -------
    return:
       model
    """
    # Define the model
    model = Sequential()

    # 2D convolution layers
    model.add(Conv2D(32, (3, 3), input_shape=(width, height, 3),
              activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # Downsizes images by 1/2 in this layer

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Flattens layers from 3D to 1D further compressing features
    model.add(Flatten())

    # Regular densely connected layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    return model


def define_VGG16_fc2_out():
    base_model = VGG16(weights='imagenet', include_top=True)
    out = base_model.get_layer("fc2").output
    model = Model(inputs=base_model.input, outputs=out)
    return model


def define_VGG19_fc2_out():
    base_model = VGG19(weights='imagenet', include_top=True)
    out = base_model.get_layer("fc2").output
    model = Model(inputs=base_model.input, outputs=out)
    return model
