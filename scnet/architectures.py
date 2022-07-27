import tensorflow.keras.backend as K

from gloro.layers import InvertibleDownsampling
from gloro.layers import MinMax
from gloro.layers import ResnetBlock
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D


def dense(input_shape, num_classes, width, depth=6):

    x = Input(input_shape)

    z = x
    for _ in range(depth):
        z = Dense(width)(z)
        z = Activation('relu')(z)

    y = Dense(num_classes)(z)

    return x, y

def dense_acas(input_shape, num_classes):
    return dense(input_shape, num_classes, width=50, depth=6)

def dense_medium(input_shape, num_classes, depth=6):
    return dense(input_shape, num_classes, width=200, depth=depth)

def dense_large(input_shape, num_classes, depth=6):
    return dense(input_shape, num_classes, width=1000, depth=depth)


def cnn_tiny(input_shape, num_classes):

    x = Input(input_shape)
    z = Conv2D(32, 3, padding='same')(x)
    z = Activation('relu')(z)
    z = MaxPooling2D()(z)

    z = Conv2D(64, 3, padding='same')(z)
    z = Activation('relu')(z)
    z = MaxPooling2D()(z)

    z = Flatten()(z)
    z = Dense(256)(z)
    z = Activation('relu')(z)

    z = Dense(256)(z)
    z = Activation('relu')(z)

    y = Dense(num_classes)(z)

    return x, y


def resnet50(input_shape, num_classes, pretrained=False):
    if pretrained:
        model = ResNet50(
            input_shape=input_shape, 
            include_top=False, 
            pooling='avg')

        y = Dense(num_classes)(model.output)

    else:
        model = ResNet50(
            input_shape=input_shape, 
            classes=num_classes,
            weights=None,
            classifier_activation=None)

        y = model.output

    return model.input, y

def pretrained_resnet50(input_shape, num_classes):
    return resnet50(input_shape, num_classes, pretrained=True)
