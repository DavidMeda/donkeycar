from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union
import donkeycar as dk
from tensorflow.keras.losses import Loss
from donkeycar.utils import normalize_image, denormalize_image
from donkeycar.pipeline.types import TubRecord
from donkeycar.parts.keras import KerasPilot
import tensorflow as tf
from tensorflow import keras
from donkeycar.config import Config
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Convolution2D, LayerNormalization
from tensorflow.keras.layers import  Dropout, Flatten
from tensorflow.keras.layers import TimeDistributed as TD
from tensorflow.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from donkeycar.pipeline.augmentations import ImageAugmentation
from PIL import Image

XY = Union[float, np.ndarray, Tuple[float, ...], Tuple[np.ndarray, ...]]

class SteeringLoss(Loss):  # inherit parent class
    # α ∈ [0.1, 1.0], β ∈ [1.0, 2.0], γ ∈ [1.0, 5.0]
    # β and γ have more impact
    # on the model than α. Specifically, if we want the model to
    # be trained faster, γ needs to be set smaller. If we want to
    # get a more accurate model, β need to be set larger or the
    # γ needs to be set bigger
    def __init__(self, alpha, beta, gamma):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    #compute loss
    def call(self, y_true, y_pred):
        return ((1+self.alpha*(tf.abs(y_true)**self.beta))**self.gamma)*(y_true-y_pred)**2


# def adjust_input_shape(input_shape, roi_crop):
#     height = input_shape[0]
#     new_height = height - roi_crop[0] - roi_crop[1]
#     return (new_height, input_shape[1], input_shape[2])

# old version
def customArchitecture(num_outputs, input_shape, roi_crop):

    # input_shape = adjust_input_shape(input_shape, roi_crop)
    img_in = Input(shape=input_shape, name='img_in')

    x = img_in
    # Dropout rate
    keep_prob = 0.9
    rate = 1 - keep_prob
    
    # Convolutional Layer 1
    x = Convolution2D(filters=24, kernel_size=5, strides=(2, 2), input_shape = input_shape)(x)
    x = Dropout(rate)(x)

    # Convolutional Layer 2
    x = Convolution2D(filters=36, kernel_size=5, strides=(2, 2), activation='relu')(x)
    x = Dropout(rate)(x)

    # Convolutional Layer 3
    x = Convolution2D(filters=48, kernel_size=5, strides=(2, 2), activation='relu')(x)
    x = Dropout(rate)(x)

    # Convolutional Layer 4
    x = Convolution2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu')(x)
    x = Dropout(rate)(x)

    # Convolutional Layer 5
    x = Convolution2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu')(x)
    x = Dropout(rate)(x)

    # Flatten Layers
    x = Flatten()(x)

    # Fully Connected Layer 1
    x = Dense(100, activation='relu')(x)

    # Fully Connected Layer 2
    x = Dense(50, activation='relu')(x)

    # Fully Connected Layer 3
    x = Dense(25, activation='relu')(x)
    
    # Fully Connected Layer 4
    x = Dense(10, activation='relu')(x)
    
    # Fully Connected Layer 5
    x = Dense(5, activation='relu')(x)
    outputs = []
    
    for i in range(num_outputs):
        # Output layer
        outputs.append(Dense(1, activation='linear', name='n_outputs' + str(i))(x))
        
    model = Model(inputs=img_in, outputs=outputs)
    
    return model


# def customArchitecture(num_outputs, input_shape, roi_crop):

#     # input_shape = adjust_input_shape(input_shape, roi_crop)
#     img_in = Input(shape=input_shape, name='img_in')

#     x = img_in
#     # Dropout rate
#     keep_prob = 0.8
#     rate = 1 - keep_prob
#     x = LayerNormalization(axis=1)(x)
#     # Convolutional Layer 1
#     x = Convolution2D(filters=24, kernel_size=5, strides=(
#         2, 2), input_shape=input_shape)(x)
#     # x = Dropout(rate)(x)

#     # Convolutional Layer 2
#     x = Convolution2D(filters=36, kernel_size=5,
#                       strides=(2, 2), activation='relu')(x)
#     # x = Dropout(rate)(x)

#     # Convolutional Layer 3
#     x = Convolution2D(filters=48, kernel_size=5,
#                       strides=(2, 2), activation='relu')(x)
#     # x = Dropout(rate)(x)

#     # Convolutional Layer 4
#     x = Convolution2D(filters=64, kernel_size=3,
#                       strides=(1, 1), activation='relu')(x)
#     # x = Dropout(rate)(x)

#     # Convolutional Layer 5
#     x = Convolution2D(filters=64, kernel_size=3,
#                       strides=(1, 1), activation='relu')(x)
#     # x = Dropout(rate)(x)

#     # Flatten Layers
#     x = Flatten()(x)

#     # Fully Connected Layer 1
#     x = Dense(1164, activation='relu')(x)
#     x = Dropout(rate)(x)

#     # Fully Connected Layer 2
#     x = Dense(100, activation='relu')(x)
#     x = Dropout(rate)(x)

#     # Fully Connected Layer 3
#     x = Dense(50, activation='relu')(x)
#     x = Dropout(rate)(x)

#     # Fully Connected Layer 4
#     x = Dense(10, activation='relu')(x)
#     x = Dropout(rate)(x)

#     # Fully Connected Layer 5
#     outputs = []
#     for i in range(num_outputs):
#         # Output layer
#         outputs.append(Dense(1, activation='tanh',
#                        name='n_outputs' + str(i))(x))

#     model = Model(inputs=img_in, outputs=outputs)

#     return model

class NvidiaModel(KerasPilot):
    def __init__(self, cfg: Config, num_outputs=2, input_shape=(120, 160, 3), roi_crop=(0, 0), *args, **kwargs):
        super(NvidiaModel, self).__init__(*args, **kwargs)
        self.model = customArchitecture(num_outputs, input_shape, roi_crop)
        self.compile()
        self.augmentation = ImageAugmentation(cfg)
        self.customLoss = SteeringLoss(1.0, 1.0, 1.0)

    def compile(self):
        self.model.compile(
            optimizer="adam", loss=self.customLoss, metrics=['accuracy'])
        # self.model.compile(optimizer="adam", loss='mse', metrics=['accuracy'] )


    def inference(self, img_arr, other_arr):
        # img = Image.fromarray(denormalize_image(img_arr))
        # img = img.resize((200,66))
        # img_arr = normalize_image(np.asarray(img))
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        outputs = self.model.predict(img_arr)
        steering = outputs[0]
        throttle = outputs[1]
        print("output: ", outputs)
        print("Throttle: ", throttle[0][0], " Steering: ", steering[0][0])
        return steering[0][0], throttle[0][0]
    
    def load(self, model_path: str) -> None:
        print(f'Loading model {model_path} with SteeringLoss')
        self.model = keras.models.load_model(model_path, compile=False, custom_objects={
            'loss': self.customLoss})

    def y_transform(self, record: TubRecord):
        angle: float = record.underlying['user/angle']
        throttle: float = record.underlying['user/throttle']
        return angle, throttle

    def y_translate(self, y: XY) -> Dict[str, Union[float, np.ndarray]]:
        if isinstance(y, tuple):
            angle, throttle = y
            return {'n_outputs0': angle, 'n_outputs1': throttle}
        else:
            raise TypeError('Expected tuple')
    
    def x_transform(self, record: TubRecord) -> XY:
        img_arr = record.image(cached=True)
        return img_arr

    def output_shapes(self):
        # need to cut off None from [None, 120, 160, 3] tensor shape
        img_shape = self.get_input_shape()[1:]
        shapes = ({'img_in': tf.TensorShape(img_shape)},
                  {'n_outputs0': tf.TensorShape([]),
                   'n_outputs1': tf.TensorShape([])})
        return shapes
