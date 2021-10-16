from donkeycar.parts.keras import KerasPilot
import numpy as np
import tensorflow as tf
from typing import Dict, Tuple,  Union
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, LeakyReLU, DenseFeatures
from tensorflow.python.keras.layers.merge import Concatenate
from donkeycar.pipeline.types import TubRecord
import cv2
import numpy as np

XY = Union[float, np.ndarray, Tuple[float, ...], Tuple[np.ndarray, ...]]

def oriModelArchitecture(inputShape):

    # Dropout rate
    keep_prob = 0.9
    rate = 1 - keep_prob

    # Input layers
    imageInput = Input(shape=inputShape, name='imageInput')
    laneInput = Input(shape=inputShape, name='laneInput')
    # behaviourInput = Input(shape=(numberOfBehaviourInputs,), name="behaviourInput")

    x = imageInput
    x = DenseFeatures()(x)
    x = Conv2D(24, (5,5), strides=(2,2), name="Conv2D_imageInput_1")(x)
    x = LeakyReLU()(x)
    x = Dropout(rate)(x)
    x = Conv2D(32, (5,5), strides=(2,2), name="Conv2D_imageInput_2")(x)
    x = LeakyReLU()(x)
    x = Dropout(rate)(x)
    x = Conv2D(64, (5,5), strides=(2,2), name="Conv2D_imageInput_3")(x)
    x = LeakyReLU()(x)
    x = Dropout(rate)(x)
    x = Conv2D(64, (3,3), strides=(1,1), name="Conv2D_imageInput_4")(x)
    x = LeakyReLU()(x)
    x = Dropout(rate)(x)
    x = Conv2D(64, (3,3), strides=(1,1), name="Conv2D_imageInput_5")(x)
    x = LeakyReLU()(x)
    x = Dropout(rate)(x)
    x = Flatten(name="flattenedx")(x)
    x = Dense(100)(x)
    x = Dropout(rate)(x)

    y = laneInput
    y = Conv2D(24, (5,5), strides=(2,2), name="Conv2D_laneInput_1")(y)
    y = LeakyReLU()(y)
    y = Dropout(rate)(y)
    y = Conv2D(32, (5,5), strides=(2,2), name="Conv2D_laneInput_2")(y)
    y = LeakyReLU()(y)
    y = Dropout(rate)(y)
    y = Conv2D(64, (5,5), strides=(2,2), name="Conv2D_laneInput_3")(y)
    y = LeakyReLU()(y)
    y = Dropout(rate)(y)
    y = Conv2D(64, (3,3), strides=(1,1), name="Conv2D_laneInput_4")(y)
    y = LeakyReLU()(y)
    y = Dropout(rate)(y)
    y = Conv2D(64, (3,3), strides=(1,1), name="Conv2D_laneInput_5")(y)
    y = LeakyReLU()(y)
    y = Flatten(name="flattenedy")(y)
    y = Dense(100)(y)
    y = Dropout(rate)(y)

    # Concatenated final convnet
    c = Concatenate(axis=1)([x, y])
    c = Dense(100, activation='relu')(c)
    c = Dense(50, activation='relu')(c)

    # Output layers
    steering_out = Dense(1, activation='linear', name='steering_out')(c)
    throttle_out = Dense(1, activation='linear', name='throttle_out')(c)
    model = Model(inputs=[imageInput, laneInput], outputs=[steering_out, throttle_out]) 
    
    return model

class OriModel(KerasPilot):
    '''
    Custom model that takes an input image and feeds it and a preprocessed version of it to the model.
    The preprocessing converts the image to HSL color space, extracts the S channel and thresholds it.
    The thresholded S channel is passed to the model to help find lane lines easier.
    '''
    def __init__(self, model=None, input_shape=(160, 120, 3), *args, **kwargs):
        super(OriModel, self).__init__(*args, **kwargs)
        self.model = oriModelArchitecture(inputShape=input_shape)
        self.compile()

    def compile(self):
        self.model.compile(optimizer="adam",loss='mse')

    def inference(self, inputImage):
        # Preprocesses the input image for easier lane detection
        extractedLaneInput = self.processImage(inputImage)
        # Reshapes to (1, height, width, channels)
        extractedLaneInput = extractedLaneInput.reshape((1,) + extractedLaneInput.shape)
        inputImage = inputImage.reshape((1,) + inputImage.shape)
        # Predicts the output steering and throttle
        steering, throttle = self.model.predict([inputImage, extractedLaneInput])
        print("Throttle: %f, Steering: %f" % (throttle[0][0], steering[0][0]))
        return steering[0][0], throttle[0][0]

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


    def output_shapes(self):
        # need to cut off None from [None, 120, 160, 3] tensor shape
        img_shape = self.get_input_shape()[1:]
        shapes = ({'img_in': tf.TensorShape(img_shape)},
                  {'n_outputs0': tf.TensorShape([]),
                   'n_outputs1': tf.TensorShape([])})
        return shapes

    def extractLaneLinesFromSChannel(self, warpedImage):
        # Convert to HSL
        hslImage = cv2.cvtColor(warpedImage, cv2.COLOR_BGR2HLS)
        # Split the image into three variables by the channels
        hChannel, lChannel, sChannel = cv2.split(hslImage)
        # Threshold the S channel image to select only the lines
        lowerThreshold = 65
        higherThreshold = 255
        # Threshold the image, keeping only the pixels/values that are between lower and higher threshold
        returnValue, binaryThresholdedImage = cv2.threshold(sChannel,lowerThreshold,higherThreshold,cv2.THRESH_BINARY)
        # Since this is a binary image, we'll convert it to a 3-channel image so OpenCV can use it
        thresholdedImage = cv2.cvtColor(binaryThresholdedImage, cv2.COLOR_GRAY2RGB)
        return thresholdedImage

    def processImage(self, image): 
        # warpedImage = self.warpImage(image)
        # We'll normalize it just to make sure it's between 0-255 before thresholding
        new_image = cv2.normalize(image,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
        thresholdedImage = self.extractLaneLinesFromSChannel(new_image)
        one_byte_scale = 1.0 / 255.0 
        # To make sure it's between 0-1 for the model
        return np.array(thresholdedImage).astype(np.float32) * one_byte_scale