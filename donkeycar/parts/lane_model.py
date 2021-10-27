from donkeycar.parts.keras import KerasPilot
import numpy as np
import tensorflow as tf
from typing import Dict, Tuple,  Union
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, LeakyReLU
from tensorflow.python.keras.layers.merge import Concatenate
from donkeycar.pipeline.types import TubRecord
import cv2
import numpy as np
from tensorflow.keras.losses import Loss


XY = Union[float, np.ndarray, Tuple[float, ...], Tuple[np.ndarray, ...]]


def create_model(inputShape):

    # Dropout rate
    keep_prob = 0.9
    rate = 1 - keep_prob

    # Input layers
    imageInput = Input(shape=inputShape, name='imageInput')
    

    # behaviourInput = Input(shape=(numberOfBehaviourInputs,), name="behaviourInput")
    x = imageInput
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
    c = Dropout(rate)(x)

    # laneInput = Input(shape=inputShape, name='laneInput')
    # y = laneInput
    # y = Conv2D(24, (5,5), strides=(2,2), name="Conv2D_laneInput_1")(y)
    # y = LeakyReLU()(y)
    # y = Dropout(rate)(y)
    # y = Conv2D(32, (5,5), strides=(2,2), name="Conv2D_laneInput_2")(y)
    # y = LeakyReLU()(y)
    # y = Dropout(rate)(y)
    # y = Conv2D(64, (5,5), strides=(2,2), name="Conv2D_laneInput_3")(y)
    # y = LeakyReLU()(y)
    # y = Dropout(rate)(y)
    # y = Conv2D(64, (3,3), strides=(1,1), name="Conv2D_laneInput_4")(y)
    # y = LeakyReLU()(y)
    # y = Dropout(rate)(y)
    # y = Conv2D(64, (3,3), strides=(1,1), name="Conv2D_laneInput_5")(y)
    # y = LeakyReLU()(y)
    # y = Flatten(name="flattenedy")(y)
    # y = Dense(100)(y)
    # y = Dropout(rate)(y)

    # Concatenated final convnet
    # c = Concatenate()([x, y])
    c = Dense(100, activation='relu')(c)
    c = Dense(50, activation='relu')(c)

    # Output layers
    steering_out = Dense(1, activation='linear', name='steering_out')(c)
    throttle_out = Dense(1, activation='linear', name='throttle_out')(c)
    model = Model(inputs=[imageInput], outputs=[steering_out, throttle_out]) 
    
    return model


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

class LaneModel(KerasPilot):
    '''
    Custom model that takes an input image and feeds it and a preprocessed version of it to the model.
    The preprocessing converts the image to HSL color space, extracts the S channel and thresholds it.
    The thresholded S channel is passed to the model to help find lane lines easier.
    '''
    def __init__(self, model=None, input_shape=(120, 160, 3), *args, **kwargs):
        # super(LaneModel, self).__init__(*args, **kwargs)
        super().__init__()
        self.model = create_model(input_shape)
        self.compile()

    def compile(self):
        self.model.compile(optimizer=self.optimizer,
                           loss=SteeringLoss(1.0, 1.0, 1.0))

    def inference(self, img_arr, other_arr):
        # Preprocesses the input image for easier lane detection
        # extractedLaneInput = self.processImage(inputImage)
        # Reshapes to (1, height, width, channels)
        # other_arr = self.processImage(img_arr)
        # other_arr = other_arr.reshape((1,) + other_arr.shape)
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        # print("image.shape: ", img_arr.shape,
        #       " otherInput.shape: ", other_arr.shape)
        # Predicts the output steering and throttle
        outputs = self.model.predict([img_arr])
        print("output: ", outputs)
        steering = outputs[0]
        throttle = outputs[1]
        print("Throttle: ", throttle[0][0], " Steering: ", steering[0][0])
        return steering[0][0], throttle[0][0]

    def x_transform(self, record: TubRecord) -> XY:
        img_arr = super().x_transform(record)
        # extractedLaneInput = self.processImage(img_arr)
        # print("<<<<",extractedLaneInput.shape)
        # print(">>>>",extractedLaneInput)
        # img_arr = self.processImage(img_arr)
        # return self.processImage(img_arr)
        return img_arr

    def x_translate(self, x: XY) -> Dict[str, Union[float, np.ndarray]]:
        # assert isinstance(x, tuple), 'Requires tuple as input'
        # return {'imageInput': x[0], 'laneInput': x[1]}
        # if isinstance(x, tuple):
        #     imageInput, laneInput = x
        #     return {'imageInput': imageInput, 'laneInput': laneInput}
        # else:
        #     raise TypeError('Expected tuple')
        return {'imageInput': x}

    def y_transform(self, record: TubRecord):
        angle: float = record.underlying['user/angle']
        throttle: float = record.underlying['user/throttle']
        return angle, throttle

    def y_translate(self, y: XY) -> Dict[str, Union[float, np.ndarray]]:
        if isinstance(y, tuple):
            angle, throttle = y
            return {'steering_out': angle, 'throttle_out': throttle}
        else:
            raise TypeError('Expected tuple')

    def output_shapes(self):
        # need to cut off None from [None, 120, 160, 3] tensor shape
        img_shape = self.get_input_shape()[1:]
        # lane_shape = self.model.inputs[1].shape[1:]
        shapes = ({'imageInput': tf.TensorShape(img_shape)},
                #   {'laneInput': tf.TensorShape(lane_shape)},
                  {'steering_out': tf.TensorShape([]),
                   'throttle_out': tf.TensorShape([])})
        return shapes


    def warpImage(self, image):
        # Define the region of the image we're interested in transforming
        regionOfInterest = np.float32([[0, 100], [50, 60], [105, 60], [153, 100]])
        newPerspective = np.float32([[15, 120], [15, 0.25], [140, 0.25], [140, 120]])
        # [[0,  100],  # Bottom left
        # [50, 60], # Top left
        #[105, 60], # Top right
        #[153, 100]]) # Bottom right
        # Define the destination coordinates for the perspective transform

        #[[80,  180],  # Bottom left
        #[80,    0.25],  # Top left
        #[230,   0.25],  # Top right
        #[230, 180]]) # Bottom right
        # Compute the matrix that transforms the perspective
        transformMatrix = cv2.getPerspectiveTransform(
            regionOfInterest, newPerspective)
        # Warp the perspective - image.shape[:2] takes the height, width, [::-1] inverses it to width, height
        warpedImage = cv2.warpPerspective(
            image, transformMatrix, image.shape[:2][::-1], flags=cv2.INTER_LINEAR)
        
        return warpedImage

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
        warpedImage = self.warpImage(image)
        # We'll normalize it just to make sure it's between 0-255 before thresholding
        new_image = cv2.normalize(warpedImage, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        thresholdedImage = self.extractLaneLinesFromSChannel(new_image)
        one_byte_scale = 1.0 / 255.0 
        # To make sure it's between 0-1 for the model
        # return array 3 channel
        return np.asarray(thresholdedImage).astype(np.float32) * one_byte_scale 

    def binary_image(self, img):    # Choose a Sobel kernel size
        # warp image
        img_size = (img.shape[1], img.shape[0])
        regionOfInterest = np.float32([[0, 100], [50, 60], [105, 60], [153, 100]])
        # Given src and dst points, calculate the perspective transform matrix
        newPerspective = np.float32(
            [[15, 120], [15, 0.25], [140, 0.25], [140, 120]])
        # Warp the image using OpenCV warpPerspective()
        M = cv2.getPerspectiveTransform(regionOfInterest, newPerspective)
        img = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)
        # HLS color
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]    # Threshold color channel
        s_thresh_min = 50
        s_thresh_max = 255
        s_binary = np.zeros_like(s_channel)
        # Combine the two binary thresholds
        s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
        # return image 1 channel
        return s_binary.astype(np.float32) 
