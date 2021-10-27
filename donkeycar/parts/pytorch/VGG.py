
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torch.optim as optim
from donkeycar.parts.pytorch.my_torch_data import get_default_transform
from torch.nn import MSELoss
import pytorch_lightning as pl

class VGG(pl.LightningModule):
    def __init__(self, input_shape=(128, 3, 224, 224), output_size=(2,)):
        super().__init__()
        # Used by PyTorch Lightning to print an example model summary
        # self.example_input_array = torch.rand(input_shape)

        # # Metrics
        # self.train_loss = MSELoss()
        # self.valid_loss = MSELoss()
    
        self.model = models.vgg16(pretrained=False)
        
        #print(self.model)
        self.model.eval()
        self.inference_transform = get_default_transform(for_inference=True)

        # Keep track of the loss history. This is useful for writing tests
        # self.loss_history = []

    def forward(self, x):
        # Forward defines the prediction/inference action
        return self.model(x)
    
    def load_model_trained(self, path, output):
        checkpoint = torch.load(path , map_location='cpu')

        self.model = models.vgg16(pretrained=False)
        
        self.model.classifier._modules['6'] = nn.Linear(4096, output)
        
        self.model.load_state_dict(checkpoint['state_dict'])
        print(self.model)
        # return model
        
    def run(self, img_arr: np.ndarray, other_arr: np.ndarray = None):
        """
        Donkeycar parts interface to run the part in the loop.

        :param img_arr:     uint8 [0,255] numpy array with image data
        :param other_arr:   numpy array of additional data to be used in the
                            pilot, like IMU array for the IMU model or a
                            state vector in the Behavioural model
        :return:            tuple of (angle, throttle)
        """
        from PIL import Image

        pil_image = Image.fromarray(img_arr)
        tensor_image = self.inference_transform(pil_image)
        tensor_image = tensor_image.unsqueeze(0)

        # Result is (1, 2)
        result = self.forward(tensor_image)

        # Resize to (2,)
        result = result.reshape(-1)
        # print(result[0].item())
        # print(result[1].item())

        # Convert from being normalized between [0, 1] to being between [-1, 1]
        result = result * 2 - 1
        print("VGG11 result: {}".format(result))
        #return result
        return result[0].item(), result[1].item()
