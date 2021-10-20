# PyTorch
import torch
from torch.utils.data import IterableDataset, DataLoader, Dataset
from donkeycar.utils import train_test_split
from donkeycar.parts.tub_v2 import Tub
from torchvision import transforms
from typing import List, Any
from donkeycar.pipeline.types import TubRecord, TubDataset
from donkeycar.pipeline.sequence import TubSequence
import pytorch_lightning as pl
import numpy as np


def get_default_transform(for_video=False, for_inference=False):
    """
    Creates a default transform to work with torchvision models

    Video transform:
    All pre-trained models expect input images normalized in the same way, 
    i.e. mini-batches of 3-channel RGB videos of shape (3 x T x H x W), 
    where H and W are expected to be 112, and T is a number of video frames 
    in a clip. The images have to be loaded in to a range of [0, 1] and 
    then normalized using mean = [0.43216, 0.394666, 0.37645] and 
    std = [0.22803, 0.22145, 0.216989].
    """

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    input_size = (224, 224)

    if for_video:
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]
        input_size = (112, 112)

    transform = transforms.Compose([
        transforms.CenterCrop((90,160)),#simil crop 
        transforms.Grayscale(3),
        transforms.Resize(input_size),
        transforms.ToTensor(), #normalize between [0,1]
        transforms.Normalize(mean=mean, std=std)
    ])

    return transform


class myDataset(Dataset):
    '''
    Loads the dataset, and creates a train/test split.
    '''
    def __init__(self, config, records: List[TubRecord], transform=None):
        """Create a PyTorch Tub Dataset
        Args:
            config (object): the configuration information
            records (List[TubRecord]): a list of tub records
            transform (function, optional): a transform to apply to the data
        """
        self.records = records
        self.config = config 
        self.size = len(records)
        # Handle the transforms
        if transform:
            self.transform = transform
        else:
            self.transform = get_default_transform()
        #self.sequence = TubSequence(records)
        #self.x, self.y = self._create_pipeline()
        #self.pipeline = self._create_pipeline
             
        
        #self.x = torch.transpose(self.x, 1, 0)
    
    def __getitem__(self, index):
        # self.x = []
        # self.y= []
        # for record in self.records:
            # y
        angle: float = self.records[index].underlying['user/angle']
        throttle: float = self.records[index].underlying['user/throttle']
        labels = torch.tensor([angle, throttle], dtype=torch.float)
        # Normalize to be between [0, 1]
        # angle and throttle are originally between [-1, 1]
        labels = (labels + 1) / 2
        # self.y.append(labels)

        # x
        img = self.records[index].image(cached=True, as_nparray=False)
        # self.x.append(self.transform(img))
        # return self.x[index], self.y[index]
        return self.transform(img), labels

    def __len__(self):
        return self.size


class TorchTubDataModule():

    def __init__(self, config: Any, tub_paths: List[str], transform=None):
        """Create a PyTorch Lightning Data Module to contain all data loading logic

        Args:
            config (object): the configuration information
            tub_paths (List[str]): a list of paths to the tubs to use (minimum size of 1).
                                   Each tub path corresponds to another training run.
            transform (function, optional): a transform to apply to the data
        """
        super().__init__()

        self.config = config
        self.tub_paths = tub_paths

        # Handle the transforms
        if transform:
            self.transform = transform
        else:
            self.transform = get_default_transform()

        self.tubs: List[Tub] = [Tub(tub_path, read_only=True) for tub_path in self.tub_paths]
        self.records: List[TubRecord] = []
        self.setup()

    def setup(self, stage=None):
        """Load all the tub data and set up the datasets.

        Args:
            stage ([string], optional): setup expects a string arg stage. 
                                        It is used to separate setup logic for trainer.fit 
                                        and trainer.test. Defaults to None.
        """
        # Loop through all the different tubs and load all the records for each of them
        for tub in self.tubs:
            for underlying in tub:
                record = TubRecord(self.config, tub.base_path, underlying=underlying)
                self.records.append(record)

        train_records, val_records = train_test_split(self.records, test_size=(1. - self.config.TRAIN_TEST_SPLIT))
        # print("len train record: ",len(train_records))
        # print("len val record: ",len(val_records))
        # print(type(train_records))
        # print(type(train_records[0]))
        # print(train_records[0],"\n")

        assert len(val_records) > 0, "Not enough validation data. Add more data"

        self.train_dataset = myDataset(self.config, train_records, transform=self.transform)
        self.val_dataset = myDataset(self.config, val_records, transform=self.transform)
        # print("len val_dataset ",len(self.val_dataset))
        # print("len train_dataset: ",len(self.train_dataset))
        # img, label = self.train_dataset[0]
        # print("Shape train_dataset- img:",img.shape," label: ",label.shape)
        # # print(label,"\n",img)
        

    def train_dataloader(self):
        # The number of workers are set to 0 to avoid errors on Macs and Windows
        # See: https://github.com/rusty1s/pytorch_geometric/issues/366#issuecomment-498022534
        return DataLoader(self.train_dataset, batch_size=self.config.BATCH_SIZE, num_workers=0)

    def val_dataloader(self):
        # The number of workers are set to 0 to avoid errors on Macs and Windows
        # See: https://github.com/rusty1s/pytorch_geometric/issues/366#issuecomment-498022534
        return DataLoader(self.val_dataset, batch_size=self.config.BATCH_SIZE, num_workers=0)
