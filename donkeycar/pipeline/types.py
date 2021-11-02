import os
from typing import Any, List, Optional, TypeVar, Tuple

import numpy as np
import pandas as pd
from donkeycar.config import Config
from donkeycar.parts.tub_v2 import Tub
from donkeycar.utils import load_image, load_pil_image, train_test_split
from typing_extensions import TypedDict
from sklearn.model_selection import train_test_split

X = TypeVar('X', covariant=True)

TubRecordDict = TypedDict(
    'TubRecordDict',
    {
        'cam/image_array': str,
        'user/angle': float,
        'user/throttle': float,
        'user/mode': str,
        'imu/acl_x': Optional[float],
        'imu/acl_y': Optional[float],
        'imu/acl_z': Optional[float],
        'imu/gyr_x': Optional[float],
        'imu/gyr_y': Optional[float],
        'imu/gyr_z': Optional[float],
    }
)


class TubRecord(object):
    def __init__(self, config: Config, base_path: str,
                 underlying: TubRecordDict) -> None:
        self.config = config
        self.base_path = base_path
        self.underlying = underlying
        self._image: Optional[Any] = None

    def image(self, cached=True, as_nparray=True) -> np.ndarray:
        """Loads the image for you

        Args:
            cached (bool, optional): whether to cache the image. Defaults to True.
            as_nparray (bool, optional): whether to convert the image to a np array of uint8.
                                         Defaults to True. If false, returns result of Image.open()

        Returns:
            np.ndarray: [description]
        """
        if self._image is None:
            image_path = self.underlying['cam/image_array']
            full_path = os.path.join(self.base_path, 'images', image_path)

            if as_nparray:
                _image = load_image(full_path, cfg=self.config)
            else:
                # If you just want the raw Image
                _image = load_pil_image(full_path, cfg=self.config)

            if cached:
                self._image = _image
        else:
            _image = self._image
        return _image

    def __repr__(self) -> str:
        return repr(self.underlying)
    
    def getItem(self):
        return dict(self.underlying)


class TubDataset(object):
    """
    Loads the dataset, and creates a train/test split.
    """

    def __init__(self, config: Config, tub_paths: List[str],
                 shuffle: bool = True) -> None:
        self.config = config
        self.tub_paths = tub_paths
        self.shuffle = shuffle
        self.tubs: List[Tub] = [Tub(tub_path, read_only=True)
                                for tub_path in self.tub_paths]
        # self.records: List[TubRecord] = list()
        self.train_filter = getattr(config, 'TRAIN_FILTER', None)

    def train_test_split(self) -> Tuple[List[TubRecord], List[TubRecord]]:
        msg = f'Loading tubs from paths {self.tub_paths}' + f' with filter ' \
              f'{self.train_filter}' if self.train_filter else ''
        print(msg)
        # self.records.clear()
        train_data_list = []
        val_data_list = []
        
        for tub in self.tubs:
            angle_values = []
            record_dict = []
            for underlying in tub:
                angle_values.append(underlying['user/angle'])
                record_dict.append(underlying)
                # record = TubRecord(self.config, tub.base_path, underlying)
                # if not self.train_filter or self.train_filter(record):
                #     self.records.append(record)
            dt = pd.DataFrame()
            dt['user/angle'] = angle_values
            dt['record'] = record_dict
            bins_index = pd.cut(dt['user/angle'], 11, labels=False)
            train,  validation, _, _ = train_test_split(
                dt['record'], dt['user/angle'], stratify=bins_index, test_size=(1. - self.config.TRAIN_TEST_SPLIT))
            train_data = [TubRecord(self.config, tub.base_path, a) for a in train]
            val_data = [TubRecord(self.config, tub.base_path, a) for a in validation]
            train_data_list.extend(train_data)
            val_data_list.extend(val_data)
        
        return train_data_list, val_data_list
        # return train_test_split(self.records, shuffle=self.shuffle,
        #                         test_size=(1. - self.config.TRAIN_TEST_SPLIT))



