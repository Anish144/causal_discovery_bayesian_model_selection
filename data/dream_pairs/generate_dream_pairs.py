from concurrent.futures import process
import os
from typing import Tuple, Generator
import numpy as np
from tqdm import tqdm
import csv


class DreamPairs:
    def __init__(self, name, path='./pairs/files'):
        """
        name should be in ["D4S1", "D4S2A", "D4S2B", "D4S2C"]
        """
        self.data = dict()
        if name not in ["D4S1", "D4S2A", "D4S2B", "D4S2C"]:
            raise NotImplementedError(f"{name} is not a valid DREAM4 dataset")
        file_pairs = f"{path}/{name}.csv"
        with open(file_pairs, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            headers = next(reader)
            data = np.array(list(reader))[:, 1:]

        processed_data = np.zeros((data.shape[0], len(data[0, 0].split()), 2))
        target_data = np.zeros((data.shape[0], 1))
        idx = 0
        for i in range(data.shape[0]):
            x = np.array(data[i, 0].split()).astype(float)
            y = np.array(data[i, 1].split()).astype(float)
            target = data[i, 2].astype(float)
            if target != 0:
                processed_data[idx, :, 0],  processed_data[idx, :, 1] = x, y
                target_data[idx] = target
                idx += 1
            else:
                continue

        self.data['cause'] = processed_data[:idx, :, 0:1]
        self.data['effect'] = processed_data[:idx, :, 1:2]
        self.data['weight'] = np.ones(data.shape[0])[:idx]

        self.data['target'] = target_data[:idx]

    def return_pairs(self) -> Generator[
        Tuple[np.ndarray, np.ndarray, float], None, None
    ]:
        """
        Produce a generator object that will yield each of the cause-effect
        pair datsets.

        Weight factor is also returned to weight the significance of the pair
        within the whole dataset (used to account for effectively duplicate
        data across dataset pairs).
        :return: Tuple of (cause, effect, weight) - cause, effect are 2D numpy
        arrays
        """
        return self.data['cause'], self.data['effect'], self.data['weight'], self.data["target"]
