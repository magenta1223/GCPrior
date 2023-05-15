"""Dataset utilities for CARLA environment"""

import pickle as pkl
from pathlib import Path
from typing import List, Union

import numpy as np
from typing_extensions import TypedDict


class Observations(TypedDict):
    sensor: np.ndarray
    image: np.ndarray


class Dataset(TypedDict):
    observations: Observations
    actions: np.ndarray
    rewards: np.ndarray
    terminals: np.ndarray
    infos: List[dict]
    lidar_bin: int


def load_datasets(dirname: Union[str, Path]):
    """Load all datasets in a directory
    
    Args:
        dirname (Union[str, Path]): Directory name
        
    Yields:
        Dataset: Dataset

    """
    if isinstance(dirname, str):
        dirname = Path(dirname)
    for filename in dirname.glob("*.pkl"):
        yield load_dataset(filename)


def load_dataset(filename: Union[str, Path]) -> Dataset:
    """Load a dataset from a file
    
    Args:
        filename (Union[str, Path]): File name
        
    Returns:
        Dataset: Dataset
        
    """
    with open(filename, "rb") as f:
        dataset = pkl.load(f)
    return dataset


def dump_dataset(dataset: Dataset, filename: Union[str, Path]):
    """Dump a dataset to a file
    
    Args:
        dataset (Dataset): Dataset
        filename (Union[str, Path]): File name
        
    """
    with open(filename, "wb") as f:
        pkl.dump(dataset, f)
