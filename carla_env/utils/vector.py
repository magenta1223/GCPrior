from typing import Union

import carla
import numpy as np


def to_array(
    array_like: Union[carla.Vector3D, carla.Rotation]
):
    """Convert a carla.Vector3D or carla.Rotation to a numpy array
    
    Args:
        array_like (Union[carla.Vector3D, carla.Rotation]): Array-like object
        
    Returns:
        np.ndarray: Numpy array
        
    Raises:
        TypeError: If the input is not a carla.Vector3D or carla.Rotation
        
    """
    if isinstance(array_like, carla.Vector3D):
        return vector_to_array(array_like)

    if isinstance(array_like, carla.Rotation):
        return rotation_to_array(array_like)

    raise TypeError(
        f"Expected array-like object, got {type(array_like)}"
    )


def vector_to_array(vector: carla.Vector3D):
    """Convert a carla.Vector3D to a numpy array
    
    Args:
        vector (carla.Vector3D): Vector
        
    Returns:
        np.ndarray: Numpy array
        
    """
    return np.array([vector.x, vector.y, vector.z])


def rotation_to_array(rotation: carla.Rotation):
    """Convert a carla.Rotation to a numpy array
    
    Args:
        rotation (carla.Rotation): Rotation
        
    Returns:
        np.ndarray: Numpy array
        
    """
    return np.array([rotation.pitch, rotation.yaw, rotation.roll])
