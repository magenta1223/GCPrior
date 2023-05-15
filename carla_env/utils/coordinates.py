import numpy as np


def cart2pol(x, y):
    """Convert cartesian to polar coordinates
    
    Args:
        x (float): x coordinate
        y (float): y coordinate
        
    Returns:
        (float, float): Polar coordinates (rho, phi)
        
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    """Convert polar to cartesian coordinates
    
    Args:
        rho (float): rho coordinate
        phi (float): phi coordinate
        
    Returns:
        (float, float): Cartesian coordinates (x, y)
        
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)
