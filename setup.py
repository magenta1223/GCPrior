from setuptools import setup, find_packages

setup(
    name='LVD',
    version='0.1',
    description='TEST',
    author='magenta1223',
    author_email='donghunikim8@gmail.com',
    requires=[],
    # install_requires=['wandb', 'dm-haiku', 'optax', 'gym==0.21', 'fire'],
    install_requires = [],
    packages=find_packages(exclude=[])
)