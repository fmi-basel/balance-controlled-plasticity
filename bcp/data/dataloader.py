# Dataset loader base class
# Julian Rossbroich
# 2025

import jax
import jax.numpy as jnp

import tensorflow as tf
import tensorflow_datasets as tfds

from typing import Optional


class DatasetLoader():
    
    dim_input: int
    dim_output: int
    task: str 
    name: str 
    path: str = None
    valid_split: float = 0.0
    has_valid_data: bool = False
    OL_eval_subset_split: float = 0.1
    
    def __init__(self, 
                 dim_input: int, 
                 dim_output: int, 
                 task: str = "UndefinedTask",
                 name: str = "AbstractDataset",
                 path: str = None,
                 valid_split: float = 0.0,
                 **kwargs):
        
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.task = task
        self.name = name
        self.path = path
        self.valid_split = valid_split
    
    def get_mock_data(self, 
                      batchsize: int = None, 
                      rng: Optional[jax.random.PRNGKey]= None, 
                      **kwargs) -> tuple:
        """
        Returns a mock dataset of the correct shape.
        """
        if rng is None:
            rng = jax.random.PRNGKey(0)
        elif isinstance(rng, int):
            rng = jax.random.PRNGKey(rng)
            
        rng1, rng2 = jax.random.split(rng)
        
        if self.task == 'classification':
            outdim = 1
        else:
            outdim = self.dim_output
        
        if batchsize is None:
            inputs = jax.random.uniform(rng1, shape=(self.dim_input,))
            targets = jax.random.uniform(rng2, shape=(outdim,))
        else:
            inputs = jax.random.uniform(rng1,shape=((batchsize, self.dim_input)))
            targets = jax.random.uniform(rng2, shape=((batchsize, outdim)))
        
        return tuple([inputs, targets])
    
    def get_train_data(self, batchsize: int, flatten=False, OL_eval_subset=False, **kwargs):
        """
        Returns a training dataset
        """
        raise NotImplementedError
    
    
    def get_test_data(self, batchsize: Optional[int] = None, flatten=False, **kwargs):
        """
        Returns a test dataset
        """
        raise NotImplementedError
    
    
    def get_valid_data(self, batchsize: Optional[int] = None, flatten=False, **kwargs):
        """
        Returns a validation dataset
        """
        raise NotImplementedError