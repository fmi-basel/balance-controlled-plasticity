import jax.numpy as jnp
import tensorflow as tf   
import tensorflow_datasets as tfds
import jax

from .dataloader import DatasetLoader

from typing import Optional

# logging
import logging
logger = logging.getLogger(__name__)

def normalize_image(image, label):
    
    image = tf.cast(image, jnp.float32) / 255.
    label = tf.one_hot(label, 10)
    
    return image, label

def soft_targets(one_hot_labels, soft_target_val):
    dim_out = 10
    return one_hot_labels * (soft_target_val - (1 - soft_target_val) / (dim_out - 1)) + (1 - soft_target_val) / (dim_out - 1)    
    

class FashionMNISTDataset(DatasetLoader):
    
    dim_input: int = 28*28*1
    dim_output: int = 10
    task: str = 'classification'
    name: str = 'FashionMNIST' 
    valid_split: float = 0.1
    has_valid_data: bool = True
    soft_targets: bool = False
    soft_target_val: float = 0.99
    
    
    def __init__(self, 
                 soft_targets: bool = False,
                 soft_target_val: float = 0.99,
                 **kwargs):
        
        self.soft_targets = soft_targets
        self.soft_target_val = soft_target_val
        
        super().__init__(**kwargs)
        
        # Convert valid_split from proportion to percentage integer
        valid_split = int(self.valid_split * 100)
        
        # strings for tfds.load
        self.valid_split_str = f"train[:{valid_split}%]"
        self.train_split_str = f"train[{valid_split}%:]" 
    
    
    def get_train_data(self, batchsize, rng=None, flatten=False, OL_eval_subset=False):
        
        logger.info(f"Loading Fashion-MNIST training data from {self.path}")
    
        # prevent any GPU memory allocation
        tf.config.experimental.set_visible_devices([], 'GPU')
        
        train_data = tfds.load('fashion_mnist',
                            split=self.train_split_str,
                            data_dir=self.path,
                            shuffle_files=True,
                            as_supervised=True)
        
        if rng is None:
            seed = None
        elif isinstance(rng, int):
            seed = rng
        elif isinstance(rng, jax.random.PRNGKey):
            seed = int(rng[0])
        else:
            raise ValueError('Unknown type for rng')
        
        train_data = train_data.map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)
        if flatten:
            train_data = train_data.map(lambda x, y: (tf.reshape(x, (28*28,)), y))
        else:
            train_data = train_data.map(lambda x, y: (tf.reshape(x, (28,28,1)), y))
        
        if self.soft_targets:
            train_data = train_data.map(lambda x, y: (x, soft_targets(y, self.soft_target_val)))
            
        train_data = train_data.shuffle(tf.data.experimental.cardinality(train_data), seed=seed)
        train_data = train_data.batch(batchsize)
        train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

        logger.info(f"Number of samples in training data: {tf.data.experimental.cardinality(train_data) * batchsize}")

        # Subset for OL evaluation
        if OL_eval_subset:
            self.OL_eval_size = int(int(tf.data.experimental.cardinality(train_data)) * self.OL_eval_subset_split)
            train_data_OL_eval = train_data.take(self.OL_eval_size)     
            
            logger.info(f"Number of samples in training data for OL evaluation: {tf.data.experimental.cardinality(train_data_OL_eval) * batchsize}")   
        
            return tfds.as_numpy(train_data), tfds.as_numpy(train_data_OL_eval)
                        
        return tfds.as_numpy(train_data)
    
    
    def get_test_data(self, batchsize=None, flatten=False):

        logger.info(f"Loading Fashion-MNIST test data from {self.path}")
        
        test_data = tfds.load('fashion_mnist',
                            split='test',
                            data_dir=self.path,
                            as_supervised=True)
        
        test_data = test_data.map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)
        if flatten:
            test_data = test_data.map(lambda x, y: (tf.reshape(x, (28*28,)), y))
        else:
            test_data = test_data.map(lambda x, y: (tf.reshape(x, (28,28,1)), y))
    
        if self.soft_targets:
            test_data = test_data.map(lambda x, y: (x, soft_targets(y, self.soft_target_val)))
    
        test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)
        
        if batchsize is not None:
            test_data = test_data.batch(batchsize)
        else:
            test_data = test_data.batch(tf.data.experimental.cardinality(test_data))

        logger.info(f"Number of samples in test data: {tf.data.experimental.cardinality(test_data) * batchsize}")
            
        return tfds.as_numpy(test_data)


    def get_valid_data(self, batchsize = None, flatten=False):
        
        logger.info(f"Loading Fashion-MNIST validation data from {self.path}")
        
        valid_data = tfds.load('fashion_mnist',
                                 split=self.valid_split_str,
                                 data_dir=self.path,
                                 as_supervised=True)
        
        valid_data = valid_data.map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)
        
        if flatten:
            valid_data = valid_data.map(lambda x, y: (tf.reshape(x, (28*28,)), y))
        else:
            valid_data = valid_data.map(lambda x, y: (tf.reshape(x, (28,28,1)), y))
        
        if self.soft_targets:
            valid_data = valid_data.map(lambda x, y: (x, soft_targets(y, self.soft_target_val)))
            
        valid_data = valid_data.prefetch(tf.data.experimental.AUTOTUNE)
        
        if batchsize is not None:
            valid_data = valid_data.batch(batchsize)
        else:
            valid_data = valid_data.batch(tf.data.experimental.cardinality(valid_data))

        logger.info(f"Number of samples in validation data: {tf.data.experimental.cardinality(valid_data) * batchsize}")

        return tfds.as_numpy(valid_data)    
    
    
    def get_mock_data(self, 
                      batchsize: int = None, 
                      rng: Optional[int] = None,
                      flatten: bool = False) -> tuple:
        
        if rng is None:
            rng = jax.random.PRNGKey(0)
        elif isinstance(rng, int):
            rng = jax.random.PRNGKey(rng)
                    
        inputs, targets = super().get_mock_data(batchsize=batchsize, rng=rng)
        
        # Reshape input to 28x28x1 if not flattened
        if not flatten:
            inputs = inputs.reshape((28, 28, 1))
        
        return inputs, targets