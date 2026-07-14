# October 2021
# J. Rossbroich
#
# Analysis tools for a single experiment run, stored in a single working 
# directory as created by Hydra.
#
# The Script used here assumes the following structure for single-run directories:
#   
#   run_dir
#       -> .hydra
#           -> config.yaml
#           -> hydra.yaml
#           -> overrides.yaml


from pathlib import Path
import os
from typing import Iterable
from warnings import WarningMessage
from numpy import isin
from omegaconf import OmegaConf
import json

import pandas as pd
import pickle
import orbax.checkpoint



def dict_from_overrides(filepath):
    """
    Extracts dot-notated variable and value pairs 
    from a `overrides.yaml` file created by hydra
    """

    # Load OmegaConf config file
    overrides = OmegaConf.load(filepath)

    out = {item.split("=")[0] : item.split("=")[1] for item in overrides}

    return out


class HydraRunOutput:
    def __init__(self, 
                 path,
                 result_files=None,
                 hydra_folder_name='.hydra',
                 conf_file='config.yaml',
                 override_file='overrides.yaml'
                 ) -> None:
        """
        The results of a single experiment run created by `hydra`
        General class that can be re-used across projects

        Parameters
            :param path:                Path to the run directory containing a .hydra folder
            :param result_files:        Additional JSON result files (e.g. 'results.json') in the path
            :param hydra_folder_name:   Which folder to look for hydra configuration in
            :param conf_file:           Name of the config file inside the hydra folder
            :param override_file:       Name of the file containing CL overrides in this simulation
        """
        
        # Store folder structure information
        self.path = Path(path)
        
        self._dir_contents = [Path(f) for f in os.scandir(path)]
        self._filelist = [f.name for f in self._dir_contents if f.is_file()]
        self._dirlist = [f.name for f in self._dir_contents if f.is_dir()]
        
        # Assert that .hydra folder is present
        assert self.contains_dir(hydra_folder_name), "Directory does not contain a {} folder with hydra config".format(hydra_folder_name)

        self._conf_file = self.path / hydra_folder_name / conf_file
        self._override_file = self.path / hydra_folder_name / override_file
        
        # Additional result files
        self._result_files = []
        
        if result_files is not None:
            
            if isinstance(result_files, (list, tuple)):
            
                for file in result_files:
                    assert self.contains_file(file), "Directory does not contain {}".format(result_files)

                self._result_files = [self.path / file for file in result_files]
                
            elif isinstance(result_files, str):
                assert self.contains_file(result_files), "Directory does not contain {}".format(result_files)
                self._result_files = [self.path / result_files]
                
    def __getitem__(self, key):
        """ Looks for key in config file and all result dictionaries """
        
        # try to access it through a dot-notated variable key
        d, k, v = self.config._select_impl(key, False, False)

        # return if the key exists
        if v is not None:
            return d[k]
        
        else:
            return self._all_dicts[key]

    def contains(self, name):
        """
        Checks whether folder contains file or folder named 'name'
        """
        return name in [f.name for f in self._dir_contents] 

    def contains_file(self, name):
        """
        Checks whether folder contains file or folder named 'name'
        """
        return name in [f for f in self._filelist] 

    def contains_dir(self, name):
        """
        Checks whether folder contains file or folder named 'name'
        """
        return name in [f for f in self._dirlist] 

    def reload(self):
        self._dir_contents = [f for f in os.scandir(self.path)]
        self._filelist = [f.name for f in self._dir_contents if f.is_file()]
        self._dirlist = [f.name for f in self._dir_contents if f.is_dir()]
    
    def load_pickle(self, filename):
        assert self.contains_file(filename), "path does not contain {}.".format(filename)
        filepath = self.path / filename
        file = open(filepath, 'rb')
        return pickle.load(file)
    
    def load_trainstate(self, dirname):
        assert self.contains_dir(dirname), "path does not contain {}.".format(dirname)
        filepath = os.path.abspath(self.path / dirname)
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        return orbax_checkpointer.restore(filepath)
    
    @property
    def content(self):
        return self._dir_contents
    
    @property
    def config(self):
        return OmegaConf.load(self._conf_file)
    
    @property
    def overrides(self):
        return dict_from_overrides(self._override_file)
    
    @property
    def results(self):
        
        out = {}
        for file in self._result_files:
            
            # if file is json
            if file.suffix == '.json':
                out.update(json.load(open(file)))
            elif file.suffix == '.pkl':
                out.update(pickle.load(open(file, 'rb')))
            
        return out
    
    @property
    def _all_dicts(self):
        dicts = {}
        dicts.update(self.config)
        dicts.update(self.results)
        dicts.update(self.overrides)
        return dicts
    
    def get(self, keys='default'):
        """
        Returns a dictionary with all configuration and result 
        keys passed into the `keys` parameter
        """
                
        # by default, return the overridden config keys and all the result dictionaries
        if keys == 'default':
            out = {}
            out.update({key: self[key] for key in self.overrides.keys()})
            out.update(self.results)
            
        else:
            out = {key: self[key] for key in keys}
        
        return out
            
