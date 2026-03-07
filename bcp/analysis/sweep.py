# October 2021
# J. Rossbroich
#
# Analysis tools for a sweep experiment started with `hydra` 

from pathlib import Path
import os
from warnings import WarningMessage
from omegaconf import OmegaConf
import json

import pandas as pd

from . import HydraRunOutput

def get_subdirs_containing(dir, content):
    """
    Recuirsively looks through a directory `dir` for folders
    that contain all files or folders in `content`

    Parameters
        :param dir:         Top level directory to search
        :param content:     Content to look for. A list of strings.
    """

    # Contents of head directory:
    dir_contents = [f for f in os.scandir(dir)]

    # Check whether it contains all necessary content:
    if all(c in [f.name for f in dir_contents] for c in content):
        return list([Path(dir)])

    # If not, recursively call this function and append to list
    else:

        # Loop through subfolders
        subfolders = [f.path for f in dir_contents if f.is_dir()]

        if len(subfolders) > 0:
            dirs = list()

            # Append dirs from all subfolders
            for subdir in subfolders:
                dirs.extend(get_subdirs_containing(subdir, content))

            return dirs

        # If no more subfolders, return empty list
        else:
            return list()


class HydraMultirun:
    def __init__(
        self, path, hydra_folder_name = '.hydra', **kwargs
    ) -> None:
        """
        Assumes a directory structure like:

        EXP_NAME
            --> a=0, b=0
                    ---> seed=0001
                        ---> .hydra
                            ---> ...
                    ---> seed=0002
                    ...

            --> a=0, b=1,
                    ---> seed=0001
                    ---> seed=0002
                    ...

        Parameters
            :param path:    Path to the EXP_NAME directory
        """

        # Store path information
        self.path = path

        # Get list of all subdirectories containing
        # a '.hydra' directory and eventual additional files specified by kwargs
        res_file_names = kwargs.get('result_files', [])
        self._run_paths = get_subdirs_containing(
            self.path, [hydra_folder_name] + res_file_names
        )
        
        self.runs = [HydraRunOutput(path, 
                                    hydra_folder_name=hydra_folder_name,
                                    **kwargs) for path in self._run_paths]

    def __len__(self):
        return len(self.runs)
    
    def __getitem__(self, idx):
        return self.runs[idx]

    def table(self, keys="default"):
        """
        Returns a pandas table with all configuration and result
        keys passed into the `keys` parameter
        """
        
        return pd.DataFrame([run.get(keys) for run in self.runs])
