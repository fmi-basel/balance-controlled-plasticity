# Tracker classes for the training process
# Trackers are used to record model variables and metrics during training

import pickle
import os

class WeightTracker():
    """ Tracks model weights during training """
    
    def __init__(self, paramnames = None):
        """
        :param paramnames: parameters to track
        """
        
        self.paramnames = paramnames
        self.rec = {}
        if paramnames is not None:
            for paramname in paramnames:
                self.rec[paramname] = []
        
    def update(self, state):
        
        if self.paramnames is None:
            self.paramnames = state.params['params'].keys()
            for paramname in self.paramnames:
                # if param has a bias term, track both kernel and bias
                if state.params['params'][paramname].get('bias') is not None:
                    self.rec[paramname] = {'kernel': [], 'bias': []}
                else:
                    self.rec[paramname] = {'kernel': []}
        
        params = state.params['params']
        for paramname in self.paramnames:
            self.rec[paramname]['kernel'].append(params[paramname]['kernel'])
            if params[paramname].get('bias') is not None:
                self.rec[paramname]['bias'].append(params[paramname]['bias'])
            
    def get(self, paramname = None):
        
        if paramname is None:
            out = self.rec
        else:
            out = self.rec[paramname]
        
        return out
    
    def save(self, filename):
        
        with open(filename, 'wb') as f:
            pickle.dump(self.rec, f)
            

class BatchWiseTracker():
    
    def __init__(self, rec_dir):
        self.epoch_count = 0
        self.batch_count = 0
        
        self.rec = []
        
        # Create directory for saving data if it doesn't exist
        if not os.path.exists(rec_dir):
            os.makedirs(rec_dir)
        
        self.rec_dir = rec_dir

    def record_batch(self, train_state, vf_sol, batch_metrics):
        
        # Write vf_sol directly to disk
        filename = os.path.join(self.rec_dir, 
                                'epoch_{}_batch_{}.pkl'.format(self.epoch_count, 
                                                               self.batch_count))
        
        with open(filename, 'wb') as f:
            pickle.dump(vf_sol, f)
            
        self.batch_count += 1
        
# TRACKERS FOR DEBUGGING
# # # # # # # # # # # # #
            
class BatchWiseSolTracker():
    """
    Tracks the `solution` object of the vector field dynamics for each minibatch.
    This leads to **a lot** of data, so use with caution.
    """
    
    def __init__(self, rec_dir):
        
        self.epoch_count = 0
        self.batch_count = 0
        
        self.rec = []
        
        # Create directory for saving data if it doesn't exist
        if not os.path.exists(rec_dir):
            os.makedirs(rec_dir)
        
        self.rec_dir = rec_dir
    
    def record_batch(self, train_state, vf_sol, batch_metrics):
        
        # Write vf_sol directly to disk
        filename = os.path.join(self.rec_dir, 
                                'epoch_{}_batch_{}.pkl'.format(self.epoch_count, 
                                                               self.batch_count))
        
        with open(filename, 'wb') as f:
            pickle.dump(vf_sol, f)
            
        self.batch_count += 1
        
    def record_epoch(self, train_state, epoch_metrics):
        
        self.epoch_count += 1
    
    def save(self, filename):
        
        with open(filename, 'wb') as f:
            pickle.dump(self.rec, f)