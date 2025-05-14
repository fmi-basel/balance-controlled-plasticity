from .noise import OUProcess
import numpy as np
import jax.numpy as jnp
import time
import jax


def get_gaussian_tiling_weightvector(pre_idx, tuning, tuning_sigma, Npre, clip_lower):
    # calculate angular distances
    angular_distances = jnp.minimum(abs(tuning - pre_idx), Npre - abs(tuning - pre_idx))
    w = jnp.exp(-angular_distances**2 / (2 * tuning_sigma**2))
    w -= clip_lower
    return w.clip(0, None)

def weight_init_gaussian_tiling(Npre, Npost, tuning_sigma, clip_lower):
    weight_tuning = jnp.linspace(0, Npre, Npost, endpoint=False)
    w = jnp.stack([get_gaussian_tiling_weightvector(pre_idx, weight_tuning, tuning_sigma, Npre, clip_lower) for pre_idx in range(Npre)])
    return w

class PatternStimuli:
    def __init__(self,
                 dt: float,
                 Nb_patterns: int,
                 OU_tau: float,
                 OU_sigma: float,
                 OU_mu: float,
                 apply_threshold: bool=True,
                 OU_threshold: float = 0.0,
                 filter_tau: float = None,
                 filter_signals: bool = True,
                 linear_combination: bool = False,
                 Nb_inputs: int = None,
                 random_weights: bool = False,
                 rng_key: jax.random.PRNGKey = None):
        
        self.Nb_patterns = Nb_patterns
        self.OU_tau = OU_tau
        self.OU_sigma = OU_sigma
        self.OU_mu = OU_mu
        self.OU_threshold = OU_threshold
        self.dt = dt
        
        self._OUprocess = OUProcess(N=Nb_patterns,
                                    tau=OU_tau,
                                    mu=OU_mu,
                                    sigma=OU_sigma,
                                    dt=dt)
        
        self.filter_tau = filter_tau
        self.filter_signals = filter_signals
        self.apply_threshold = apply_threshold
            
        if linear_combination:
            
            # Gaussian tiling of the input space with gaussians (circular boundaries)
            if Nb_inputs is None:
                raise ValueError('Nb_inputs must be defined')
            
            if rng_key is None:
                rng_key = jax.random.PRNGKey(time.time())
                
            key1, key2 = jax.random.split(rng_key)
                
            if random_weights:
                self.W = np.array(jax.random.lognormal(key1, shape=(Nb_patterns, Nb_inputs))) * 0.5
            
            else:
                self.W = weight_init_gaussian_tiling(Npre=Nb_patterns, Npost=Nb_inputs, 
                                                     tuning_sigma=2*Nb_patterns/Nb_inputs, clip_lower=0.0)
                
            # normalize weights
            self.W = self.W / np.sum(self.W, axis=0)
            
            # random baselines (0.1 * lognormal)
            self.b = np.array(0.1 * jax.random.lognormal(key2, shape=(Nb_inputs,)))
            
        self.reset()     
                         
    def reset(self):
        self._OUprocess.reset()
        
    def simulate(self, timesteps, seed=None):
        
        pattern_activity = self._OUprocess.simulate(timesteps, seed)
        
        # apply threshold
        if self.apply_threshold:
            pattern_activity[pattern_activity < self.OU_threshold] = 0
        
        # Re-scale so that the mean is 1.0
        pattern_activity = pattern_activity / np.mean(pattern_activity)

        # Smooth the process
        if self.filter_signals:
            pattern_smooth = np.zeros_like(pattern_activity)
            pattern_smooth[0] = pattern_activity[0]
            for t in range(1, timesteps):
                pattern_smooth[t] = pattern_smooth[t-1] + 1 / self.filter_tau * (pattern_activity[t] - pattern_smooth[t-1]) * self.dt
        
        else:
            pattern_smooth = pattern_activity
            
        if hasattr(self, 'W'):
            inputs = np.dot(pattern_smooth, self.W) + self.b
            return pattern_smooth, inputs
        
        else:
            return pattern_smooth
        
    def get_x_given_p(self, p):
        if hasattr(self, 'W'):
            return np.dot(p, self.W) + self.b
        else:
            raise ValueError('No linear combination matrix W defined')
        