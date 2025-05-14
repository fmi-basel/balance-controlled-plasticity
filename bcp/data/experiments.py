# In-vivo-like experiments for BCP paper
# Julian Rossbroich
# 2025

import numpy as np
import jax
import jax.numpy as jnp

import time

from ..utils.noise import OUProcess
from diffrax import LinearInterpolation


class KrabbeExperiment:
    """
    Krabbe fear conditioning experiment.
    Wrapper for the KrabbeFearConditioning task that linearly interpolates the inputs and target.
    """
    
    def __init__(self, task):
        
        self.task = task
        self.x_ts = jnp.arange(0, self.task.T, self.task.dt)

    def simulate(self, key):
        x, y = self.task.simulate_pairing(key)
        
        # interpolation linear
        x_interp = LinearInterpolation(self.x_ts, x)
        y_interp = LinearInterpolation(self.x_ts, y)
        
        return x, y, x_interp, y_interp
    
    def simulate_test(self, key):
        x, y = self.task.simulate_test(key)
                
        # interpolation linear
        x_interp = LinearInterpolation(self.x_ts, x)
        y_interp = LinearInterpolation(self.x_ts, y)
        
        return x, y, x_interp, y_interp
    

class KrabbeFearConditioning:
    def __init__(
        self,
        N: int,
        T: float,
        T_CS: float,
        T_US: float,
        T_ITI: float,
        dt: float,
        noise: bool,
        noise_tau: float,
        noise_mu: float,
        noise_sigma: float,
        tau_rise: float,  # Added rise time constant for alpha kernel
        tau_decay: float,  # Added decay time constant for alpha kernel,
        us_input_strength: float = 1.0,
        key: jax.random.PRNGKey = None
    ):
        self.N = N
        self.T = T
        self.T_CS = T_CS
        self.T_US = T_US
        self.T_ITI = T_ITI
        self.dt = dt
        self.noise = noise
        self.tau_rise = tau_rise  # Store rise time constant
        self.tau_decay = tau_decay  # Store decay time constant
        self.us_input_strength = us_input_strength
        
        # RNG key
        if key is None:
            key = jax.random.PRNGKey(int(time.time()))
            
        # ASSURE T_CS + T_US + T_ITI = T
        assert T_CS + T_US + T_ITI == T, "T_CS + T_US + T_ITI must be equal to T"
        

        # CS+ and CS- are random (lognormal) weights on N//4 random inputs
        key1, key2 = jax.random.split(key)
        CSplus_weights = jax.random.lognormal(key1, shape=(N//4,))
        CSminus_weights = jax.random.lognormal(key2, shape=(N//4,))
        
        # choose 5 random indices for CS+ and CS- between 1 and 20
        CSplus_indices = jax.random.randint(key1, shape=(N//4,), minval=1, maxval=20)
        CSminus_indices = jax.random.randint(key2, shape=(N//4,), minval=1, maxval=20)
        
        # Create CS+ and CS- weights
        self.CSplus = jnp.zeros(N).at[CSplus_indices].set(CSplus_weights)
        self.CSminus = jnp.zeros(N).at[CSminus_indices].set(CSminus_weights)
        
        # Normalize CS+ and CS- weights to sum to 1
        self.CSplus = self.CSplus / jnp.sum(self.CSplus)
        self.CSminus = self.CSminus / jnp.sum(self.CSminus)
        
        # MAKE INPUTS
        # Total trial time is T
        # Starts with CS+ activity, then US, then ITI
        self.timesteps = int(T / dt)
        
        # Create unfiltered signals first
        unfiltered_csplus = jnp.zeros((self.timesteps, N))
        unfiltered_csminus = jnp.zeros((self.timesteps, N))
        
        # CS+ activity
        self.timesteps_CS = int(T_CS / dt)
        unfiltered_csplus = unfiltered_csplus.at[:self.timesteps_CS].set(self.CSplus)
        unfiltered_csminus = unfiltered_csminus.at[:self.timesteps_CS].set(self.CSminus)
        
        # Create alpha kernel for filtering
        alpha_kernel = self._create_alpha_kernel()
        
        # Apply alpha kernel filtering to CS+ and CS- activity
        self.csplus_activity = self._apply_alpha_filter(unfiltered_csplus, alpha_kernel)
        self.csminus_activity = self._apply_alpha_filter(unfiltered_csminus, alpha_kernel)
        
        # MAKE NOISE
        self.OU = OUProcess(N, tau=noise_tau, mu=noise_mu/N, sigma=noise_sigma, dt=dt)
        
        # MAKE TARGET = US
        self.US = jnp.zeros((self.timesteps, 1))
        self.timesteps_US = int(T_US / dt)
        self.US = self.US.at[self.timesteps_CS:self.timesteps_CS+self.timesteps_US].set(1)
        
        # activity for US at input in channel 0
        self.us_activity = jnp.zeros((self.timesteps, N))
        self.us_activity = self.us_activity.at[self.timesteps_CS:self.timesteps_CS+self.timesteps_US,0].set(us_input_strength)
        self.us_activity = self._apply_alpha_filter(self.us_activity, alpha_kernel)
        
        # simulate OU one time to get an initial state that is not zero
        if self.noise:
            self.OU.simulate(self.timesteps, seed=key)
    
    
    def _create_alpha_kernel(self):
        """Create an alpha kernel with specified rise and decay time constants"""
        # Create time vector
        # Make the kernel length 5x the slower time constant to capture most of the response
        max_tau = max(self.tau_rise, self.tau_decay)
        kernel_length = int(10 * max_tau / self.dt)
        t = jnp.arange(0, kernel_length * self.dt, self.dt)
        
        # Alpha function: f(t) = t/tau_rise * exp(1 - t/tau_rise)
        # Modified for separate rise and decay
        # f(t) = (t/tau_rise) * exp(-(t/tau_decay))
        alpha = (t / self.tau_rise) * jnp.exp(-(t / self.tau_decay))
        
        # Normalize to ensure kernel sums to 1
        alpha = alpha / jnp.sum(alpha)
        
        return alpha
    
    def _apply_alpha_filter(self, signal, kernel):
        """Apply alpha filter to signal using convolution"""
        # For each neuron, convolve the signal with the alpha kernel
        filtered_signal = jnp.zeros_like(signal)
        
        # Implement convolution for each neuron
        for n in range(self.N):
            # Use JAX's convolve function with 'valid' mode
            neuron_signal = signal[:, n]
            # Pad the signal to handle edge effects
            padded_signal = jnp.pad(neuron_signal, (len(kernel) - 1, 0))
            # Convolve and take the appropriate slice
            conv_result = jnp.convolve(padded_signal, kernel, mode='valid')
            # Ensure the result has the right length by padding or truncating
            if len(conv_result) < len(neuron_signal):
                conv_result = jnp.pad(conv_result, (0, len(neuron_signal) - len(conv_result)))
            elif len(conv_result) > len(neuron_signal):
                conv_result = conv_result[:len(neuron_signal)]
            filtered_signal = filtered_signal.at[:, n].set(conv_result)
        
        return filtered_signal
    
    @property
    def ts(self):
        return jnp.arange(0, self.T, self.dt)
    
    @property
    def csplus(self):
        return self.CSplus
    
    @property
    def csminus(self):
        return self.CSminus
    
    def simulate_pairing(self, key):
        """
        Simulate CS+ and US pairing
        """
        # make noise
        if self.noise:
            noise = self.OU.simulate(self.timesteps, seed=key)
        else:
            noise = jnp.zeros((self.timesteps, self.N))
        # inputs
        inputs = self.csplus_activity + noise + self.us_activity
        return inputs, self.US
    
    def simulate_test(self, key):
        """
        CS only, no US
        """
        # make noise
        if self.noise:
            noise = self.OU.simulate(self.timesteps, seed=key)
        else:
            noise = jnp.zeros((self.timesteps, self.N))
        # inputs
        inputs = self.csplus_activity + noise
        
        return inputs, self.US
    
    
class RenMotorLearning:
    
    def __init__(
        self,
        N: int,
        N_sines: int,
        T: float,
        t_cue: float,
        T_leverpress: float,
        dt: float,
        noise: bool,
        noise_tau: float,
        noise_mu: float,
        noise_sigma: float,
        tau_rise: float,  # Added rise time constant for alpha kernel
        tau_decay: float,  # Added decay time constant for alpha kernel,
        sine_minfreq: float = 0.1,
        sine_maxfreq: float = 2.0,
        key: jax.random.PRNGKey = None
    ):
        
        self.N = N
        self.T = T
        self.t_cue = t_cue      # start of cue
        
        self.dt = dt
        self.noise = noise
        # Total trial time is T
        self.timesteps = int(T / dt)
        
        self.noise_tau = noise_tau
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma
        self.tau_rise = tau_rise
        self.tau_decay = tau_decay
        
        # RNG key
        if key is None:
            key = jax.random.PRNGKey(int(time.time()))
        
        # MAKE CUE INPUTS
        # # # # # # # # # # # 
        
        # Random sine waves for cue activity
        key1, key2, key3 = jax.random.split(key, 3)
        cue_freqs = jax.random.uniform(key1, shape=(N_sines,), minval=sine_minfreq, maxval=sine_maxfreq)
        cue_phases = jax.random.uniform(key2, shape=(N_sines,), minval=0, maxval=2*jnp.pi)
        self.alpha_kernel = self._create_alpha_kernel()
        self.sine_waves = jnp.array([jnp.sin(2*jnp.pi*f*jnp.linspace(0, len(self.alpha_kernel)*dt, int(len(self.alpha_kernel)*dt/dt)) + p) for f, p in zip(cue_freqs, cue_phases)])
        self.sine_weights = jax.random.normal(key3, shape=(N_sines, N)) * 10
        self.cue_activity = jnp.dot(self.sine_waves.T, self.sine_weights).T * self.alpha_kernel
        
        # Cue activity
        self.timestep_start_cue = int(self.t_cue / self.dt)
        self.timestep_end_cue = self.timestep_start_cue + self.cue_activity.shape[1]
        
        self.cue_inputs = jnp.zeros((self.timesteps, self.N))
        self.cue_inputs = self.cue_inputs.at[self.timestep_start_cue:self.timestep_end_cue,:].set(self.cue_activity.T)
        # normalize cue-inputs so max is 1
        self.cue_inputs = self.cue_inputs / jnp.max(self.cue_inputs)
        
        # MAKE MOTOR OUTPUT
        # Generate time vector
        time_vec = np.arange(0, T, dt)  # shape (timesteps,)
        
        # Define motor command as a full sine-wave cycle active during movement.
        # The motor command is active from t_cue to t_cue+T_leverpress.
        motor_command = np.zeros_like(time_vec)
        active = (time_vec >= t_cue) & (time_vec <= t_cue + T_leverpress)
        # The sine wave makes one full cycle (0 to 2π) during T_leverpress.
        motor_command[active] = 1.0 * np.sin(2 * np.pi * (time_vec[active] - t_cue) / T_leverpress)
        
        # Compute lever trace as the cumulative numerical integration of the motor command.
        lever_trace = np.cumsum(motor_command) * dt
        # Subtract the offset so that the lever trace starts at zero at t_cue.
        idx_press = np.where(time_vec >= t_cue)[0][0]
        offset = lever_trace[idx_press]
        lever_trace = lever_trace - offset
        
        # Store as JAX arrays (reshape motor_output to shape (timesteps, 1))
        self.motor_output = - jnp.reshape(jnp.array(motor_command), (self.timesteps, 1))
        self.lever_output = - jnp.reshape(jnp.array(lever_trace), (self.timesteps, 1))
        
        # MAKE NOISE
        self.OU = OUProcess(N, tau=noise_tau, mu=noise_mu/N, sigma=noise_sigma, dt=dt)
        # simulate OU one time to get an initial state that is not zero
        if self.noise:
            self.OU.simulate(self.timesteps, seed=key)
        
    def _create_alpha_kernel(self):
        """Create an alpha kernel with specified rise and decay time constants"""
        # Create time vector
        # Make the kernel length 5x the slower time constant to capture most of the response
        max_tau = max(self.tau_rise, self.tau_decay)
        kernel_length = int(10 * max_tau / self.dt)
        t = jnp.arange(0, kernel_length * self.dt, self.dt)
        
        # Alpha function: f(t) = t/tau_rise * exp(1 - t/tau_rise)
        # Modified for separate rise and decay
        # f(t) = (t/tau_rise) * exp(-(t/tau_decay))
        alpha = (t / self.tau_rise) * jnp.exp(-(t / self.tau_decay))
        
        # Normalize to ensure kernel sums to 1
        alpha = alpha / jnp.sum(alpha)
        
        return alpha
    
    def _apply_alpha_filter(self, signal, kernel):
        """Apply alpha filter to signal using convolution"""
        # For each neuron, convolve the signal with the alpha kernel
        filtered_signal = jnp.zeros_like(signal)
        
        # Implement convolution for each neuron
        for n in range(self.N):
            # Use JAX's convolve function with 'valid' mode
            neuron_signal = signal[:, n]
            # Pad the signal to handle edge effects
            padded_signal = jnp.pad(neuron_signal, (len(kernel) - 1, 0))
            # Convolve and take the appropriate slice
            conv_result = jnp.convolve(padded_signal, kernel, mode='valid')
            # Ensure the result has the right length by padding or truncating
            if len(conv_result) < len(neuron_signal):
                conv_result = jnp.pad(conv_result, (0, len(neuron_signal) - len(conv_result)))
            elif len(conv_result) > len(neuron_signal):
                conv_result = conv_result[:len(neuron_signal)]
            filtered_signal = filtered_signal.at[:, n].set(conv_result)
        
        return filtered_signal
    
    def simulate(self, key):
        
        # make noise
        if self.noise:
            noise = self.OU.simulate(self.timesteps, seed=key)
        else:
            noise = jnp.zeros((self.timesteps, self.N))
            
        # inputs
        inputs = self.cue_inputs + noise
        return inputs, self.motor_output
    
class RenExperiment:
    """
    Ren et al. motor learning experiment.
    Wrapper for the RenMotorLearning task that linearly interpolates the inputs and target.
    """
    
    def __init__(self, task):
        
        self.task = task
        self.x_ts = jnp.arange(0, self.task.T, self.task.dt)

    def simulate(self, key):
        x, y = self.task.simulate(key)
        
        # interpolation linear
        x_interp = LinearInterpolation(self.x_ts, x)
        y_interp = LinearInterpolation(self.x_ts, y)
        
        return x, y, x_interp, y_interp
    