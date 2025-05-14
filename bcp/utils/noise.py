import numpy as np

class OUProcess:
    
    def __init__(self, N, tau, mu, sigma, dt):
        """
        Initialize the OUProcess class.

        Parameters:
        N (int): Number of presynaptic neurons.
        tau (float): Time constant of the Ornstein-Uhlenbeck process.
        mu (float): Mean of the noise.
        sigma (float): Standard deviation of the noise.
        dt (float): Time step.
        """
        
        self.N = N
        self.tau = tau
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.state = np.zeros(N)
        
    def reset(self):
        self.state = np.zeros(self.N)
    
    def simulate(self, timesteps, seed=None):
        """
        Simulate the Ornstein-Uhlenbeck process for a given number of timesteps.
        """
        
        if seed is None:
            np.random.seed()
        else:   
            np.random.seed(seed)
                    
        noise = np.random.normal(0, 1, (timesteps, self.N))
        ou_process = self._simulate_ou_process(noise)
        self.state = ou_process[-1]
        
        return ou_process
    
    def _simulate_ou_process(self, noise):
        """
        Simulate the Ornstein-Uhlenbeck process.
        """
        
        ou_process = np.zeros((noise.shape[0], self.N))
        ou_process[0] = self.state
        for i in range(1, noise.shape[0]):
            ou_process[i] = ou_process[i-1] + self.dt * (self.mu - ou_process[i-1]) / self.tau + self.sigma * np.sqrt(self.dt) * noise[i]
            
        return ou_process
    
