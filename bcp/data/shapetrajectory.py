# Trajectory of a shape in 2D space
from svgpathtools import svg2paths, Path
import numpy as np
import os
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation

import time

matplotlib.style.use('spiffy')


def embed_trajectory(X, N, key):
    '''
    Embeds a 2D trajectory X of shape [T, 2] into N dimensions, resulting in [T, N],
    such that when projected back to 2D via PCA, it resembles the original trajectory.
    '''

    T = X.shape[0]
        
    # Generate a random orthogonal matrix Q of shape [N, N]
    normal_matrix = jax.random.normal(key, (N, N))
    Q, R = jnp.linalg.qr(normal_matrix)
    
    # Adjust Q to ensure it's a proper rotation matrix with determinant = 1
    det = jnp.linalg.det(Q)
    Q = Q * det  # Adjust Q if necessary

    # Embed X into N dimensions by padding with zeros
    zeros_padding = jnp.zeros((T, N - 2))
    X_padded = jnp.concatenate([X, zeros_padding], axis=1)  # Shape [T, N]

    # Apply the random rotation
    X_embedded = X_padded @ Q.T  # Shape [T, N]

    return X_embedded, Q


class ShapeTrajectory:
    def __init__(self,
                 shape: str,
                 T: float,
                 N_points: int,
                 highDembedding: bool = False,
                 embedding_dim: int = 2,
                 path: str = 'data/shapes',
                 key: jax.random.PRNGKey = None):
        
        N = embedding_dim if highDembedding else 2
        dt = T / N_points
        
        # RNG key
        if key is None:
            key = jax.random.PRNGKey(int(time.time()))
                    
        self.T = T
        self.dt = dt
        self.N = N
        self.key = key
        self.path = path
        self.shape = shape
        self.N_points = N_points
        
        # Load the SVG file 
        # Look for path/shape.svg
        filepath = os.path.join(path, shape + '.svg')
        print(filepath)
        paths, _ = svg2paths(filepath)
        combined_path = Path(*[segment for p in paths for segment in p])

        # Sample points along the path
        points = np.array([combined_path.point(t) for t in np.linspace(0, 1, N_points)])
        
        # Extract X and Y coordinates
        x = points.real
        y = points.imag
        
        self.trajectory = np.vstack((x, y))
        
        # Center the trajectory around 0
        self.trajectory -= self.trajectory.mean(axis=1)[:, None]

        # Fit into unit sphere
        self.trajectory /= np.max(np.abs(self.trajectory))

        # Invert the Y-axis
        self.trajectory[1] *= -1
        
        # Convert to jax array and transpose to get shape [T, 2]
        self.trajectory = jnp.array(self.trajectory).T

        # High-dimensional embedding
        if highDembedding:
            self.Y, self.Q = embed_trajectory(self.trajectory, N, key)
        else:
            self.Y = self.trajectory
            self.Q = None
        
    @property
    def ts(self):
        return jnp.arange(0, self.T, self.dt)
    
    @property
    def embedding_Q(self):
        return self.Q       # shape [N, N] or None
    
    def get_trajectory(self):
        return self.trajectory
    
    def get_Y(self):
        return self.Y
    
    def save_plot(self, path = 'auto'):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.plot(self.trajectory[:,0], self.trajectory[:,1], c='black')
        ax.scatter(self.trajectory[:,0], self.trajectory[:,1], c=self.ts, cmap='viridis')
        
        if path == 'auto':
            path = os.path.join(self.path, self.shape + '_trajectory.png')
            
        plt.savefig(path)
        
        return fig
        
    def save_gif(self, path = 'auto'):
        
        # Create a new figure and set up the axes
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_axis_off()

        # Pre-set the axis limits to avoid recalculating them each frame
        ax.set_xlim(self.trajectory[:,0].min() * 1.1, self.trajectory[:,0].max() * 1.1)
        ax.set_ylim(self.trajectory[:,1].min() * 1.1, self.trajectory[:,1].max() * 1.1)

        # Initialize the plot elements; these will be updated in each frame
        line, = ax.plot([], [], c='black')
        scatter = ax.scatter([], [], c=[], cmap='viridis')
        
        def init():
            """Initialize the background of the animation."""
            line.set_data([], [])
            scatter.set_offsets(np.empty((0, 2)))
            scatter.set_array(np.array([]))
            return line, scatter

        def animate(t):
            """Update the plot for frame t."""
            x = self.trajectory[:t, 0]
            y = self.trajectory[:t, 1]
            line.set_data(x, y)
            scatter.set_offsets(np.column_stack((x, y)))
            scatter.set_array(self.ts[:t])
            return line, scatter

        # Create the animation
        ani = animation.FuncAnimation(
            fig, animate, frames=range(1, self.N_points+1), init_func=init, blit=False)

        if path == 'auto':
            path = os.path.join(self.path, self.shape + '_trajectory.gif')
        
        ani.save(path, writer='pillow', fps=20)
        
        return fig
