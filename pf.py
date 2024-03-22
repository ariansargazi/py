import numpy as np
from env import Field

class ParticleFilter:
    def __init__(self, mean, cov, num_particles, alphas, beta):
        self.alphas = alphas
        self.beta = beta
        self._init_mean = mean
        self._init_cov = cov
        self.num_particles = num_particles
        self.reset()

    def reset(self):
        self.particles = np.zeros((self.num_particles, 3))
        for i in range(self.num_particles):
            self.particles[i, :] = np.random.multivariate_normal(
                self._init_mean.ravel(), self._init_cov)
        self.weights = np.ones(self.num_particles) / self.num_particles


    def move_particles(self, env, u):
        """Move particles according to the action `u` and add motion noise."""
        for i in range(self.num_particles):
            # Apply the action to move the particle
            self.particles[i, :] = env.forward(self.particles[i, :].reshape(-1, 1), u).ravel()
            # Add noise, ensuring it is the correct shape (a 3-element 1D array)
            motion_noise = env.noise_from_motion(u, self.alphas)
            noise = np.random.multivariate_normal(np.zeros(3), motion_noise)
            self.particles[i, :] += noise
        return self.particles

    def update(self, env, u, z, marker_id):
        """Update the state estimate after an action `u` and observation `z`."""
        self.move_particles(env, u)

        for i in range(self.num_particles):
            expected_z = env.observe(self.particles[i, :].reshape(-1, 1), marker_id)
            innovation = z - expected_z
            # Update the weight, ensure likelihood is non-zero by adding a small constant
            self.weights[i] *= env.likelihood(innovation, self.beta) + 1e-12
        
        # Normalize the weights to sum to 1, avoiding division by zero
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            self.weights = np.ones(self.num_particles) / self.num_particles

        self.particles = self.resample()
        mean, cov = self.mean_and_variance()
        return mean, cov

    def resample(self):
        """Resample particles to focus on high-probability regions."""
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0  # Ensure sum is exactly 1
        indexes = np.searchsorted(cumulative_sum, np.random.rand(self.num_particles))
        # Use advanced NumPy indexing to select new set of particles
        resampled_particles = self.particles[indexes, :]
        self.weights = np.ones(self.num_particles) / self.num_particles  # Reset weights
        return resampled_particles

    def mean_and_variance(self, particles):
        """Compute the mean and covariance matrix for a set of equally-weighted
        particles.

        particles: (n x 3) matrix of poses
        """
        mean = particles.mean(axis=0)
        mean[2] = np.arctan2(
            np.sin(particles[:, 2]).sum(),
            np.cos(particles[:, 2]).sum(),
        )

        zero_mean = particles - mean
        for i in range(zero_mean.shape[0]):
            zero_mean[i, 2] = Field.minimized_angle(zero_mean[i, 2])
        cov = np.dot(zero_mean.T, zero_mean) / self.num_particles
        cov += np.eye(particles.shape[1]) * 1e-6  # Avoid bad conditioning

        return mean.reshape((-1, 1)), cov
