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
        """Update particles after taking an action
        
        u: action
        """
        for m in range(self.num_particles):
            # Apply motion model (sample noisy action)
            u_noisy = env.sample_noisy_action(u, self.alphas)
            # Update particle state
            self.particles[m] = env.forward(self.particles[m], u_noisy).ravel()
        return self.particles

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving
        a landmark observation.

        u: action
        z: landmark observation
        marker_id: landmark ID
        """
        # Move particles with sampled noisy motion
        self.move_particles(env, u)
        
        # Update weights based on observation likelihood
        for m in range(self.num_particles):
            predicted_z = env.observe(self.particles[m], marker_id)
            self.weights[m] = env.likelihood(z - predicted_z, self.beta)
        
        # Normalize weights
        self.weights += 1.e-300  # Avoid round-off to zero
        self.weights /= np.sum(self.weights)
        
        # Resample based on updated weights
        self.particles = self.resample(self.particles, self.weights)
        
        # Calculate mean and covariance
        mean, cov = self.mean_and_variance(self.particles)
        
        return mean, cov

    def resample(self, particles, weights):
        """Resample particles and weights using a low-variance sampler.

        particles: (n x 3) matrix of poses
        weights: (n,) array of weights
        """
        M = self.num_particles
        indexes = np.arange(self.num_particles)
        positions = (np.arange(self.num_particles) + np.random.random()) / M
        
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        resampled_particles = np.zeros_like(particles)
        
        while i < M:
            if positions[i] < cumulative_sum[j]:
                resampled_particles[i] = particles[j]
                i += 1
            else:
                j += 1
        return resampled_particles

    def mean_and_variance(self, particles):
        """Compute the mean and covariance matrix for a set of equally-weighted
        particles.

        particles: (n x 3) matrix of poses
        """
        mean = np.mean(particles, axis=0)
        mean[2] = np.arctan2(
            np.sum(np.sin(particles[:, 2])),
            np.sum(np.cos(particles[:, 2]))
        )

        zero_mean = particles - mean
        zero_mean[:, 2] = Field.minimized_angle(zero_mean[:, 2])
        cov = np.dot(zero_mean.T, zero_mean) / self.num_particles
        cov += np.eye(particles.shape[1]) * 1e-6  # Avoid bad conditioning

        return mean.reshape((-1, 1)), c
