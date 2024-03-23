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
        self.particles = np.random.multivariate_normal(self._init_mean.ravel(), self._init_cov, self.num_particles)
        self.weights = np.ones(self.num_particles) / self.num_particles

    def move_particles(self, env, u):
        new_particles = np.array([env.sample_noisy_action(u, self.alphas).ravel() for _ in range(self.num_particles)])
        return new_particles

    def update(self, env, u, z, marker_id):
        # Move particles with sampled noisy motion
        new_particles = self.move_particles(env, u)
        
        # Update weights based on observation likelihood
        for m in range(self.num_particles):
            predicted_z = env.observe(new_particles[m], marker_id)
            self.weights[m] *= env.likelihood(z - predicted_z, self.beta)

        # Normalize weights
        self.weights += 1.e-300  # Avoid round-off to zero
        self.weights /= np.sum(self.weights)
        
        # Resample based on updated weights
        self.particles = self.resample(new_particles, self.weights)
        
        # Calculate mean and covariance
        mean, cov = self.mean_and_variance(self.particles)

        return mean, cov

    def resample(self, particles, weights):
        indices = np.random.choice(range(self.num_particles), size=self.num_particles, p=weights)
        resampled_particles = particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles  # Reset weights after resampling
        return resampled_particles

    def mean_and_variance(self, particles):
        mean = np.average(particles, weights=self.weights, axis=0)
        # For the angle, need to use atan2 to properly average
        mean[2] = np.arctan2(
            np.sum(np.sin(particles[:, 2]) * self.weights),
            np.sum(np.cos(particles[:, 2]) * self.weights)
        )
        # Compute covariance, need to account for circular statistics for angle
        zero_mean = particles - mean
        zero_mean[:, 2] = (zero_mean[:, 2] + np.pi) % (2 * np.pi) - np.pi  # Normalize angles
        cov = np.zeros((3, 3))
        for i in range(self.num_particles):
            cov += self.weights[i] * np.outer(zero_mean[i], zero_mean[i])
        cov /= np.sum(self.weights)  # Normalize covariance by the sum of weights
        return mean.reshape((-1, 1)), cov
