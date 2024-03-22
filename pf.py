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
        self.particles = np.random.multivariate_normal(
            self._init_mean.ravel(), self._init_cov, self.num_particles)
        self.weights = np.ones(self.num_particles) / self.num_particles

    def move_particles(self, env, u):
        """Move particles according to the action `u` and add motion noise."""
        for i in range(self.num_particles):
            # Sample motion to move the particle
            self.particles[i, :] = env.sample_noisy_action(self.particles[i, :], u).ravel()
        return self.particles

    def update(self, env, u, z, marker_id):
        """Update the state estimate after an action `u` and observation `z`."""
        self.move_particles(env, u)

        for i in range(self.num_particles):
            expected_z = env.observe(self.particles[i, :], marker_id)
            innovation = z - expected_z
            self.weights[i] = env.likelihood(innovation, self.beta)
        
        # Avoid division by zero and normalize the weights
        self.weights += 1.e-300
        self.weights /= np.sum(self.weights)

        self.particles = self.resample(self.particles, self.weights)
        mean, cov = self.mean_and_variance(self.particles)

        return mean, cov

    def resample(self, particles, weights):
        """Resample particles to focus on high-probability regions."""
        M = len(weights)
        indices = np.zeros(M, dtype=int)
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.0  # Ensure the sum is exactly 1
        u0 = np.random.random() * (1.0/M)
        position = u0
        i = 0
        for m in range(M):
            while position > cumulative_sum[i]:
                i += 1
            indices[m] = i
            position += 1.0/M
        return particles[indices]

    def mean_and_variance(self, particles):
        """Compute the mean and covariance matrix for a set of equally-weighted particles."""
        mean = np.average(particles, weights=self.weights, axis=0)
        mean[2] = np.arctan2(
            np.sum(np.sin(particles[:, 2])*self.weights),
            np.sum(np.cos(particles[:, 2])*self.weights)
        )
        zero_mean_particles = particles - mean
        for i in range(len(zero_mean_particles)):
            zero_mean_particles[i, 2] = Field.minimized_angle(zero_mean_particles[i, 2])
        cov = np.cov(zero_mean_particles.T, aweights=self.weights)
        return mean.reshape((-1, 1)), cov
