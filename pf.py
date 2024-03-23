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
        self.particles = np.array([np.random.multivariate_normal(
            self._init_mean.ravel(), self._init_cov) for _ in range(self.num_particles)])
        self.weights = np.ones(self.num_particles) / self.num_particles

    def move_particles(self, env, u):
        """Update particles after taking an action u: action"""
        new_particles = np.zeros_like(self.particles)
        for i in range(self.num_particles):
            noise_u = env.sample_noisy_action(u, self.alphas)
            new_particles[i] = env.forward(self.particles[i], noise_u).ravel()
        return new_particles

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving
        a landmark observation. u: action z: landmark observation
        marker_id: landmark ID"""
        self.particles = self.move_particles(env, u)
        for i in range(self.num_particles):
            predicted_observation = env.observe(self.particles[i], marker_id)
            self.weights[i] *= Field.likelihood(z - predicted_observation, self.beta)

        self.weights += 1.e-300      # avoid round-off to zero
        self.weights /= np.sum(self.weights)  # normalize
        self.particles = self.resample(self.particles, self.weights)
        mean, cov = self.mean_and_variance(self.particles)

        return mean, cov

    def resample(self, particles, weights):
        """Sample new particles and weights given current particles and weights."""
        indices = []
        C = [0.] + [np.sum(weights[:i+1]) for i in range(len(weights))]
        u0, j = np.random.random(), 0
        for u in [(u0 + i) / len(weights) for i in range(len(weights))]:
            while u > C[j]:
                j += 1
            indices.append(j-1)
        resampled_particles = particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles  # reset weights
        return resampled_particles

    def mean_and_variance(self, particles):
        """Compute the mean and covariance matrix for a set of equally-weighted
        particles."""
        mean = np.mean(particles, axis=0)
        mean[2] = np.arctan2(
            np.sum(np.sin(particles[:, 2])),
            np.sum(np.cos(particles[:, 2])),
        )
        mean[2] = Field.minimized_angle(mean[2])

        zero_mean_particles = particles - mean
        for i in range(zero_mean_particles.shape[0]):
            zero_mean_particles[i, 2] = Field.minimized_angle(zero_mean_particles[i, 2])

        cov = np.cov(zero_mean_particles.T) + np.eye(particles.shape[1]) * 1e-6  # Avoid singular matrix

        return mean.reshape((-1, 1)), cov
