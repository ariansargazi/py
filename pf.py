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
        """Update particles after taking an action u."""
        for i in range(self.num_particles):
            noisy_u = env.sample_noisy_action(u, self.alphas).ravel()
            self.particles[i, :] = env.forward(self.particles[i, :], noisy_u).ravel()
        return self.particles

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action u and receiving a landmark observation z."""
        self.move_particles(env, u)
        for i in range(self.num_particles):
            predicted_z = env.observe(self.particles[i, :], marker_id).ravel()
            innovation = z.ravel() - predicted_z
            innovation[0] = Field.minimized_angle(innovation[0])
            self.weights[i] = env.likelihood(innovation.reshape(-1, 1), self.beta)
        self.weights += 1.e-300      # Avoid round-off to zero
        self.weights /= np.sum(self.weights)
        self.particles = self.resample(self.particles, self.weights)
        mean, cov = self.mean_and_variance(self.particles)
        return mean, cov

    def resample(self, particles, weights):
        """Resample particles according to the weights using a low-variance sampler."""
        M = self.num_particles
        indexes = np.zeros(M, 'i')
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.  # Avoid round-off error
        position = (np.arange(M) + np.random.rand()) / M

        i, j = 0, 0
        while i < M:
            if position[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return particles[indexes, :]

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
