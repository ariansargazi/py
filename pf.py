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
        for i in range(self.num_particles):
            noise = np.random.multivariate_normal(np.zeros(3), env.noise_from_motion(u, self.alphas))
            self.particles[i, :] = env.forward(self.particles[i, :].reshape(-1, 1), u).ravel() + noise
        return self.particles

    def update(self, env, u, z, marker_id):
        self.particles = self.move_particles(env, u)

        for i in range(self.num_particles):
            expected_z = env.observe(self.particles[i, :].reshape(-1, 1), marker_id)
            innovation = z - expected_z
            self.weights[i] *= env.likelihood(innovation, self.beta)
        
        max_weight = np.max(self.weights)
        if max_weight > 0:
            self.weights /= max_weight
        self.weights /= np.sum(self.weights)
        
        
        N_eff = 1 / np.sum(self.weights ** 2)
        if N_eff < self.num_particles / 2:
            self.particles = self.resample(self.particles, self.weights)
        
        mean, cov = self.mean_and_variance(self.particles)
        return mean, cov

    def resample(self, particles, weights):
        N = self.num_particles
        positions = (np.arange(N) + np.random.uniform()) / N
        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return particles[indexes

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
