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
            translation_noise = np.random.normal(0, self.alphas[0] * abs(u[0]) + self.alphas[1] * abs(u[1]))
            rotation_noise = np.random.normal(0, self.alphas[2] * abs(u[1]) + self.alphas[3] * (abs(u[0]) + abs(u[1])))
            noise = np.array([translation_noise, 0, rotation_noise])
            self.particles[i, :] = env.forward(self.particles[i, :].reshape(-1, 1), u).ravel() + noise
        return self.particles

    def update(self, env, u, z, marker_id):
        self.particles = self.move_particles(env, u)
        weights_sum = 0

        for i in range(self.num_particles):
            expected_z = env.observe(self.particles[i, :].reshape(-1, 1), marker_id)
            innovation = z - expected_z
            self.weights[i] *= (env.likelihood(innovation, self.beta) + 1e-12)
            weights_sum += self.weights[i]

        # Normalize weights safely
        if weights_sum > 0:
            self.weights /= weights_sum
        else:
            self.weights = np.ones(self.num_particles) / self.num_particles  # Reset weights if sum is 0
        N_eff = 1 / np.sum(self.weights ** 2)
        if N_eff < self.num_particles / 2:
            self.particles = self.resample(self.particles, self.weights)

        mean, cov = self.mean_and_variance(self.particles)
        return mean, cov

    def resample(self, particles, weights):
        N = self.num_particles
        indexes = np.zeros(N, dtype=int)
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.0  # Ensure sum is exactly 1
        positions = (np.arange(N) + np.random.uniform()) / N

        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return particles[indexes, :]

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
