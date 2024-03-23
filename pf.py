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
        new_particles = np.array([env.forward(particle, u.ravel()) for particle in self.particles])
        return new_particles

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving
        a landmark observation.

        u: action
        z: landmark observation
        marker_id: landmark ID
        """
        self.particles = self.move_particles(env, u)
        
        for m in range(self.num_particles):
            predicted_z = env.observe(self.particles[m], marker_id)
            self.weights[m] = env.likelihood(z - predicted_z, self.beta)

        self.weights += 1.e-300      # avoid round-off to zero
        self.weights /= sum(self.weights)  # normalize

        self.particles = self.resample(self.particles, self.weights)
        mean, cov = self.mean_and_variance(self.particles)

        return mean, cov

    def resample(self, particles, weights):
        """Sample new particles and weights given current particles and weights. Use the low-variance sampler from class.

        particles: (n x 3) matrix of poses
        weights: (n,) array of weights
        """
        M = self.num_particles
        resampled_particles = np.zeros(particles.shape)
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.0  # to make sure it's a valid cumulative distribution function
        index = np.searchsorted(cumulative_sum, np.random.rand())
        
        beta = 0.0
        mw = max(weights)
        for m in range(M):
            beta += np.random.rand() * 2.0 * mw
            while beta > weights[index]:
                beta -= weights[index]
                index = (index + 1) % M
            resampled_particles[m, :] = particles[index, :]

        return resampled_particles

    def mean_and_variance(self, particles):
        """Compute the mean and covariance matrix for a set of equally-weighted
        particles.

        particles: (n x 3) matrix of poses
        """
        mean = np.mean(particles, axis=0)
        mean[2] = np.arctan2(
            np.sum(np.sin(particles[:, 2])),
            np.sum(np.cos(particles[:, 2])),
        )

        zero_mean = particles - mean
        for i in range(zero_mean.shape[0]):
            zero_mean[i, 2] = Field.minimized_angle(zero_mean[i, 2])
        cov = np.dot(zero_mean.T, zero_mean) / self.num_particles
        cov += np.eye(particles.shape[1]) * 1e-6  # Avoid bad conditioning

        return mean.reshape((-1, 1)), cov
