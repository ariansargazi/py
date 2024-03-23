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
        """Update particles after taking an action.

        u: action
        """
        for i in range(self.num_particles):
            noisy_u = env.sample_noisy_action(u, self.alphas).ravel()
            self.particles[i, :] = env.forward(self.particles[i, :], noisy_u).ravel()
        return self.particles

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving
        a landmark observation.

        u: action
        z: landmark observation
        marker_id: landmark ID
        """
        self.particles = self.move_particles(env, u)

        # Update weights based on observation likelihood
        for i in range(self.num_particles):
            predicted_observation = env.observe(self.particles[i, :], marker_id)
            innovation = z - predicted_observation
            self.weights[i] = env.likelihood(innovation, self.beta)

        # Normalize weights
        self.weights += 1.e-300      
        self.weights /= sum(self.weights)

        self.particles = self.resample(self.particles, self.weights)
        mean, cov = self.mean_and_variance(self.particles)

        return mean, cov

    def resample(self, particles, weights):
        """Sample new particles and weights given current particles and weights.
        Implements the low-variance sampler.

        particles: (n x 3) matrix of poses
        weights: (n,) array of weights
        """
        N = len(particles)
        indices = []
        C = [0.] + [sum(weights[:i+1]) for i in range(N)]
        u0, j = np.random.random(), 0
        for u in [(u0 + n) / N for n in range(N)]:
            while u > C[j]:
                j += 1
            indices.append(j-1)
        return particles[indices]

    def mean_and_variance(self, particles):
        """Compute the mean and covariance matrix for a set of equally-weighted
        particles.

        particles: (n x 3) matrix of poses
        """
        mean = np.average(particles, weights=self.weights, axis=0)
        mean[2] = np.arctan2(
            np.sum(np.sin(particles[:, 2])*self.weights),
            np.sum(np.cos(particles[:, 2])*self.weights)
        )

        zero_mean = particles - mean
        for i in range(zero_mean.shape[0]):
            zero_mean[i, 2] = Field.minimized_angle(zero_mean[i, 2])
        cov = np.dot((zero_mean*self.weights[:, np.newaxis]).T, zero_mean) / sum(self.weights)
        cov += np.eye(3) * 1e-6

        return mean.reshape((-1, 1)), cov
