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
        new_particles = self.particles

        # YOUR CODE HERE
        new_particles = np.array([env.sample_noisy_action(u, self.alphas) for _ in range(self.num_particles)])
        return new_particles

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving
        a landmark observation.

        u: action
        z: landmark observation
        marker_id: landmark ID
        """
        self.particles = self.move_particles(env, u)
        # YOUR CODE HERE

        for i in range(self.num_particles):
        expected_z = env.observe(self.particles[i, :].reshape(-1, 1), marker_id)
        innovation = z - expected_z
        self.weights[i] = env.likelihood(innovation, self.beta)

        self.weights += 1.e-300  # Avoid division by zero
        self.weights /= np.sum(self.weights)

        self.particles = self.resample(self.particles, self.weights)
        mean, cov = self.mean_and_variance(self.particles) 

        return mean, cov

    def resample(self, particles, weights):
        """Sample new particles and weights given current particles and weights. Be sure
        to use the low-variance sampler from class.

        particles: (n x 3) matrix of poses
        weights: (n,) array of weights
        """
        # YOUR CODE HERE
        N = len(particles)
        indices = []
        C = np.cumsum(weights)
        r = np.random.uniform(0, 1/N)
        i = 0
        u = r - 1/N
        for m in range(N):
          u += 1/N
        while u > C[i]:
          i += 1
         indices.append(i)
        
        resampled_particles = particles[indices]
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
