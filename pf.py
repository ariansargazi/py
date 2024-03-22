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
        self.particles = np.array([np.random.multivariate_normal(self._init_mean.ravel(), self._init_cov) for _ in range(self.num_particles)])
        self.weights = np.ones(self.num_particles) / self.num_particles

    def move_particles(self, env, u):
        """Update particles after taking an action

        u: action
        """
        # YOUR CODE HERE
        for i in range(self.num_particles):
            self.particles[i, :] = env.sample_noisy_action(self, u, alphas=None).ravel()
            forward(self, x, u):
        return self.particles

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

        # Normalize the weights
        self.weights += 1.e-300  # Avoid division by zero
        self.weights /= np.sum(self.weights)

        self.particles = self.resample(self.particles, self.weights)
        mean, cov = self.mean_and_variance(self.particles)

        return mean, cov

def resample(self, particles, weights):
    # YOUR CODE HERE
    N = len(particles)
    positions = (np.arange(N) + np.random.random()) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1

    resampled_particles = particles[indexes]
    self.weights = np.ones(self.num_particles) / self.num_particles  # Reset weights

    return resampled_particles

    def mean_and_variance(self, particles):
        """Compute the mean and covariance matrix for a set of equally-weighted
        particles.

        particles: (n x 3) matrix of poses
        """
        mean = np.mean(particles, axis=0)
        # Handling the circular nature of the angle
        mean[2] = np.arctan2(np.sum(np.sin(particles[:, 2])), np.sum(np.cos(particles[:, 2])))

        zero_mean = particles - mean
        for i in range(zero_mean.shape[0]):
            zero_mean[i, 2] = Field.minimized_angle(zero_mean[i, 2])
        cov = np.dot(zero_mean.T, zero_mean) / self.num_particles
        cov += np.eye(3) * 1e-6  # Avoid bad conditioning

        return mean.reshape((-1, 1)), cov
