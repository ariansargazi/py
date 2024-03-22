import numpy as np

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
        """Update particles after taking an action u."""
        # Move each particle according to the motion model (with noise)
        for i in range(self.num_particles):
            self.particles[i, :] = env.sample_noisy_action(u, self.alphas).ravel()
        return self.particles

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving
        a landmark observation.

        u: action
        z: landmark observation
        marker_id: landmark ID
        """
        # Move particles based on the action taken
        self.move_particles(env, u)

        # Update weights based on observation likelihood
        for i in range(self.num_particles):
            predicted_z = env.observe(self.particles[i, :], marker_id)
            self.weights[i] = env.likelihood(z - predicted_z, self.beta)
        
        # Normalize the weights
        self.weights += 1e-300  # Avoid division by zero
        self.weights /= np.sum(self.weights)

        # Resample particles based on updated weights
        self.particles = self.resample(self.particles, self.weights)
        
        # Calculate new mean and covariance
        mean, cov = self.mean_and_variance(self.particles)

        return mean, cov

    def resample(self, particles, weights):
        """Sample new particles and weights given current particles and weights."""
        # Cumulative sum of weights
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.0  # Ensure sum is exactly one
        indexes = np.searchsorted(cumulative_sum, np.random.uniform(0, 1, self.num_particles))

        # Resample according to indexes
        resampled_particles = particles[indexes]

        return resampled_particles

    def mean_and_variance(self, particles):
        """Compute the mean and covariance matrix for a set of equally-weighted
        particles.

        particles: (n x 3) matrix of poses
        """
        mean = np.average(particles, weights=self.weights, axis=0)
        # Wrap the angles of the particles to be between -pi and pi
        particles[:, 2] = [Field.minimized_angle(theta) for theta in particles[:, 2]]
        mean[2] = Field.minimized_angle(np.arctan2(
            np.sum(np.sin(particles[:, 2])),
            np.sum(np.cos(particles[:, 2]))
        ))
        centered_particles = particles - mean
        cov = np.cov(centered_particles.T, aweights=self.weights)

        return mean.reshape((-1, 1)), cov
