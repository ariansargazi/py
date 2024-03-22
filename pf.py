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
        # Move each particle according to the odometry motion model
        new_particles = np.copy(self.particles)
        for i in range(self.num_particles):
            new_particles[i, :] = env.sample_noisy_action(u, self.alphas).reshape(-1)
        return new_particles

    def update(self, env, u, z, marker_id):
        # Update particle weights based on observation likelihood
        self.particles = self.move_particles(env, u)
        for i in range(self.num_particles):
            predicted_observation = env.observe(self.particles[i, :].reshape(-1, 1), marker_id)
            self.weights[i] = env.likelihood(z - predicted_observation, self.beta)

        # Normalize the weights
        self.weights += 1e-300  # Avoid division by zero
        self.weights /= sum(self.weights)

        # Resample particles based on weights
        self.particles = self.resample(self.particles, self.weights)

        # Calculate mean and covariance
        mean, cov = self.mean_and_variance(self.particles)
        return mean, cov

    def resample(self, particles, weights):
        # Resample particles using systematic resampling
        indices = []
        C = [0.] + [np.sum(weights[:i+1]) for i in range(len(weights))]
        u0, j = np.random.random(), 0
        for u in [(u0 + i)/len(weights) for i in range(len(weights))]:
            while u > C[j]:
                j += 1
            indices.append(j-1)
        resampled_particles = particles[indices]
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
