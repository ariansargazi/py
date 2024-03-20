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
        new_particles = np.array([env.forward(x, u) for x in self.particles])
        return new_particles

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving
        a landmark observation.

        u: action
        z: landmark observation
        marker_id: landmark ID
        """
        #self.particles = self.move_particles(env, u)
        u_noisy = env.sample_noisy_action(u, self.alphas)
        self.particles = self.move_particles(env, u_noisy)
        # YOUR CODE HERE
        for m in range(self.num_particles):
            x_t = self.particles[m, :].reshape((-1, 1))
            z_noisy = env.sample_noisy_observation(x_t, marker_id, self.beta)
            z_expected = env.observe(x_t, marker_id)
            inovation = z - z_expected
            self.weights[m] = env.likelihood(inovation, self.beta)
        
        self.weights += 1.e-300      
        self.weights /= sum(self.weights)  # normalize
        
        self.particles = self.resample(self.particles,self.weights)
        mean, cov = self.mean_and_variance(self.particles)

        return mean, cov

    def resample(self, particles, weights):
        """Sample new particles and weights given current particles and weights. Be sure
        to use the low-variance sampler from class.

        particles: (n x 3) matrix of poses
        weights: (n,) array of weights
        """
        # YOUR CODE HERE
        indices = np.random.choice(
            range(self.num_particles), size=self.num_particles, replace=True, p=weights)
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
