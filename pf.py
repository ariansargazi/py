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
        new_particles = np.array([env.forward(p, u.ravel()) for p in self.particles])
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
            z_hat = env.observe(self.particles[m].reshape(-1,1), marker_id)
            self.weights[m] *= Field.likelihood(z - z_hat, self.beta)
        
        self.weights += 1.e-300  # avoid round-off to zero
        self.weights /= sum(self.weights) # normalize
        
        self.particles = self.resample(self.particles, self.weights)
        mean, cov = self.mean_and_variance(self.particles)

        return mean, cov

    def resample(self, particles, weights):
        """Sample new particles and weights given current particles and weights. Be sure
        to use the low-variance sampler from class.

        particles: (n x 3) matrix of poses
        weights: (n,) array of weights
        """
        M = self.num_particles
        resampled_particles = np.zeros(particles.shape)
        index = int(np.random.rand()*M)
        beta = 0.0
        mw = max(weights)
        for i in range(M):
            beta += np.random.rand()*2.0*mw
            while beta > weights[index]:
                beta -= weights[index]
                index = (index + 1) % M
            resampled_particles[i] = particles[index]
        
        return resampled_particles

    def mean_and_variance(self, particles):
        """Compute the mean and covariance matrix for a set of equally-weighted
        particles.

        particles: (n x 3) matrix of poses
        """
        mean = np.average(particles, weights=self.weights, axis=0)
        centered_particles = particles - mean
        cov = np.cov(centered_particles.T, aweights=self.weights)

        return mean.reshape((-1, 1)), cov + np.eye(3) * 1e-6  # Avoid bad conditioning
