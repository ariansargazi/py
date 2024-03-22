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
        self.particles = np.random.multivariate_normal(
            self._init_mean.ravel(), self._init_cov, self.num_particles)
        self.weights = np.ones(self.num_particles) / self.num_particles

    def move_particles(self, env, u):
        """Move particles according to the action `u` and add motion noise."""
        for i in range(self.num_particles):
            # Apply the action to move the particle
            self.particles[i, :] = env.forward(self.particles[i, :], u).ravel()
            # Add noise
            motion_noise = env.noise_from_motion(u, self.alphas)
            self.particles[i, :] += np.random.multivariate_normal(np.zeros(3), motion_noise)
        return self.particles

    def update(self, env, u, z, marker_id):
        """Update the state estimate after an action `u` and observation `z`."""
        self.move_particles(env, u)

        for i in range(self.num_particles):
            expected_z = env.observe(self.particles[i, :], marker_id)
            innovation = z - expected_z
            self.weights[i] *= env.likelihood(innovation, self.beta) + 1e-12

        self.weights += 1.e-300      # Avoid round-off to zero
        self.weights /= np.sum(self.weights)  # Normalize

        self.particles = self.resample(self.particles, self.weights)
        mean, cov = self.mean_and_variance(self.particles)

        return mean, cov

    def resample(self, particles, weights):
        """Resample particles to focus on high-probability regions."""
        indices = []
        C = [0.] + [np.sum(weights[:i+1]) for i in range(len(weights))]
        u0, j = np.random.random(), 0
        for u in [(u0+i)/len(weights) for i in range(len(weights))]:
            while u > C[j]:
                j += 1
            indices.append(j-1)
        return particles[indices]

    def mean_and_variance(self, particles):
        """Compute the mean and covariance matrix for a set of equally-weighted particles."""
        mean = np.mean(particles, axis=0)
        mean[2] = np.arctan2(
            np.sin(particles[:, 2]).sum(),
            np.cos(particles[:, 2]).sum()
        )
        zero_mean_particles = particles - mean
        for i in range(len(zero_mean_particles)):
            zero_mean_particles[i, 2] = Field.minimized_angle(zero_mean_particles[i, 2])
        cov = np.dot(zero_mean_particles.T, zero_mean_particles) / len(particles)
        cov += np.eye(3) * 1e-12  # Add a small value to diagonal to ensure non-singularity
        return mean.reshape((-1, 1)), cov
