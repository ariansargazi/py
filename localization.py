import time
import argparse
import numpy as np

from env import Field
from pf import ParticleFilter

# TODO: use these plot utils to plot mean and covariance error
from plot import plot_mean_error, plot_squared_sum_diag_cov

class OpenLoopRectanglePolicy:
    def __init__(self, dt=0.1):
        self.dt = dt

    def __call__(self, x, t):
        n = round(t / self.dt)
        index = n % round(5 / self.dt)

        if index == 2 * int(1 / self.dt):
            u = np.array([np.deg2rad(45), 100 * self.dt, np.deg2rad(45)])
        elif index == 4 * int(1 / self.dt):
            u = np.array([np.deg2rad(45), 0, np.deg2rad(45)])
        else:
            u = np.array([0, 100 * self.dt, 0])
        return u.reshape((-1, 1))


def pf_localization(env, policy, filt, x0, num_steps, plot=False, step_pause=0., step_breakpoint=False):

    # Collect data from an entire rollout
    (states_noisefree, states_real, action_noisefree, obs_noisefree, obs_real) = env.rollout(x0,
                                                                                    policy, num_steps)
    states_filter = np.zeros(states_real.shape)
    states_filter[0, :] = x0.ravel()

    errors = np.zeros((num_steps, 3))
    position_errors = np.zeros(num_steps)
    mahalanobis_errors = np.zeros(num_steps)
    cov_mats = []
    
    for i in range(num_steps):
        x_real = states_real[i+1, :].reshape((-1, 1))
        u_noisefree = action_noisefree[i, :].reshape((-1, 1))
        z_real = obs_real[i, :].reshape((-1, 1))
        marker_id = env.get_marker_id(i)

        if filt is None:
            mean, cov = x_real, np.eye(3)
        else:
            # filters only know the action and observation
            mean, cov = filt.update(env, u_noisefree, z_real, marker_id)
        states_filter[i+1, :] = mean.ravel()
        cov_mats.append(cov)

        if plot:
            # move the robot
            env.move_robot(x_real)

            # plot observation
            env.plot_observation(x_real, z_real, marker_id)

            # plot actual trajectory
            x_real_previous = states_real[i, :].reshape((-1, 1))
            env.plot_path_step(x_real_previous, x_real, [0,0,1])

            # plot noisefree trajectory
            noisefree_previous = states_noisefree[i]
            noisefree_current = states_noisefree[i+1]
            env.plot_path_step(noisefree_previous, noisefree_current, [0,1,0])

            # plot estimated trajectory
            if filt is not None:
                filter_previous = states_filter[i]
                filter_current = states_filter[i+1]
                env.plot_path_step(filter_previous, filter_current, [1,0,0])

            # plot particles
            env.plot_particles(filt.particles, filt.weights)

        # pause/breakpoint
        if step_pause:
            time.sleep(step_pause)
        if step_breakpoint:
            breakpoint()

        errors[i, :] = (mean - x_real).ravel()
        errors[i, 2] = Field.minimized_angle(errors[i, 2])
        position_errors[i] = np.linalg.norm(errors[i, :2])
        

        cond_number = np.linalg.cond(cov)
        if cond_number > 1e12:
            print('Badly conditioned cov (setting to identity):', cond_number)
            print(cov)
            cov = np.eye(3)
        mahalanobis_errors[i] = \
            errors[i:i+1, :].dot(np.linalg.inv(cov)).dot(errors[i:i+1, :].T)
        

    mean_position_error = position_errors.mean()
    mean_mahalanobis_error = mahalanobis_errors.mean()

    if filt is not None:
        print('-' * 80)
        print('Mean position error:', mean_position_error)
        print('Mean Mahalanobis error:', mean_mahalanobis_error)

    plot_mean_error(position_errors)
    plot_mean_error(mahalanobis_errors)
    plot_squared_sum_diag_cov(cov_mats)
    
    if plot:
        while True:
            env.p.stepSimulation()

    return mean_position_error


def setup_parser():
    parser = argparse.ArgumentParser()

    # General args
    parser.add_argument('--gui', action='store_true', help='turn on plotting')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--num-steps', type=int, default=200, help='timesteps to simulate')

    # Noise scaling factors
    parser.add_argument(
        '--data-factor', type=float, default=1,
        help='scaling factor for motion and observation noise (data)')
    parser.add_argument(
        '--filter-factor', type=float, default=1,
        help='scaling factor for motion and observation noise (filter)')
    parser.add_argument(
        '--num-particles', type=int, default=100,
        help='number of particles')

    # Debugging arguments
    parser.add_argument(
        '--step-pause', type=float, default=0.,
        help='slows down the rollout to make it easier to visualize')
    parser.add_argument(
        '--step-breakpoint', action='store_true',
        help='adds a breakpoint to each step for debugging purposes')

    return parser


if __name__ == '__main__':
    args = setup_parser().parse_args()
    print('Data factor:', args.data_factor)
    print('Filter factor:', args.filter_factor)

    if args.seed is not None:
        np.random.seed(args.seed)

    alphas = np.array([0.05**2, 0.005**2, 0.1**2, 0.01**2])
    beta = np.diag([np.deg2rad(5)**2])

    env = Field(
        args.data_factor * alphas,
        args.data_factor * beta,
        gui=args.gui
    )

    policy = OpenLoopRectanglePolicy()

    initial_mean = np.array([180, 50, 0]).reshape((-1, 1))
    initial_cov = np.diag([10, 10, 1])

    filt = ParticleFilter(
        initial_mean,
        initial_cov,
        args.num_particles,
        args.filter_factor * alphas,
        args.filter_factor * beta
        )

    pf_localization(env, policy, filt, initial_mean, args.num_steps, args.gui, args.step_pause, args.step_breakpoint)


