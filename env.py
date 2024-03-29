import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt


class Field:
    NUM_MARKERS = 6

    INNER_OFFSET_X = 32
    INNER_OFFSET_Y = 13

    INNER_SIZE_X = 420
    INNER_SIZE_Y = 270

    COMPLETE_SIZE_X = INNER_SIZE_X + 2 * INNER_OFFSET_X
    COMPLETE_SIZE_Y = INNER_SIZE_Y + 2 * INNER_OFFSET_Y

    MARKER_OFFSET_X = 21
    MARKER_OFFSET_Y = 0

    MARKER_DIST_X = 442
    MARKER_DIST_Y = 292

    MARKERS = (1, 2, 3, 4, 5, 6)

    MARKER_X_POS = {
        1: MARKER_OFFSET_X,
        2: MARKER_OFFSET_X + 0.5 * MARKER_DIST_X,
        3: MARKER_OFFSET_X + MARKER_DIST_X,
        4: MARKER_OFFSET_X + MARKER_DIST_X,
        5: MARKER_OFFSET_X + 0.5 * MARKER_DIST_X,
        6: MARKER_OFFSET_X,
    }

    MARKER_Y_POS = {
        1: MARKER_OFFSET_Y,
        2: MARKER_OFFSET_Y,
        3: MARKER_OFFSET_Y,
        4: MARKER_OFFSET_Y + MARKER_DIST_Y,
        5: MARKER_OFFSET_Y + MARKER_DIST_Y,
        6: MARKER_OFFSET_Y + MARKER_DIST_Y,
    }


    def __init__(
        self,
        alphas,
        beta,
        gui=True
    ):
        self.alphas = alphas
        self.beta = beta

        # initialize pybullet environment
        if gui:
            physicsClient = p.connect(p.GUI)
        else:
            physicsClient = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-10)
        self.p = p

        # add the robot and landmarks to the pybullet scene
        self.create_scene()
        self.add_robot()

    def forward(self, x, u):
        """Compute next state, given current state and action.

        Implements the odometry motion model.

        x: [x, y, theta]
        u: [rot1, trans, rot2]
        """
        prev_x, prev_y, prev_theta = x
        rot1, trans, rot2 = u

        x_next = np.zeros(x.size)
        theta = prev_theta + rot1
        x_next[0] = prev_x + trans * np.cos(theta)
        x_next[1] = prev_y + trans * np.sin(theta)
        x_next[2] = Field.minimized_angle(theta + rot2)

        return x_next.reshape((-1, 1))

    def get_marker_id(self, step):
        """Compute the landmark ID at a given timestep."""
        return ((step // 2) % self.NUM_MARKERS) + 1

    def observe(self, x, marker_id):
        """Compute observation, given current state and landmark ID.

        x: [x, y, theta]
        marker_id: int
        """
        dx = self.MARKER_X_POS[marker_id] - x[0]
        dy = self.MARKER_Y_POS[marker_id] - x[1]
        return np.array(
            [Field.minimized_angle(np.arctan2(dy, dx) - x[2])]
        ).reshape((-1, 1))

    def noise_from_motion(self, u, alphas):
        """Compute covariance matrix for noisy action.

        u: [rot1, trans, rot2]
        alphas: noise parameters for odometry motion model
        """
        variances = np.zeros(3)
        variances[0] = alphas[0] * u[0]**2 + alphas[1] * u[1]**2
        variances[1] = alphas[2] * u[1]**2 + alphas[3] * (u[0]**2 + u[2]**2)
        variances[2] = alphas[0] * u[2]**2 + alphas[1] * u[1]**2
        return np.diag(variances)

    def likelihood(self, innovation, beta):
        """Compute the likelihood of innovation, given covariance matrix beta.

        innovation: x - mean, column vector
        beta: noise parameters for landmark observation model
        """
        norm = np.sqrt(np.linalg.det(2 * np.pi * beta))
        inv_beta = np.linalg.inv(beta)

        return np.exp(-0.5 * innovation.T.dot(inv_beta).dot(innovation)) / norm

    def sample_noisy_action(self, u, alphas=None):
        """Sample a noisy action, given a desired action and noise parameters.

        u: desired action
        alphas: noise parameters for odometry motion model (default: data alphas)
        """
        if alphas is None:
            alphas = self.alphas

        cov = self.noise_from_motion(u, alphas)
        return np.random.multivariate_normal(u.ravel(), cov).reshape((-1, 1))

    def sample_noisy_observation(self, x, marker_id, beta=None):
        """Sample a noisy observation given a current state, landmark ID, and noise
        parameters.

        NOTE: this is just a naive observation model, improve it for additional points.

        x: current state
        marker_id: int
        beta: noise parameters for landmark observation model (default: data beta)
        """

        if beta is None:
            beta = self.beta

        z = self.observe(x, marker_id)
        z = np.random.multivariate_normal(
            z.ravel(), beta).reshape((-1, 1))

        return z

    def get_figure(self):
        return plt.figure(1)

    def rollout(self, x0, policy, num_steps, dt=0.1):
        """Collect data from an entire rollout."""
        states_noisefree = np.zeros((num_steps, 3))
        states_real = np.zeros((num_steps, 3))
        action_noisefree = np.zeros((num_steps, 3))
        obs_noisefree = np.zeros((num_steps, 1))
        obs_real = np.zeros((num_steps, 1))

        x_noisefree = x_real = x0
        for i in range(num_steps):
            t = i * dt

            u_noisefree = policy(x_real, t)
            x_noisefree = self.forward(x_noisefree, u_noisefree)

            u_real = self.sample_noisy_action(u_noisefree)
            x_real = self.forward(x_real, u_real)

            marker_id = self.get_marker_id(i)
            z_noisefree = self.observe(x_real, marker_id)
            z_real = self.sample_noisy_observation(x_real, marker_id)

            states_noisefree[i, :] = x_noisefree.ravel()
            states_real[i, :] = x_real.ravel()
            action_noisefree[i, :] = u_noisefree.ravel()
            obs_noisefree[i, :] = z_noisefree.ravel()
            obs_real[i, :] = z_real.ravel()

        states_noisefree = np.concatenate([x0.T, states_noisefree], axis=0)
        states_real = np.concatenate([x0.T, states_real], axis=0)

        return (
            states_noisefree, states_real,
            action_noisefree,
            obs_noisefree, obs_real
        )

    def create_scene(self):
        self.plane_id = self.p.loadURDF("plane.urdf")
        h = 1
        r = 0.1

        pillar_shape = self.p.createCollisionShape(
            self.p.GEOM_CYLINDER, radius=r, height=h)
        colors = [
            [0.9, 0.0, 0.0, 1.0],
            [0.0, 0.9, 0.0, 1.0],
            [0.0, 0.0, 0.9, 1.0],
            [0.5, 0.5, 0.0 ,1.0],
            [0.0, 0.5, 0.5, 1.0],
            [0.5, 0.0, 0.5, 1.0],
        ]
        self.pillar_ids = []
        self.text_ids = []
        for m in self.MARKERS:
            x, y = self.MARKER_X_POS[m]/100, self.MARKER_Y_POS[m]/100
            pillar_id = self.p.createMultiBody(
                baseCollisionShapeIndex=pillar_shape, basePosition=[x, y, h/2])
            self.pillar_ids.append(pillar_id)
            self.p.setCollisionFilterGroupMask(pillar_id, -1, 0, 0)
            self.p.changeVisualShape(pillar_id, -1, rgbaColor=colors[m-1])

            text_id = self.p.addUserDebugText(
                str(m),
                textPosition=[0,0,h/2+0.1],
                textColorRGB=[0, 0, 0],
                textSize=2,
                parentObjectUniqueId=pillar_id,
            )
            self.text_ids.append(text_id)

    def add_robot(self):
        self.racer_car_id = self.p.loadURDF('racecar.urdf', [0,0,0], [0,0,0,1])

    def move_robot(self, x):
        p = [x[0]/100., x[1]/100., 0]
        q = self.p.getQuaternionFromEuler([0,0,x[2]+np.pi])
        self.p.resetBasePositionAndOrientation(self.racer_car_id, p, q)

    def get_robot_x(self):
        p, q = self.p.getBasePositionAndOrientation(
            self.racer_car_id)
        theta = self.p.getEulerFromQuaternion(q)[2] + np.pi
        return [p[0]*100, p[1]*100, theta]

    def plot_observation(self, x, z, marker_id):
        xyz0 = np.array([x[0,0]/100., x[1,0]/100., 0.05])

        marker_x = np.array([
            self.MARKER_X_POS[marker_id]/100.,
            self.MARKER_Y_POS[marker_id]/100.,
            0.05
        ])
        distance = np.linalg.norm(xyz0 - marker_x)

        dx = np.cos(z+x[2])
        dy = np.sin(z+x[2])
        xyz1 = [x[0]/100. + dx*distance, x[1]/100. + dy*distance, 0.05]

        kwargs = {}
        if hasattr(self, 'obs_id'):
            kwargs['replaceItemUniqueId'] = self.obs_id
        self.obs_id = self.p.addUserDebugLine(
                xyz0, xyz1, [0,0,0], 2, **kwargs)

    def plot_path_step(self, x_previous, x_current, color):
        xyz_previous = [x_previous[0]/100., x_previous[1]/100., 0.05]
        xyz_current = [x_current[0]/100., x_current[1]/100., 0.05]
        self.p.addUserDebugLine(xyz_previous, xyz_current, color, 2)

    def plot_particles(self, particles, weights):
        xyz = np.concatenate(
            (particles[:,:2]/100., np.full((len(particles),1), 0.2)), axis=1)
        color = np.zeros((len(particles),3))
        color[:,0] = 1
        color = color * weights.reshape(-1,1) * 50
        color = np.clip(color, 0, 1)
        kwargs = {}
        if hasattr(self, 'particle_id'):
            kwargs['replaceItemUniqueId'] = self.particle_id
        self.particle_id = self.p.addUserDebugPoints(
            xyz, color, pointSize=2, **kwargs)

    @staticmethod
    def minimized_angle(angle):
        """ Normalize an angle to [-pi, pi].
        """

        while angle < -np.pi:
            angle += 2 * np.pi
        while angle >= np.pi:
            angle -= 2 * np.pi
        return angle