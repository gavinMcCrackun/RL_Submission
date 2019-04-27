import math
import os
import typing as tp
from dataclasses import dataclass
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from project.mmove import Dataset, MovementFilter
from project.neural.discriminator import Discriminator
import seaborn as sns
sns.set()


def square_action_array(max_delta: int, randomize: bool = True, exclude_zero: bool = True, circular: bool = False):
    # integer deltas from [-max_delta, +max_delta]
    deltas = np.arange(-max_delta, max_delta + 1, dtype=int)
    actions = [(dx, dy) for dx in deltas for dy in deltas]
    if exclude_zero:
        actions = [(dx, dy) for (dx, dy) in actions if dx != 0 and dy != 0]

    if circular:
        actions = [a for a in actions if math.hypot(a[0], a[1]) <= max_delta]

    actions = np.array(actions)
    if randomize:
        np.random.shuffle(actions)

    return actions


@dataclass
class LineState:
    step: int = 0
    x: int = 0
    y: int = 0
    target_x: int = 0
    target_y: int = 0

    def to_output(self):
        return np.array([
            self.step,
            self.target_x - self.x,
            self.target_y - self.y,
        ])

    def translate(self, dx: int, dy: int):
        self.x += dx
        self.y += dy


class LineEnv(gym.Env):
    def __init__(
            self,
            # expert datafile
            expert_data: tp.Optional[str] = None,
            # number of points to interpolate
            num_interp_points: int = 40,
            # (max_x, max_y)
            bounds: (int, int) = (250, 250),
            # minimum distance away that the target can be generated
            min_target_dist: float = 100.0,
            # maximum distance away that the target can be generated
            max_target_dist: float = 200.0,
            # max distance to move in each dimension (e.g. dx and dy from [-9, 9])
            max_delta: int = 5,
            target_dist_tolerance: tp.Optional[float] = None,
            # prng seed
            seed: tp.Optional[int] = None,
            # max steps allowed before forced termination
            max_steps: int = 1000,
            # place to render episode plots, defaults to current directory
            render_dir: tp.Optional[str] = None,
    ):
        self.num_interp_points = num_interp_points
        if render_dir is None:
            render_dir = os.getcwd()
        self.render_dir = render_dir

        # build discriminator if we have data for it
        self.discriminator: tp.Optional[Discriminator]
        if expert_data is not None:
            assert max_steps > 5
            assert max_target_dist > min_target_dist

            movement_filter = MovementFilter(
                num_points_bounds=(5, max_steps),
                distance_bounds=(min_target_dist, max_target_dist),
                time_duration_bounds=(0, 1.5)
            )

            self.expert_data = Dataset(
                filename=expert_data,
                interpolation_pts=num_interp_points,
                movement_filter=movement_filter,
            )
            self.discriminator = Discriminator(
                expert_data=self.expert_data.interpolated
            )

            self.discriminator.pretrain_discriminator()
        else:
            self.discriminator = None

        # by default, make sure the maximum step we can make such that we won't go over the target
        if target_dist_tolerance is None:
            self.target_dist_tolerance = 4 * math.hypot(max_delta, max_delta)
        else:
            self.target_dist_tolerance = target_dist_tolerance

        # number of steps before we automatically move the state to the target
        self.max_steps = max_steps
        if max_steps <= 0:
            raise ValueError("bad max_steps")

        # build action list from the maximum distance we can move
        self.max_delta = max_delta
        self.action_array = square_action_array(max_delta, circular=True)
        self.action_space = spaces.Discrete(len(self.action_array))

        # boundaries
        if not (0 <= min_target_dist < max_target_dist < min(bounds[0], bounds[1])):
            raise ValueError("targets can be generated outside of bounds (bad min/max_target_dist")

        self.bounds = bounds
        # state looks like (x, y, delta_x to target, delta_y to target)
        self.state: tp.Optional[(int, int, int, int)] = None
        # target is just a (x, y) point
        self.target: tp.Optional[(int, int)] = None
        self.target_bounds = (
            min_target_dist,
            max_target_dist
        )

        state_bounds = np.array([bounds[0], bounds[1], max_target_dist, max_target_dist])
        # build observation space
        self.observation_space = spaces.Box(-state_bounds, state_bounds, dtype=np.float32)

        # initialize stats/misc:
        # current episode
        self.episode = 0
        self.state_hist = []
        self.all_state_hist = []
        # current step within an episode
        self.steps = 0
        # total distance within an episode
        self.total_episode_distance = 0
        # total steps overall
        self.total_steps = 0
        # environment
        self.np_random, _ = seeding.np_random(seed)
        self.is_done = False
        self.last_reward = 0

        # weight of the discriminator
        self.discriminator_weight = 100.0

        self.actor = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_expert_data(self):
        return self.expert_data.interpolated

    def __reset_target(self):
        # generate a random target position between the minimum and maximum distances
        rmin, rmax = self.target_bounds
        # just pick a random r/theta and use trig
        r = self.np_random.uniform(rmin, rmax)
        theta = self.np_random.uniform(0.0, 2 * np.pi)
        self.target = (
            int(r * math.cos(theta)),
            int(r * math.sin(theta))
        )

    def __reset_state(self):
        self.__reset_target()
        self.state = (0, 0, self.target[0], self.target[1])
        self.all_state_hist.append(self.state_hist)
        self.state_hist = [self.state]

    def __last_trajectory(self):
        return np.array([
            [s[0], s[1]] for s in self.state_hist
        ])

    def __interpolate_trajectory(self, trajectory):
        interp_pts = np.linspace(0.0, 1.0, self.num_interp_points)
        time_pts = np.linspace(0.0, 1.0, len(trajectory))
        xy_pts = np.array([[s[0], s[1]] for s in trajectory])
        return np.column_stack((
            np.interp(interp_pts, time_pts, xy_pts[:, 0]),
            np.interp(interp_pts, time_pts, xy_pts[:, 1]),
        ))

    def __calc_actor_heatmap(self, state):
        # x, y, tx, ty = ob
        env_bounds = np.array(self.bounds).astype(int)

        _, _, tx, ty = state
        x_coords = np.arange(-env_bounds[0], env_bounds[0] + 1)
        y_coords = np.arange(-env_bounds[1], env_bounds[1] + 1)
        n_total = len(x_coords) * len(y_coords)
        states = np.column_stack((
            np.tile(x_coords, len(y_coords)),
            np.repeat(y_coords, len(x_coords)),
            [tx] * n_total,
            [ty] * n_total,
        ))

        # print("heatmap")
        probabilities = self.actor.sess.run(self.actor.actor_proba, {self.actor.s: states})

        target_dirs = (np.array([tx, ty]) - states[:, :2]).astype(float)
        norms = np.linalg.norm(target_dirs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # make sure we don't divide by zero and produce nans
        target_dirs /= norms

        # calculate unit vectors for the action array first before we select them
        action_dirs = np.asarray(self.action_array, dtype=float)
        norms = np.linalg.norm(action_dirs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        action_dirs /= norms

        max_prob_ids = np.argmax(probabilities, axis=1)
        action_dirs = action_dirs[max_prob_ids]

        # ...finally calculate dot products
        action_dots = np.einsum('ij,ij->i', action_dirs, target_dirs)
        # print("heatmap done")
        return action_dots.reshape(env_bounds * 2 + 1)

    def __render_episode(self):
        assert self.actor is not None
        if len(self.state_hist) == 0 or self.actor is None:
            return

        ax1: Axes
        # ax2: Axes
        fig: Figure
        fig, ax1 = plt.subplots(nrows=1, ncols=1)
        ax1.set_aspect(aspect='equal')
        # ax2.set_aspect(aspect='equal')
        fig.set_size_inches(w=8, h=8)

        ax1.set_title("Episode {} Reward {:.1f}".format(self.episode, self.last_reward))

        # render actor stuff
        direction_heatmap = self.__calc_actor_heatmap(self.state_hist[-1])
        hshape = direction_heatmap.shape

        ax = sns.heatmap(
            # note: we flip upside down cause plotting
            direction_heatmap,
            cmap='viridis',
            # center=0,
            ax=ax1
        )

        # ax1.set_xlabel("x [pixel]")
        # ax1.set_ylabel("y [pixel]")

        trajectory = self.__last_trajectory()
        def to_hmap(x, y):
            return hshape[0] * (x + self.bounds[0]) / (2 * self.bounds[0]), \
                   hshape[1] * (y + self.bounds[1]) / (2 * self.bounds[1])

        x_pts, y_pts = to_hmap(trajectory[:, 0], trajectory[:, 1])

        target_pt = to_hmap(self.state[2], self.state[3])
        target_circ = plt.Circle(target_pt, self.target_dist_tolerance, color='w', fill=False, linewidth=2)

        ax1.plot(x_pts, y_pts, label="all points", color='white', linewidth=2)
        ax1.add_artist(target_circ)

        # interpolated = self.__interpolate_trajectory(trajectory)
        # ax1.plot(interpolated[:, 0], interpolated[:, 1], label="interpolated")
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax1.get_yticklabels(), visible=False)

        fig.savefig(os.path.join(self.render_dir, 'episode_{}.png').format(self.episode))
        plt.close(fig)

    def set_actor(self, actor):
        self.actor = actor
        print(self.actor)

    def get_path(self):
        return os.path.join(self.render_dir, 'episode_heat{}.png'.format(self.episode))

    def reset(self):
        # make sure to call this before resetting the state
        if self.episode % 10 == 0 and len(self.state_hist) > 0:
            # if self.episode % 10 == 0:
            self.__render_episode()

            if self.discriminator is not None:
                # train on last 10 runs
                loss = self.discriminator.train([
                    self.__interpolate_trajectory(t)
                    for t in self.all_state_hist[-9:]
                ])
                print("training loss: {}".format(loss))

        self.__reset_state()
        self.steps = 0
        self.total_episode_distance = 0
        self.episode += 1
        self.is_done = False
        self.last_reward = 0
        return np.array(self.state)

    def update_state(self, action_index: int):
        if not self.action_space.contains(action_index):
            raise ValueError("invalid action {} ({})".format(action_index, type(action_index)))

        target_x, target_y = self.target
        if self.steps < self.max_steps:
            dx, dy = self.action_array[action_index]
        else:
            # automatically move to target if we hit max # of steps
            dx = target_x - self.state[0]
            dy = target_y - self.state[1]

        # update and clamp state into bounds
        max_x, max_y = self.bounds
        # new_x = self.state[0] + dx
        # new_y = self.state[1] + dy
        new_x = max(min(self.state[0] + dx, max_x), -max_x)
        new_y = max(min(self.state[1] + dy, max_y), -max_y)

        self.state = (new_x, new_y, target_x, target_y)

        # record new state and update distance tally
        self.state_hist.append(self.state)
        # note: we don't use the clamped dx/dy since it will learn to run into walls
        distance = math.hypot(dx, dy)
        self.total_episode_distance += distance
        return distance, self.state

    def target_dist(self):
        return math.hypot(
            self.state[0] - self.target[0],
            self.state[1] - self.target[1],
        )

    def __calc_best_dist(self):
        x, y, _, _ = self.state
        tx, ty = self.target

        dist = None
        for dx, dy in self.action_array:
            new_dist = math.hypot(tx - (x + dx), ty - (y + dy))
            if dist is None or new_dist < dist:
                dist = new_dist

        return dist

    # Accepts an action and returns a tuple (observation, reward, done, info).
    def step(self, action: int) -> (np.array, float, bool, tp.Any):
        if self.is_done:
            raise AssertionError("environment already finished, please call reset() instead of step")
        if not self.state:
            raise ValueError("environment not initialized, please call reset()")
        info = False

        delta, _ = self.update_state(action)
        new_dist = self.target_dist()

        self.is_done = new_dist <= self.target_dist_tolerance

        reward = 0

        if self.discriminator is not None and self.is_done:
            predict = self.discriminator.predict(self.__interpolate_trajectory(self.state_hist))
            if self.episode % 10 == 0:
                # self.discriminator_weight += 0.02
                print("predict: {}".format(predict))

            reward += predict * self.discriminator_weight
            # loss = self.discriminator.train([
            #     self.__interpolate_trajectory(self.state_hist)
            # ])

        self.steps += 1
        self.total_steps += 1
        self.last_reward = reward
        return np.array(self.state), reward, self.is_done, info

    def render(self, mode='human'):
        pass


register(
    id='mdp-line-v0',
    entry_point='project.mdp.line_env:LineEnv',
    max_episode_steps=None,
    nondeterministic=True,
)

# class MdpEnv(gym.Env):
#
#     def __init__(self):
#         self.target_dist_tolerance = 17.0
#         self.steps_so_far = 0
#         # self._actions = np.array(itertools.product(
#         #     np.linspace(0, 2 * np.pi, 180),
#         #     np.linspace(0, 5, 60)
#         # ))
#         # self._actions = np.transpose([np.tile(np.linspace(0, 2 * np.pi, 45), len(np.linspace(0, 5, 30))),
#         #                 np.repeat(np.linspace(0, 5, 30), len(np.linspace(0, 2 * np.pi, 45)))])
#         self._actions = np.array(
#             [(dx, dy) for dx in range(-9, 10) for dy in range(-9, 10)]
#         ).astype(int)
#
#         np.random.shuffle(self._actions)
#         self._bounds = np.array([
#             250, 250,  # max position
#             250, 250,  # max target position
#             # 1000, 1000 # max dx dy
#         ])  # / 2
#
#         self.action_space = spaces.Discrete(len(self._actions))
#         self.action_space_len = len(self._actions)
#         self.observation_space = spaces.Box(
#             -self._bounds, self._bounds, dtype=np.float32)
#
#         self.seed()
#         self.state = None
#         self.steps_beyond_done = None
#
#         self.episode = 0
#         self.log_this = []
#         self.hit_targ = 0
#
#         # Getting human movements
#         self.human_data = return_timeless_movements()
#         print(self.human_data.shape)
#         self.train_x = self.human_data
#         self.train_y = np.ones(len(self.human_data))
#
#         # Discriminator Neural Network
#         self.discriminator = Sequential()
#
#         self.discriminator.add(Flatten(input_shape=self.human_data[0].shape))
#         self.discriminator.add(Dense(32))
#         self.discriminator.add(LeakyReLU(alpha=0.4))
#         self.discriminator.add(Dense(32))
#         self.discriminator.add(LeakyReLU(alpha=0.2))
#         self.discriminator.add(Dense(1, activation='sigmoid'))
#         self.discriminator.summary()
#
#         self.discriminator.compile(optimizer=tf.train.AdamOptimizer(),
#                                    loss='sparse_categorical_crossentropy',
#                                    metrics=['accuracy'])
#
#         # self trajectory
#         self.trajectory = []
#         self.distance_travelled = 0
#         # keras.Sequential([
#         #     keras.layers.Flatten(input_shape=(120,)),
#         #     keras.layers.Dense(32, activation=tf.nn.relu),
#         #     model.add(Dense(1, activation='sigmoid'))
#         # ])
#
#     # def parse_data(self):
#
#     def seed(self, seed=None):
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]
#
#     def reset(self):
#         self.steps_beyond_done = None
#         self.log_this = []
#         self.distance_travelled = 0
#         self.steps_so_far = 0
#         x = int(self.np_random.uniform(-1, 1) * 200)
#         y = int(self.np_random.uniform(-1, 1) * 200)
#         space_dif = 49
#         if x < 0:
#             x = x - space_dif
#         else:
#             x = x + space_dif
#         if y < 0:
#             y = y - space_dif
#         else:
#             y = y + space_dif
#
#         self.state = (0, 0, x, y)
#         return np.array(self.state)
#
#     def set_episodes(self, episodes):
#         self.episode = episodes
#
#     def step(self, action):
#
#         assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
#         self.steps_so_far = self.steps_so_far + 1
#         state = self.state
#         x, y, target_x, target_y = state
#
#         self.log_this.append([x, y])
#
#         old_dist_to_target = math.hypot(x - target_x, y - target_y)
#
#         done1 = False
#         # get theta/r to determine the new coordinate
#         # theta, r = self._actions[action]
#         # x += r * math.cos(theta)
#         # y += r * math.sin(theta)
#
#         dx, dy = self._actions[action]
#         x += dx
#         y += dy
#         # self.distance_travelled = self.distance_travelled + math.hypot(dx, dy) + 0.01
#         max_x = 250
#         max_y = 250
#         if x >= max_x:
#             x = max_x - 1
#             done1 = True
#         if x <= -max_x:
#             x = -max_x + 1
#             done1 = True
#         if y >= max_y:
#             y = max_y - 1
#             done1 = True
#         if y <= -max_y:
#             y = -max_y + 1
#             done1 = True
#
#         new_dist_to_target = math.hypot(x - target_x, y - target_y)
#
#         # print("x: " + str(x) + " , y: " + str(y))
#
#         # update state
#         self.state = (x, y, target_x, target_y)
#
#         # check if done
#
#         done = bool(new_dist_to_target <= self.target_dist_tolerance)
#
#         # if done1:
#         # reward = - 20000
#         # reward = - 100
#         # print("seems often")
#         # done = False
#         # x = 0
#         # y = 0
#         # self.state = (x, y, target_x, target_y, x_to_goal + y_to_goal)
#         # self.steps_beyond_done = -1
#         # print("off-screen")
#         # done = True
#         # return np.array(self.state), reward, done, {}
#
#         # not done yet, give simple reward based on distance moved
#         # towards the target
#         if not done:
#             # reward = 0.0
#             # if(self.steps_so_far == 95000):
#             #     reward = self.distance_travelled
#
#             if old_dist_to_target - new_dist_to_target >= 1.5:
#                 reward = old_dist_to_target - new_dist_to_target - 13
#             else:
#                 reward = old_dist_to_target - new_dist_to_target - 30
#         elif self.steps_beyond_done is None:
#             # just got to target
#             self.steps_beyond_done = 0
#             if (self.hit_targ % 10 == 0):
#                 print("at target")
#             reward = 1000.0
#             # reward = -self.distance_travelled
#             self.log_this.append([x, y])
#             self.hit_targ = self.hit_targ + 1
#             if (self.hit_targ % 100 == 0):
#                 print(self.log_this)
#                 print(target_x)
#                 print(target_y)
#                 # temp_x = []
#                 # temp_y = []
#                 # if self.hit_targ%100 == 0:
#                 #     for i in range(0, len(self.log_this)):
#                 #         temp_x.append(self.log_this[i][0])
#                 #         temp_y.append(self.log_this[i][1])
#                 #     plt.plot(temp_x, temp_y)
#                 #     plt.ylabel('y')
#                 #     plt.title("Mouse Movement " + str(self.hit_targ) + " Episodes")
#                 #     plt.xlabel('x')
#                 # plt.show()
#                 print("WOOO")
#                 print("WOOO")
#                 print("WOOO")
#             return np.array(self.state), reward, True, self.log_this
#         else:
#             if self.steps_beyond_done >= 0:
#                 logger.warn(
#                     "You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
#             self.steps_beyond_done += 1
#             reward = -300.0
#
#         return np.array(self.state), reward, done, self.log_this
#
#     # def pretrain_discriminator(self):
#     #     self.discriminator
#
#     def step_NN(self, action):
#         assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
#
#         # if last "batch_size" trajectory rewards are all > 0.5"
#         # train the discriminator
#         # if last_30_rewards >= 30:
#         #     last_30_rewards = 25
#         # Train the discriminator
#         # Draw random samples to make a batch from train x and train y
#
#         self.steps_so_far = self.steps_so_far + 1
#         state = self.state
#         x, y, target_x, target_y = state
#         self.log_this.append([x, y])
#
#         old_dist_to_target = math.hypot(x - target_x, y - target_y)
#
#         done1 = False
#         # get theta/r to determine the new coordinate
#         # theta, r = self._actions[action]
#         # x += r * math.cos(theta)
#         # y += r * math.sin(theta)
#
#         dx, dy = self._actions[action]
#         x += dx
#         y += dy
#
#         max_x = 250
#         max_y = 250
#         if x >= max_x:
#             x = max_x - 1
#             done1 = True
#         if x <= -max_x:
#             x = -max_x + 1
#             done1 = True
#         if y >= max_y:
#             y = max_y - 1
#             done1 = True
#         if y <= -max_y:
#             y = -max_y + 1
#             done1 = True
#
#         new_dist_to_target = math.hypot(x - target_x, y - target_y)
#
#         # print("x: " + str(x) + " , y: " + str(y))
#
#         # update state
#         self.state = (x, y, target_x, target_y)
#
#         # check if done
#
#         done = bool(new_dist_to_target <= self.target_dist_tolerance)
#
#         # if done1:
#         # reward = - 20000
#         # reward = - 100
#         # print("seems often")
#         # done = False
#         # x = 0
#         # y = 0
#         # self.state = (x, y, target_x, target_y, x_to_goal + y_to_goal)
#         # self.steps_beyond_done = -1
#         # print("off-screen")
#         # done = True
#         # return np.array(self.state), reward, done, {}
#
#         # not done yet, give simple reward based on distance moved
#         # towards the target
#         if not done:
#             reward = 0
#         elif self.steps_beyond_done is None:
#             # just got to target
#             print("training")
#             self.steps_beyond_done = 0
#             if (self.hit_targ % 10 == 0):
#                 print("at target")
#             # interpolate to 40 points and ask the discriminator it's confidence
#             self.log_this.append([x, y])
#             interped = lerp_n_xy(40, self.log_this)
#             interped = np.array([interped])
#             predictions = self.discriminator.predict(interped, 1)
#             print(predictions)
#             reward = predictions[0][0] * 900
#             print("reward")
#             print(str(reward))
#
#             idx = np.random.randint(0, self.train_x.shape[0], 1)
#             movement_train_x = self.train_x[idx]
#             # movement_train_y = self.train_y[idx]
#             movement_train_y = 0.99999
#             self.discriminator.train_on_batch(movement_train_x, np.array([movement_train_y]), sample_weight=None,
#                                               class_weight=None)
#             self.discriminator.train_on_batch(interped, np.array([0]), sample_weight=None, class_weight=None)
#
#             print(self.log_this)
#             self.hit_targ = self.hit_targ + 1
#             if (self.hit_targ % 100 == 0):
#                 print(target_x)
#                 print(target_y)
#                 temp_x = []
#                 temp_y = []
#                 # if self.episode > 1000:
#                 #     if self.episode%10 == 0:
#                 #         for i in range(0, len(self.log_this)):
#                 #             temp_x.append(self.log_this[i][0])
#                 #             temp_y.append(self.log_this[i][1])
#                 #         plt.plot(temp_x, temp_y)
#                 #         plt.ylabel('y')
#                 #         plt.title("Mouse Movement " + str(self.hit_targ) + " Episodes")
#                 #         plt.xlabel('x')
#                 #         plt.show()
#                 print("WOOO")
#                 print("WOOO")
#                 print("WOOO")
#             return np.array(self.state), reward, True, self.log_this
#         else:
#             if self.steps_beyond_done >= 0:
#                 logger.warn(
#                     "You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
#             self.steps_beyond_done += 1
#             reward = -300.0
#
#         return np.array(self.state), reward, done, self.log_this
#
#     @staticmethod
#     def angle_to_goal(x, y, target_x, target_y):
#         return math.atan2(y - target_y, x - target_x)
#
