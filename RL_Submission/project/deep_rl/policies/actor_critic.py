import math
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from gym.spaces import Discrete
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numba import jit

from project.deep_rl.policies.base import BaseModelMixin, Config, Policy
from project.deep_rl.policies.memory import ReplayMemory, Transition
from project.deep_rl.utils.misc import plot_learning_curve, plot_from_monitor_results, REPO_ROOT
from project.deep_rl.utils.tf_ops import dense_nn
import project.util.bresenham_line as bresen

import matplotlib.pyplot as plt
import matplotlib.ticker

import seaborn as sns
import pandas as pd


def plot_model_progress(filename, reward_history, step_history, batch_size=10):
    ax1: Axes
    ax2: Axes
    fig: Figure
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12, 4 * 1))
    ax2 = ax1.twinx()

    n_batch = len(reward_history) // batch_size
    data = pd.DataFrame({
        'batch': np.arange(start=1, stop=n_batch + 1).repeat(batch_size) * batch_size,
        'length': np.array(step_history),
        'reward': np.array(reward_history),
    })

    ax1 = sns.lineplot(x="batch", y="reward", ax=ax1, data=data, label="Reward")
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(left=batch_size, right=len(reward_history))
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Episode Reward")

    ax2 = sns.lineplot(x="batch", y="length", ax=ax2, data=data, color='#CC4F1B', label="Length")
    ax2.set_ylim(bottom=0)
    ax2.set_ylabel("Episode Length")
    ax2.grid(False)
    plt.tight_layout()

    # legend insanity
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
    ax1.legend(lines, labels, loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)
    ax2.legend().remove()

    # y tick insanity
    ticker = matplotlib.ticker.MaxNLocator(
        nbins=8,
        steps=[1, 2, 5, 10])

    def get_ticks(ax):
        bottom, top = ax.get_ylim()
        return ticker.tick_values(bottom, top)

    ticks1 = list(get_ticks(ax1))
    ticks2 = list(get_ticks(ax2))

    if max(len(ticks1), len(ticks2)) < 2:
        print("bad ticks, skipping")
        return

    def adjust_lengths(t1, t2):
        dt = t1[1] - t1[0]
        while len(t1) < len(t2):
            t1.append(t1[-1] + dt)

    adjust_lengths(ticks1, ticks2)
    adjust_lengths(ticks2, ticks1)

    print("tick lengths:", len(ticks1), len(ticks2))

    ax1.set_ylim(bottom=ticks1[0], top=ticks1[-1])
    ax1.set_yticks(ticks1)

    ax2.set_ylim(bottom=ticks2[0], top=ticks2[-1])
    ax2.set_yticks(ticks2)

    os.makedirs(os.path.join(REPO_ROOT, 'figs'), exist_ok=True)
    plt.savefig(os.path.join(REPO_ROOT, 'figs', filename))
    plt.close(fig)


def plot_model_progress2(filename, reward_history, step_history, batch_size=10):
    ax1: Axes
    ax2: Axes
    fig: Figure
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 4 * 2))
    xlabel = "episode batch [{}]".format(batch_size)

    n_batch = len(reward_history) // batch_size
    data = pd.DataFrame({
        'batch': np.arange(start=0, stop=n_batch).repeat(batch_size) + 1,
        'length': np.array(step_history),
        'reward': np.array(reward_history),
    })

    # ax1 = sns.lineplot(x="batch", y="rewards", err_style="ci_band", ci=90, data=data)
    ax1 = sns.lineplot(x="batch", y="reward", ax=ax1, data=data)
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(left=1, right=n_batch)
    ax1.set_xlabel(xlabel)

    ax2 = sns.lineplot(x="batch", y="length", ax=ax2, data=data)
    ax2.set_ylim(bottom=0)
    ax2.set_xlim(left=1, right=n_batch)
    ax2.set_xlabel(xlabel)

    plt.tight_layout()
    os.makedirs(os.path.join(REPO_ROOT, 'figs'), exist_ok=True)
    plt.savefig(os.path.join(REPO_ROOT, 'figs', filename))


class BinMapper:
    def __init__(
            self,
            bins_x: int,
            bins_y: int,
            x_low: int,
            x_high: int,
            y_low: int,
            y_high: int
    ):
        self.bins_x = bins_x
        self.bins_y = bins_y

        self.x_low = x_low
        self.y_low = y_low

        self.bin_x_width = (x_high - x_low) / bins_x
        self.bin_y_width = (y_high - y_low) / bins_y

    def is_in_bins(self, bin_pos):
        return 0 <= bin_pos[0] < self.bins_x and 0 <= bin_pos[1] < self.bins_y

    def to_bin_coords(self, pos):
        # bx = np.floor((pos[0] - self.x_low) / self.bin_x_width).astype(int)
        # by = np.floor((pos[1] - self.y_low) / self.bin_y_width).astype(int)
        bx = math.floor((pos[0] - self.x_low) / self.bin_x_width)
        by = math.floor((pos[1] - self.y_low) / self.bin_y_width)
        return bx, by


class ActorCriticPolicy(Policy, BaseModelMixin):

    def __init__(self, env, name, training=True, gamma=0.9, layer_sizes=None, clip_norm=None, **kwargs):
        Policy.__init__(self, env, name, training=training, gamma=gamma, **kwargs)
        BaseModelMixin.__init__(self, name)

        assert isinstance(self.env.action_space, Discrete), \
            "Current ActorCriticPolicy implementation only works for discrete action space."

        self.layer_sizes = [64] if layer_sizes is None else layer_sizes
        self.clip_norm = clip_norm
        self.movements = None
        self.normalized_movements = None
        self.exploration_histogram = None
        self.hist_mapper = None

    def setup_experts_direction_distribution(self):
        print("making normalized movements for human exploration...")
        # shape is two four tuples (-x, -y, -target_x, -target_y),(x, y, target_x, target_y)
        shape = [None] + self.state_dim
        if self.movements is None:
            print("movements were none")

        normalized_movements = np.zeros(np.shape(np.array(self.movements)))
        for idx, m in enumerate(self.movements):
            dx = m[-1, 0]
            dy = m[-1, 1]
            length = np.hypot(dx, dy)

            theta = -np.arctan2(dy, dx)
            rot = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ])

            normalized_movements[idx, :, :] = np.dot(rot, m.T).T / length
        self.normalized_movements = normalized_movements

        # making the exploration histogram
        theta_bins = 16
        mapper = BinMapper(
            bins_x=600,
            bins_y=300,
            x_low=-4,
            x_high=5,
            y_low=-4,
            y_high=4,
        )
        self.hist_mapper = mapper

        expected_shape = (mapper.bins_x, mapper.bins_y, theta_bins)

        hist_file = "histogram.npz"
        try:
            loaded_histogram = np.load(hist_file)['histogram']
            if loaded_histogram is not None and tuple(loaded_histogram.shape) == tuple(expected_shape):
                self.exploration_histogram = loaded_histogram
            else:
                print("wrong shape for loaded histogram {} vs expected {}"
                      .format(loaded_histogram.shape, expected_shape))
        except Exception as e:
            print("failed to load histogram {}".format(e))

        if self.exploration_histogram is None:
            exploration_histogram = np.ones(expected_shape)

            total_mv = len(normalized_movements)
            report_mv = total_mv // 200

            start_time = time.time()
            for im, m in enumerate(normalized_movements):
                if im > 0 and im % report_mv == 0:
                    print("normalizing: {:.1f}%, {:.1f}s left".format(
                        100.0 * im / total_mv,
                        (total_mv - im) / (im / (time.time() - start_time))
                    ))
                    sc = mapper.to_bin_coords([0, 0])
                    ec = mapper.to_bin_coords([1, 0])
                    print(sc, "start ", exploration_histogram[sc])
                    print(ec, "end", exploration_histogram[ec])

                for idx in range(len(m) - 1):
                    a = m[idx]
                    b = m[idx + 1]

                    delta = b - a
                    theta = (np.arctan2(delta[1], delta[0]) + np.pi) / (2 * np.pi)
                    theta = int(theta * theta_bins) % theta_bins

                    bin_a = mapper.to_bin_coords(a)
                    bin_b = mapper.to_bin_coords(b)

                    if idx == 0 and mapper.is_in_bins(bin_a):
                        index = (bin_a[0], bin_a[1], theta)
                        exploration_histogram[index] += 1.0

                    for bin_pt in bresen.bresenhamline(np.array([bin_a]), np.array([bin_b])):
                        if mapper.is_in_bins(bin_pt):
                            index = (bin_pt[0], bin_pt[1], theta)
                            exploration_histogram[index] += 1.0

            np.savez(hist_file, histogram=exploration_histogram)
            self.exploration_histogram = exploration_histogram

        # normalize so total is one
        self.exploration_histogram /= np.sum(self.exploration_histogram, axis=2, keepdims=True)
        # add fixed probability to pick a random direction
        self.exploration_histogram = self.exploration_histogram * 0.8 + 0.2 / theta_bins

        # turn into CDF
        self.exploration_histogram = np.cumsum(self.exploration_histogram, axis=2)
        print("start", self.exploration_histogram[mapper.to_bin_coords([0, 0])])
        print("end", self.exploration_histogram[mapper.to_bin_coords([1, 0])])

    def act(self, state, eps=0.1):
        # Discrete actions
        # If exploring:
        if self.training and np.random.random() < eps:
            x, y, tx, ty = state
            length = np.hypot(ty, tx)
            theta = -np.arctan2(ty, tx)

            rot = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ])
            point = np.array([x, y])
            normalized_point = np.dot(rot, point.T).T / length

            bin_point = self.hist_mapper.to_bin_coords(normalized_point)
            # randomly sample if we aren't within the histogram
            if not self.hist_mapper.is_in_bins(bin_point):
                return self.env.action_space.sample()

            # Now access the 3d histogram and sample the direction we want
            theta_distribtion = self.exploration_histogram[bin_point]
            p = np.random.random_sample()

            random_theta = None
            theta_bin_size = (2 * np.pi) / len(theta_distribtion)

            for idx, cdf in enumerate(theta_distribtion):
                if p <= cdf:
                    # note: we need to subtract pi because we added it before when putting it in the histogram
                    random_theta = ((np.random.random_sample() + idx) * theta_bin_size) - np.pi
                    break

            assert random_theta is not None
            # need to undo our rotation into normal coordinates
            random_theta = random_theta - theta
            # also pick a random distance to move
            u = np.random.random_sample() + np.random.random_sample()
            if u > 1:
                random_r = 2 - u
            else:
                random_r = u

            random_r *= self.env.max_delta

            rx = random_r * math.cos(random_theta)
            ry = random_r * math.sin(random_theta)

            best_action = None
            best_action_dist = None
            for idx, (ax, ay) in enumerate(self.env.action_array):
                dist = math.hypot(rx - ax, ry - ay)
                if best_action_dist is None or dist < best_action_dist:
                    best_action_dist = dist
                    best_action = idx

            assert best_action is not None
            return best_action

        # return self.sess.run(self.sampled_actions, {self.states: [state]})
        proba = self.sess.run(self.actor_proba, {self.s: [state]})[0]
        return max(range(self.act_size), key=lambda i: proba[i])

    def _build_networks(self):
        # Define input placeholders
        self.s = tf.placeholder(tf.float32, shape=[None] + self.state_dim, name='state')
        self.a = tf.placeholder(tf.int32, shape=(None,), name='action')
        self.s_next = tf.placeholder(tf.float32, shape=[None] + self.state_dim, name='next_state')
        self.r = tf.placeholder(tf.float32, shape=(None,), name='reward')
        self.done = tf.placeholder(tf.float32, shape=(None,), name='done_flag')

        # Actor: action probabilities
        self.actor = dense_nn(self.s, self.layer_sizes + [self.act_size], name='actor')
        self.sampled_actions = tf.squeeze(tf.multinomial(self.actor, 1))
        self.actor_proba = tf.nn.softmax(self.actor)
        self.actor_vars = self.scope_vars('actor')

        # Critic: action value (V value)
        self.critic = dense_nn(self.s, self.layer_sizes + [1], name='critic')
        self.critic_next = dense_nn(self.s_next, self.layer_sizes + [1], name='critic', reuse=True)
        self.critic_vars = self.scope_vars('critic')

        # TD target
        self.td_target = self.r + self.gamma * tf.squeeze(self.critic_next) * (1.0 - self.done)
        self.td_error = self.td_target - tf.squeeze(self.critic)

    def _build_train_ops(self):
        self.lr_c = tf.placeholder(tf.float32, shape=None, name='learning_rate_c')
        self.lr_a = tf.placeholder(tf.float32, shape=None, name='learning_rate_a')

        with tf.variable_scope('critic_train'):
            # self.reg_c = tf.reduce_mean([tf.nn.l2_loss(x) for x in self.critic_vars])
            self.loss_c = tf.reduce_mean(tf.square(self.td_error))  # + 0.001 * self.reg_c
            self.optim_c = tf.train.AdamOptimizer(self.lr_c)
            self.grads_c = self.optim_c.compute_gradients(self.loss_c, self.critic_vars)
            if self.clip_norm:
                self.grads_c = [(tf.clip_by_norm(grad, self.clip_norm), var) for grad, var in self.grads_c]

            self.train_op_c = self.optim_c.apply_gradients(self.grads_c)

        with tf.variable_scope('actor_train'):
            # self.reg_a = tf.reduce_mean([tf.nn.l2_loss(x) for x in self.actor_vars])
            # self.entropy_a =- tf.reduce_sum(self.actor * tf.log(self.actor))
            self.loss_a = tf.reduce_mean(
                tf.stop_gradient(self.td_error) * tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.actor, labels=self.a), name='loss_actor')  # + 0.001 * self.reg_a
            self.optim_a = tf.train.AdamOptimizer(self.lr_a)
            self.grads_a = self.optim_a.compute_gradients(self.loss_a, self.actor_vars)
            if self.clip_norm:
                self.grads_a = [(tf.clip_by_norm(grad, self.clip_norm), var) for grad, var in self.grads_a]

            self.train_op_a = self.optim_a.apply_gradients(self.grads_a)

        with tf.variable_scope('summary'):
            self.ep_reward = tf.placeholder(tf.float32, name='episode_reward')
            self.summary = [
                tf.summary.scalar('loss/critic', self.loss_c),
                tf.summary.scalar('loss/actor', self.loss_a),
                tf.summary.scalar('episode_reward', self.ep_reward)
            ]
            self.summary += [tf.summary.scalar('grads/a_' + var.name, tf.norm(grad)) for
                             grad, var in self.grads_a if grad is not None]
            self.summary += [tf.summary.scalar('grads/c_' + var.name, tf.norm(grad)) for
                             grad, var in self.grads_c if grad is not None]
            self.merged_summary = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)

        self.train_ops = [self.train_op_a, self.train_op_c]

        self.sess.run(tf.global_variables_initializer())

    def build(self):
        self._build_networks()
        self._build_train_ops()

    class TrainConfig(Config):
        lr_a = 0.02
        lr_a_decay = 0.995
        lr_c = 0.01
        lr_c_decay = 0.995
        batch_size = 32
        n_episodes = 800
        warmup_episodes = 1000
        log_every_episode = 10
        done_rewards = -100
        # for epsilon-greedy exploration
        epsilon = 1.0
        epsilon_final = 0.05

    def train(self, config: TrainConfig):
        buffer = ReplayMemory(tuple_class=Transition)

        step = 0
        episode_reward = 0.
        reward_history = []
        reward_averaged = []

        step_history = []

        lr_c = config.lr_c
        lr_a = config.lr_a

        eps = config.epsilon
        warmup_episodes = config.warmup_episodes or config.n_episodes
        eps_drop = (eps - config.epsilon_final) / warmup_episodes
        print("Decrease epsilon per step:", eps_drop)

        # get expert_data.movements
        self.movements = self.env.get_expert_data()
        self.setup_experts_direction_distribution()

        # let env know about us
        self.env.set_actor(self)
        for n_episode in range(config.n_episodes):
            ob = self.env.reset()
            self.act(ob, eps)
            done = False

            while not done:
                a = self.act(ob, eps)
                ob_next, r, done, info = self.env.step(a)
                step += 1
                episode_reward += r

                record = Transition(self.obs_to_inputs(ob), a, r, self.obs_to_inputs(ob_next), done)
                buffer.add(record)

                ob = ob_next

                if done:
                    batch = buffer.pop(self.env.steps)
                    # set rewards to average of all rewards
                    batch['r'].fill(np.average(batch['r']))
                    _, summ_str = self.sess.run(
                        [self.train_ops, self.merged_summary], feed_dict={
                            self.lr_c: lr_c,
                            self.lr_a: lr_a,
                            self.s: batch['s'],
                            self.a: batch['a'],
                            self.r: batch['r'],
                            self.s_next: batch['s_next'],
                            self.done: batch['done'],
                            self.ep_reward: batch['r'].sum(),
                        })
                    self.writer.add_summary(summ_str, step)

            # One trajectory is complete!
            step_history.append(self.env.steps)
            reward_history.append(episode_reward)
            reward_averaged.append(np.mean(reward_history[-10:]))
            episode_reward = 0.

            lr_c *= config.lr_c_decay
            lr_a *= config.lr_a_decay
            if eps > config.epsilon_final:
                eps -= eps_drop

            # render monitor plot as well
            plot_batch_size = 20
            if len(step_history) % plot_batch_size == 0 and len(step_history) > plot_batch_size:
                plot_model_progress(
                    self.model_name + "-progress",
                    reward_history,
                    step_history,
                    batch_size=plot_batch_size)

            if (reward_history and config.log_every_episode and
                    n_episode % config.log_every_episode == 0):
                # Report the performance every `every_step` steps
                print(
                    "[episodes:{}/step:{}], best:{}, avg:{:.2f}:{}, lr:{:.4f}|{:.4f} eps:{:.4f}".format(
                        n_episode, step, np.max(reward_history),
                        np.mean(reward_history[-10:]), reward_history[-5:],
                        lr_c, lr_a, eps,
                    ))

        self.save_checkpoint(step=step)

        print("[FINAL] episodes: {}, Max reward: {}, Average reward: {}".format(
            len(reward_history), np.max(reward_history), np.mean(reward_history)))

        data_dict = {
            'reward': reward_history,
            'reward_smooth10': reward_averaged,
        }
        plot_learning_curve(self.model_name, data_dict, xlabel='episode')
