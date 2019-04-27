import typing as tp
import numpy as np

import tensorflow as tf
from keras import Sequential
import keras.layers as layers
from keras.optimizers import Adam


class Discriminator:
    def __init__(self, expert_data: np.ndarray):
        self.expert_data = expert_data
        self.num_samples = expert_data.shape[0]

        val_std = np.std(self.expert_data)
        val_avg = np.average(self.expert_data)
        self.normalize = lambda trajectories: (trajectories - val_avg) / val_std

        self.discriminator = Sequential()
        self.discriminator.add(layers.Flatten(input_shape=expert_data[0].shape))
        self.discriminator.add(layers.Dense(32))
        self.discriminator.add(layers.LeakyReLU(alpha=0.4))
        self.discriminator.add(layers.Dense(32))
        self.discriminator.add(layers.LeakyReLU(alpha=0.2))
        self.discriminator.add(layers.Dense(1, activation='sigmoid'))
        self.discriminator.summary()
        # self.discriminator.compile(
        #     optimizer=tf.train.AdamOptimizer(),
        #     loss='sparse_categorical_crossentropy',
        #     metrics=['accuracy'],
        # )

        optimizer = Adam(0.0002, 0.5)
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )

    def pretrain_discriminator(self):
        pass
        # the discriminator should at least be able to distinguish random noise
        batch_size = 32

        for i in range(4):
            expert_trajectories = self.sample_expert_data(batch_size)
            noise = np.random.normal(0.0, 1.0, size=expert_trajectories.size).reshape(expert_trajectories.shape)

            x_train = np.concatenate((noise, expert_trajectories))
            y_train = np.concatenate((np.zeros(batch_size), np.ones(batch_size)))

            loss = self.discriminator.train_on_batch(x_train, y_train)
            if i % 8 == 0:
                print("pretrain loss: {}".format(loss))

    def predict(self, trajectory: np.ndarray):
        inputs = self.normalize(np.array([trajectory]))
        return self.discriminator.predict_on_batch(inputs)[0][0]

    def sample_expert_data(self, n: int):
        ids = np.random.randint(0, self.num_samples, size=n)
        return self.normalize(self.expert_data[ids])

    def train(self, trajectories: np.ndarray, expert_trajectories: tp.Optional[np.ndarray] = None):
        trajectories = self.normalize(trajectories)

        n_trajectories = len(trajectories)
        if expert_trajectories is None:
            expert_trajectories = self.sample_expert_data(n_trajectories)
        else:
            assert n_trajectories == len(expert_trajectories)

        x_train = np.concatenate((trajectories, expert_trajectories))
        y_train = np.concatenate((np.zeros(n_trajectories), np.ones(n_trajectories)))

        return self.discriminator.train_on_batch(x_train, y_train)
