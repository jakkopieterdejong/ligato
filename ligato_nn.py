import ligato as lig
import numpy as np
import tensorflow as tf
import os
import time
import pandas as pd


# preproces the board_state of a LigatoGame into input data for the Ligato Neural Network AI
# The AI is considered to be player 0 on the board_state.
def preproces_input(board_state):
    p0_vector = (board_state == 0).astype('int16').flatten()
    p1_vector = (board_state == 1).astype('int16').flatten()
    input_vector = np.concatenate((p0_vector, p1_vector))
    return input_vector


class LigatoNN:
    def __init__(self, input_size, lr=0.05, chkpt_dir='lig_checkpoints'):
        self.lr = lr
        self.input_size = input_size
        self.state_memory = []
        self.action_memory = []
        self.disc_reward_memory = []
        self.sess = tf.Session()
        self.build_net()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.train_log = pd.DataFrame(columns=['lr', 'num_eps', 'mean_win', 'mean_turns', 'gamma'])
        self.chkpt_dir = chkpt_dir
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)

    def build_net(self):
        with tf.variable_scope('parameters'):
            self.input = tf.placeholder(tf.float32, shape=(None, self.input_size), name='input')
            self.label = tf.placeholder(tf.int32, shape=(None, 12), name='label')
            self.G = tf.placeholder(tf.float32, shape=(None,), name='disc_reward')

        with tf.variable_scope('fc1'):
            fc1 = tf.layers.dense(self.input, units=256)
            fc1_activated = tf.nn.relu(fc1)

        with tf.variable_scope('fc2'):
            fc2 = tf.layers.dense(fc1_activated, units=12)

        self.actions = tf.nn.softmax(fc2, name='actions')

        with tf.variable_scope('loss'):
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=fc2, labels=self.label)
            self.loss = tf.reduce_mean(neg_log_prob * self.G)

        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss)
            self.reset_optimizer_op = tf.variables_initializer(optimizer.variables())

    def reset_optimizer(self):
        self.sess.run(self.reset_optimizer_op)

    def forward(self, observation):
        probabilities = self.sess.run(self.actions, feed_dict={self.input: observation})[0]
        return probabilities

    def store_transitions(self, observations, actions, discounted_rewards):
        self.state_memory = observations
        self.action_memory = np.eye(12)[actions]
        mean = np.mean(discounted_rewards)
        std = np.std(discounted_rewards) if np.std(discounted_rewards) > 0 else 1
        discounted_rewards = (discounted_rewards - mean) / std
        self.disc_reward_memory = discounted_rewards

    def learn(self, print_reports=False):
        start_time = time.time()
        inputs = self.state_memory.copy()
        labels = self.action_memory.copy()
        disc_rewards = self.disc_reward_memory.copy()
        _ = self.sess.run(self.train_op, feed_dict={self.input: inputs, self.label: labels, self.G: disc_rewards})
        if print_reports:
            print("Finished training epoch in %s seconds." % (time.time() - start_time))

    def load_checkpoint(self, filename):
        print("... Loading Checkpoint ...")
        checkpoint_path = os.path.join(self.chkpt_dir, filename)
        csv_path = os.path.join(self.chkpt_dir, filename + '.csv')
        self.saver.restore(self.sess, checkpoint_path)
        self.train_log = pd.DataFrame.from_csv(csv_path)

    def save_checkpoint(self, filename):
        print("... Saving Checkpoint ...")
        checkpoint_path = os.path.join(self.chkpt_dir, filename)
        csv_path = os.path.join(self.chkpt_dir, filename + '.csv')
        self.saver.save(self.sess, checkpoint_path)
        self.train_log.to_csv(csv_path)
