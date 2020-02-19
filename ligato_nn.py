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


def create_train_data(game: lig.LigatoGame, opponent: lig.LigatoAI, model, num_games: int = 1):
    x = []
    y = []
    r = []
    for g in range(num_games):
        # Initialize random game
        game.random_state(start=True, seed=int(time.time()))
        done = False
        x_this_game = []
        y_this_game = []
        r_this_game = []
        while not done:
            # Observation
            bs = preproces_input(game.board_state)

            # Determine action from neural network
            all_actions = lig.check_available_actions(game.board_state, return_all=True)
            if all(a is None for a in all_actions):
                index = np.random.choice(range(model.num_actions))
                action = None
            else:
                all_probs = model.forward(np.expand_dims(bs, axis=0))
                actions = []
                probs = []
                for a, p in zip(all_actions, all_probs):
                    if a is not None:
                        actions.append(a)
                        probs.append(p)
                probs = np.array(probs) / np.array(probs).sum()
                index = np.random.choice(range(len(actions)), p=probs)
                action = actions[index]

            # Take action
            game.move(0, action=action)

            # Check if game is finished and determine reward
            if game.winner == 0:
                reward = 1
                done = True
            elif game.winner == 1:
                reward = -1
                done = True
            elif game.winner == -1:
                reward = 0
                done = True
            else:
                reward = 0

            # add to data arrays
            x_this_game.append(bs)
            y_this_game.append(index)
            r_this_game.append(reward)

            # check if done
            if done and ((reward == -1) or (reward == 1)):
                x.extend(x_this_game)
                y.extend(y_this_game)
                r.extend(r_this_game)

            # Let opponent take step
            board_state = game.get_board_state(player=1)
            opp_action = opponent.play(board_state=board_state)
            game.move(player=1, action=opp_action)

    # Lists to numpy arrays
    x = np.stack(x)
    y = np.stack(y)
    r = np.stack(r)
    return x, y, r


def discount_rewards(r: np.array, gamma: float):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r, dtype='float32')
    running_add = 0
    for t in reversed(range(r.size)):
        if r[t] != 0:
            running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def run_training(game, opponent, model, times, num_games):
    for t in range(times):
        x, y, r, = create_train_data(game=game, opponent=opponent, model=model, num_games=num_games)
        score = r.sum()
        wins = np.where(r == 1)[0].size
        losses = np.where(r == -1)[0].size
        print("Trainstep %s: %s games played. \t Score: %s \t wins: %s \t losses: %s" % (t, num_games, score, wins, losses))
        model.store_transitions(x, y, r)
        model.learn()
        model.train_log = model.train_log.append({'lr': model.lr,
                                                  'num_games': num_games ,
                                                  'score': score,
                                                  'gamma': model.gamma}, ignore_index=True)



class LigatoNN:
    def __init__(self, board_size, lr=1e-5, gamma=0.9, chkpt_dir='lig_checkpoints'):
        self.lr = lr
        self.gamma = gamma
        self.input_size = 2 * board_size[0] * board_size [1]
        self.num_actions = 2 * board_size[1]
        self.state_memory = []
        self.action_memory = []
        self.disc_reward_memory = []
        self.sess = tf.Session()
        self.build_net()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.train_log = pd.DataFrame(columns=['lr', 'num_games', 'score', 'gamma'])
        self.chkpt_dir = chkpt_dir
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)

    def build_net(self):
        with tf.variable_scope('parameters'):
            self.input = tf.placeholder(tf.float32, shape=(None, self.input_size), name='input')
            self.label = tf.placeholder(tf.int32, shape=(None, self.num_actions), name='label')
            self.G = tf.placeholder(tf.float32, shape=(None,), name='disc_reward')

        with tf.variable_scope('fc1'):
            fc1 = tf.layers.dense(self.input, units=256)
            fc1_activated = tf.nn.relu(fc1)

        with tf.variable_scope('fc2'):
            fc2 = tf.layers.dense(fc1_activated, units=self.num_actions)

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

    def store_transitions(self, observations, actions, rewards):
        self.state_memory = observations
        self.action_memory = np.eye(self.num_actions)[actions]
        discounted_rewards = discount_rewards(np.array(rewards), self.gamma)
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
