import numpy as np
import tensorflow as tf
import argparse

from cnn import CNN
from net_manager import NetManager
from reinforce import Reinforce

from colors import colors

import datetime
from tensorflow.examples.tutorials.mnist import input_data


'''
driver pogram
'''

def parse_args():
    desc = "TensorFlow implementation of 'Neural Architecture Search with Reinforcement Learning'"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--max_layers', default=2)

    args = parser.parse_args()
    args.max_layers = int(args.max_layers)
    return args


'''
    Policy network is a main network for searching optimal architecture
    it uses NAS - Neural Architecture Search recurrent network cell.
    https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/contrib/rnn/python/ops/rnn_cell.py#L1363

    Args:
        state: current state of required topology
        max_layers: maximum number of layers
    Returns:
        3-D tensor with new state (new topology)
'''
def policy_network(state, max_layers):
    with tf.name_scope("policy_network"):
        nas_cell = tf.contrib.rnn.NASCell(4*max_layers)
        outputs, state = tf.nn.dynamic_rnn(
            nas_cell,
            tf.expand_dims(state, -1),
            dtype=tf.float32
        )
        bias = tf.Variable([0.05]*4*max_layers)
        outputs = tf.nn.bias_add(outputs, bias)
        print("outputs: ", outputs, outputs[:, -1:, :],  tf.slice(outputs, [0, 4*max_layers-1, 0], [1, 1, 4*max_layers]))
        #return tf.slice(outputs, [0, 4*max_layers-1, 0], [1, 1, 4*max_layers]) # Returned last output of rnn
        return outputs[:, -1:, :]

def train(mnist):
    global args

    sess = tf.Session()
    # basically counts the number of reps of the whole RL loop
    # we've done
    global_step = tf.Variable(0, trainable=False)

    starter_learning_rate = 0.1
    learning_rate = tf.train.exponential_decay(0.99, global_step,
                                           500, 0.96, staircase=True)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

    # This the controller: used to generate new state describing the neural network
    reinforce = Reinforce(sess, optimizer, policy_network, args.max_layers, global_step)
    net_manager = NetManager(num_input=784,
                             num_classes=10,
                             learning_rate=0.001,
                             mnist=mnist,
                             bathc_size=100)

    MAX_EPISODES = 2500
    step = 0
    state = np.array([[10.0, 128.0, 1.0, 1.0]*args.max_layers], dtype=np.float32)
    pre_acc = 0.0
    total_rewards = 0
    
    for i_episode in range(MAX_EPISODES):
        print(colors.fg.orange, state, colors.reset)
        action = reinforce.get_action(state)

        print("ca:", action)

        # This is just a sanity check that all the configurables are
        # positive. If not the reward is -1 so worse than getting it all wrong.
        if all(ai > 0 for ai in action[0][0]):

            # So I have a state: now I need to hand over contril to network manager
            # The steps involve: (Just regular supervised learning)
            # 1 Generate a neural network based on cofiguration
            # 2 Train it
            # 3 Get test accuracyx
            reward, pre_acc = net_manager.get_reward(action,
                                                     step,
                                                     pre_acc)
            print("=====>", reward, pre_acc)
        else:
            reward = -1.0

        total_rewards += reward

        # Now our new action is equal to the new state
        # I wonder why this author used this fuck all indexing
        # but whatever i'll change it all later anyway
        state = action[0]

        # Cache the new state and last reward
        reinforce.storeRollout(state, reward)

        step += 1
        # Get a new state
        ls = reinforce.train_step(1)

        # Log shit so I can recover in case shit breaks
        log_str = "current time:  "+str(datetime.datetime.now().time())+" episode:  "+str(i_episode)+" loss:  "+str(ls)+" last_state:  "+str(state)+" last_reward:  "+str(reward)+"\n"
        log = open("lg3.txt", "a+")
        log.write(log_str)
        log.close()
        print(log_str)

def main():
    global args
    args = parse_args()

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train(mnist)

if __name__ == '__main__':
  main()
