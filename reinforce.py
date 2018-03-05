import tensorflow as tf
import random
import numpy as np

class Reinforce():
    def __init__(self,
                 sess,
                 optimizer,
                 policy_network,
                 max_layers,
                 global_step,
                 division_rate= 100.0,
                 reg_param=0.001,
                 discount_factor=0.99,
                 exploration=0.3):

        '''
        Notation:
        policy network : used describe model that predicts hyperparameters
        learned network :  learned network with hyper params as recommended

        Args:
        sess: tensorflow session
        optimizer : type of optimization algorithm used for minimization
        policy network : final tensorflow output state of the policy network
        max_layers: the maximum number of layers for the learned neural network
        global_step : number of cycles of learning of policy network (i,e gradient updates)
        reg_param : lambda for l2 regularizaion of loss of policy network
        discoun_factor : as stated
        exploration : not used for anything right now (but meant for random exploration)
        '''
        
        self.sess = sess
        self.optimizer = optimizer
        self.policy_network = policy_network 
        self.division_rate = division_rate
        self.reg_param = reg_param
        self.discount_factor=discount_factor
        self.max_layers = max_layers
        self.global_step = global_step

        self.reward_buffer = []
        self.state_buffer = []

        self.create_variables()
        var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.sess.run(tf.variables_initializer(var_lists))

    def get_action(self, state):
        '''Given the state of the neural network (Rewards so far are stored
        interanally as member variables) get new state.
        '''
        return self.sess.run(self.predicted_action, {self.states: state})

    def create_variables(self):

        with tf.name_scope("model_inputs"):
            # raw state representation
            self.states = tf.placeholder(tf.float32, [None, self.max_layers*4], name="states")

        with tf.name_scope("predict_actions"):
            
            # initialize policy network
            with tf.variable_scope("policy_network"):

                # In this case this is just the final state of the RNN
                self.policy_outputs = self.policy_network(self.states,
                                                          self.max_layers)

            # Identity is used to remember the last policy_output how
            # tf.identity works isn't completely clear to me but for
            # now I'll trust that this works: it's basically deep copy
            self.action_scores = tf.identity(self.policy_outputs,
                                             name="action_scores")

            # Scale them and cast them into int:
            # Note this doesn't depend on the reward
            # All that matters is the hidden weights of my policy controller
            # The reward is used to update those weights
            self.predicted_action = tf.cast(tf.scalar_mul(self.division_rate, self.action_scores),
                                            tf.int32,
                                            name="predicted_action")


        # regularization loss
        policy_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_network")

        # compute loss and gradients
        with tf.name_scope("compute_gradients"):
            # gradients for selecting action from policy network
            self.discounted_rewards = tf.placeholder(tf.float32, (None,), name="discounted_rewards")

            with tf.variable_scope("policy_network", reuse=True):
                self.logprobs = self.policy_network(self.states,
                                                    self.max_layers)
                
                print("self.logprobs", self.logprobs)

            # compute policy loss and regularization loss
            self.cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logprobs[:, -1, :],
                                                                              labels=self.states)
            
            self.pg_loss            = tf.reduce_mean(self.cross_entropy_loss)
            self.reg_loss           = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in policy_network_variables]) # L2 by the look of itRegularization
            self.loss               = self.pg_loss + self.reg_param * self.reg_loss

            #compute gradients
            self.gradients = self.optimizer.compute_gradients(self.loss)
            
            # compute policy gradients
            for i, (grad, var) in enumerate(self.gradients):
                if grad is not None:
                    self.gradients[i] = (grad * self.discounted_rewards, var)

            # training update
            with tf.name_scope("train_policy_network"):
                # apply gradients to update policy network
                self.train_op = self.optimizer.apply_gradients(self.gradients,
                                                               global_step=self.global_step)

    def storeRollout(self, state, reward):
        '''Caching for the win: for long running programs this is a shite
        solution
        '''
        self.reward_buffer.append(reward)
        self.state_buffer.append(state[0])

        
    def train_step(self, steps_count):
        '''
        This is where policy gradientx happens 
        but to understand this also understand create_variable function
        
        steps_count: how many previous states to consider
        '''

        # take the last steps_count number of states
        states = np.array(self.state_buffer[-steps_count:])/self.division_rate

        # rewards are never discounted
        rewars = self.reward_buffer[-steps_count:]
        
        _, ls = self.sess.run([self.train_op, self.loss],
                     {self.states: states,
                      self.discounted_rewards: rewars})
        
        return ls
