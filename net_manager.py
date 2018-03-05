import tensorflow as tf
from cnn import CNN

class NetManager():
    def __init__(self, num_input,
                 num_classes,
                 learning_rate,
                 mnist,
                 max_step_per_action=1000,
                 bathc_size=100,
                 dropout_rate=0.85):

        '''
        Args self explanatory
        '''
        self.num_input = num_input
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.mnist = mnist

        self.max_step_per_action = max_step_per_action
        self.bathc_size = bathc_size
        self.dropout_rate = dropout_rate

    def get_reward(self,
                   action,
                   step,
                   pre_acc):

        '''
        action : new configuration of neural network
        step   : episode number
        pre_acc: test accuracy of previous neural network

        each 4 tuple in action[0][0] is a layer:
        
        '''
        
        # parse the layers
        # now action is a list of list and each list is a layer
        action = [action[0][0][x:x+4] for x in range(0, len(action[0][0]), 4)]
        
        cnn_drop_rate = [c[3] for c in action]
        
        with tf.Graph().as_default() as g:
            
            with g.container('experiment'+str(step)):

                # 1. Training the CONV NET
                model = CNN(self.num_input, self.num_classes, action)                
                loss_op = tf.reduce_mean(model.loss)
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                train_op = optimizer.minimize(loss_op)

                with tf.Session() as train_sess:
                    init = tf.global_variables_initializer()
                    train_sess.run(init)

                    for step in range(self.max_step_per_action):
                        batch_x, batch_y = self.mnist.train.next_batch(self.bathc_size)
                        feed = {model.X: batch_x,
                                model.Y: batch_y,
                                model.dropout_keep_prob: self.dropout_rate,
                                model.cnn_dropout_rates: cnn_drop_rate}
                        
                        _ = train_sess.run(train_op, feed_dict=feed)

                        # This shit is just print training
                        # logs
                        if step % 50 == 0:
                            # Calculate batch loss and accuracy
                            loss, acc = train_sess.run(
                                [loss_op, model.accuracy],
                                feed_dict={model.X: batch_x,
                                           model.Y: batch_y,
                                           model.dropout_keep_prob: 1.0,
                                           model.cnn_dropout_rates: [1.0]*len(cnn_drop_rate)})
                            print("Step " + str(step) +
                                  ", Minibatch Training  Loss= " + "{:.4f}".format(loss) +
                                  ", Training accuracy= " + "{:.3f}".format(acc))

                    # Training over
                    batch_x, batch_y = self.mnist.test.next_batch(10000)
                    loss, acc = train_sess.run(
                                [loss_op, model.accuracy],
                                feed_dict={model.X: batch_x,
                                           model.Y: batch_y,
                                           model.dropout_keep_prob: 1.0,
                                           model.cnn_dropout_rates: [1.0]*len(cnn_drop_rate)})
                    
                    print("Reward accuracy:", acc, pre_acc)

                    # If accuracy got better return reward = acc, and test acc
                    if acc - pre_acc <= 0.01:
                        return acc, acc
                    
                    # If not give it a shitty reward, and return test acc
                    # Not sure if this is a great idea but wtf do i know
                    else:
                        return 0.01, acc                    
