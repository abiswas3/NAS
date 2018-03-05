import tensorflow as tf

class CNN():
    def __init__(self,
                 num_input,
                 num_classes,
                 cnn_config):

        '''
        num_input  : input dimensions
        num_classes: number of output classes
        cnn_config : list of lists with each list being layer

        one layer has the form [w x y z]

        w :
        x : number of cnn filters
        y : max pooling layer size
        z : cnn_drop_rate

        '''

        # dimension of conv filter per layer
        cnn = [c[0] for c in cnn_config]

        # list of cnn filters for each layer
        cnn_num_filters = [c[1] for c in cnn_config]

        # list of maxpool filer size
        max_pool_ksize = [c[2] for c in cnn_config]

        self.X = tf.placeholder(tf.float32,
                                [None, num_input], 
                                name="input_X")
        
        self.Y = tf.placeholder(tf.int32, [None, num_classes], name="input_Y")
        
        self.dropout_keep_prob = tf.placeholder(tf.float32, [], name="dense_dropout_keep_prob")
        
        self.cnn_dropout_rates = tf.placeholder(tf.float32, [len(cnn), ], name="cnn_dropout_keep_prob")

        Y = self.Y
        X = tf.expand_dims(self.X, -1)
        pool_out = X
        
        with tf.name_scope("Conv_part"):
            
            for idd, filter_size in enumerate(cnn):
                with tf.name_scope("L"+str(idd)):
                    conv_out = tf.layers.conv1d(
                        pool_out,
                        filters=cnn_num_filters[idd],
                        kernel_size=(int(filter_size)),
                        strides=1,
                        padding="SAME",
                        name="conv_out_"+str(idd),
                        activation=tf.nn.relu,
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        bias_initializer=tf.zeros_initializer
                    )
                    
                    pool_out = tf.layers.max_pooling1d(
                        conv_out,
                        pool_size=(int(max_pool_ksize[idd])),
                        strides=1,
                        padding='SAME',
                        name="max_pool_"+str(idd)
                    )
                    
                    pool_out = tf.nn.dropout(pool_out,
                                             self.cnn_dropout_rates[idd])

            # flatten the last layer
            flatten_pred_out = tf.contrib.layers.flatten(pool_out)

            # This is the fully connected latyer going from flatten_pred to final num_classes
            self.logits = tf.layers.dense(flatten_pred_out, num_classes)

        # normalize
        self.prediction = tf.nn.softmax(self.logits, name="prediction")

        # this is the shit we need to minimize
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=Y, name="loss")
        
        correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")
