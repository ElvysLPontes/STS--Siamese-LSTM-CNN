# -*- coding: utf-8 -*-
""" Siamese CNN+LSTM model in tensorflow
    Author : Elvys LINHARES PONTES
    Version: 0.5"""

import tensorflow as tf

class SiameseLSTMCNN:
    def __init__(self, sequence_embedding, forget_bias, learning_rate, number_layers, max_length, word_emb_size, local_context_size, dropout):
        self.sentence_emb_size  = sequence_embedding
        self.forget_bias        = forget_bias
        self.learning_rate      = learning_rate
        self.number_of_layers   = number_layers
        self.word_emb_size      = word_emb_size
        self.sentence_length    = max_length
        self.local_context_size = local_context_size
        self.dropout            = dropout

        # Create model
        self.placeholders()
        self.similarity()
        self.loss_optimizer()

        # Initialize variables
        self.initialize_variables = tf.group(tf.global_variables_initializer(), tf.variables_initializer(tf.local_variables()))
        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()

    def placeholders(self):
        self.x1         = tf.placeholder(tf.float32, shape=[None, self.sentence_length, self.word_emb_size],name="x1")
        self.x2         = tf.placeholder(tf.float32, shape=[None, self.sentence_length, self.word_emb_size],name="x2")
        self.len1       = tf.placeholder(tf.int32, shape=[None],name="len1")
        self.len2       = tf.placeholder(tf.int32, shape=[None],name="len2")
        self.y          = tf.placeholder(tf.float32, shape=[None], name="y")
        self.batch_size = tf.placeholder(tf.int64, name="batch_size")
        dataset = tf.data.Dataset.from_tensor_slices((self.x1, self.len1, self.x2, self.len2, self.y)).batch(self.batch_size).repeat()

        self.iter = dataset.make_initializable_iterator()
        self.iter_x1, self.iter_len1, self.iter_x2, self.iter_len2, self.iter_y = self.iter.get_next()

    def similarity(self):
        def sentence_similarity(dropout):
            with tf.variable_scope("local_context", reuse=tf.AUTO_REUSE):
                def conv2d(x, W, b, strides=1):
                    # Conv2D wrapper, with bias and tanh activation
                    x = tf.nn.conv2d(x, W, strides=[1, self.word_emb_size, 1, 1], padding='SAME')
                    x = tf.nn.bias_add(x, b)
                    return tf.nn.tanh(x)

                def get_attention(sentence, weights, biases):
                    # Reshape
                    x = tf.reshape(sentence, shape=[-1, self.sentence_length * self.word_emb_size])
                    x = tf.nn.dropout(x, keep_prob=dropout)
                    # Reshape
                    x = tf.reshape(x, shape=[-1, self.sentence_length * self.word_emb_size, 1, 1])
                    # Layer 1
                    conv1 = conv2d(x, weights['wc1'], biases['bc1'])

                    return tf.reshape(conv1, shape=[-1, self.sentence_length, self.word_emb_size])

                # Store layers weight & bias
                weights = {
                    'wc1':  tf.get_variable('wc1',shape=[self.local_context_size * self.word_emb_size, 1, 1, self.word_emb_size], initializer=tf.contrib.layers.xavier_initializer()),
                }

                biases = {
                    'bc1':  tf.get_variable('bc1',shape=[self.word_emb_size], initializer=tf.contrib.layers.xavier_initializer()),
                }

                self.local_context_x1 = get_attention(self.iter_x1, weights, biases)
                self.local_context_x2 = get_attention(self.iter_x2, weights, biases)

            with tf.variable_scope("siamese_lstm", reuse=tf.AUTO_REUSE):
                def extract_axis_1(data, ind):
                    """
                    Get specified elements along the first axis of tensor.
                    :param data: Tensorflow tensor that will be subsetted.
                    :param ind: Indices to take (one for each element along axis 0 of data).
                    :return: Subsetted tensor.
                    """

                    batch_range = tf.range(tf.shape(data)[0])
                    indices = tf.stack([batch_range, ind], axis=1)
                    res = tf.gather_nd(data, indices)

                    return res

                def lstm(sentence, lstm_cell, lstm_cell_b, seqlen):
                    outputs, state = \
                        tf.nn.dynamic_rnn(
                            lstm_cell, sentence, sequence_length=seqlen, dtype=tf.float32)
                    #outputs = tf.concat(outputs,2)
                    return extract_axis_1(outputs, seqlen - 1)

                lstm_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(
                            [tf.contrib.rnn.BasicLSTMCell(self.sentence_emb_size, forget_bias=self.forget_bias)
                                for _ in range(self.number_of_layers)]), state_keep_prob=dropout, variational_recurrent=True, dtype=tf.float32, input_size=self.word_emb_size)
                lstm_cell_b = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(
                            [tf.contrib.rnn.BasicLSTMCell(self.sentence_emb_size, forget_bias=self.forget_bias)
                                for _ in range(self.number_of_layers)]), state_keep_prob=dropout, variational_recurrent=True, dtype=tf.float32, input_size=self.word_emb_size)

                # Sentence x1
                x1 = tf.concat([self.iter_x1, self.local_context_x1], axis=2)
                lstm_outputs_1 = lstm(x1, lstm_cell , lstm_cell_b, self.iter_len1)
                self.sentence_embedding_1 = lstm_outputs_1
                # Sentence x2
                x2 = tf.concat([self.iter_x2, self.local_context_x2], axis=2)
                lstm_outputs_2 = lstm(x2, lstm_cell , lstm_cell_b, self.iter_len2)
                self.sentence_embedding_2 = lstm_outputs_2
                # Calculate the similarity of pairs of sentences
                dif             = tf.norm( tf.subtract(self.sentence_embedding_1, self.sentence_embedding_2), ord=1, axis=1 )
                sim             = tf.exp(-dif)

                return sim

        sim                     = sentence_similarity(dropout=self.dropout)
        sim_test                = sentence_similarity(dropout=1.0)
        self.prediction         = tf.clip_by_value(sim, 1e-7, 1.0-1e-7)
        self.prediction_test    = tf.clip_by_value(sim_test, 1e-7, 1.0-1e-7)
        self.reference          = tf.clip_by_value((self.iter_y-1.0)/4.0, 1e-7, 1.0-1e-7)

    def loss_optimizer(self):
        self.loss       = tf.reduce_sum( tf.square( tf.subtract(self.prediction, self.reference) ) )
        self.loss_test  = tf.reduce_sum( tf.square( tf.subtract(self.prediction_test, self.reference) ) )

        # Create an optimizer.
        opt = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
        # Compute the gradients for a list of variables.
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 5)
        self.train_op = opt.apply_gradients(zip(grads, tvars))
