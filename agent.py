"""
### NOTICE ###

You need to upload this file.
You can add any function you want in this file.

"""

import os
import random
import numpy as np
import tensorflow as tf
import cv2

from functools import reduce


def clipped_error(x):
    # Huber loss
    try:
        return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
    except:
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

def conv2d(x, 
    output_dim,
    kernel_size,
    stride,
    initializer=tf.contrib.layers.xavier_initializer(),
    activation_fn=tf.nn.relu,
    data_format='NHWC',
    padding='VALID',
    name='conv2d'):
    with tf.variable_scope(name):
        if data_format == 'NCHW':
          stride = [1, 1, stride[0], stride[1]]
          kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[1], output_dim]
        elif data_format == 'NHWC':
          stride = [1, stride[0], stride[1], 1]
          kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]

        w = tf.get_variable('w', kernel_shape, tf.float32, initializer=initializer)
        conv = tf.nn.conv2d(x, w, stride, padding, data_format=data_format)

        b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, b, data_format)

    if activation_fn != None:
        out = activation_fn(out)

    return out, w, b

def linear(input_, output_size, stddev=0.02, bias_start=0.0, activation_fn=None, name='linear'):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        w = tf.get_variable('Matrix', [shape[1], output_size], tf.float32,
            tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('bias', [output_size],
            initializer=tf.constant_initializer(bias_start))

        out = tf.nn.bias_add(tf.matmul(input_, w), b)

        if activation_fn != None:
            return activation_fn(out), w, b
        else:
            return out, w, b

class History:
    def __init__(self):

        self.cnn_format = 'NCHW'
        self.batch_size = 32
        self.history_length = 4
        self.screen_height = 84
        self.screen_width = 84

        self.history = np.zeros(
            [self.history_length, self.screen_height, self.screen_width], dtype=np.float32)

    def add(self, screen):
        self.history[:-1] = self.history[1:]
        self.history[-1] = screen

    def reset(self):
        self.history *= 0

    def get(self):
        if self.cnn_format == 'NHWC':
          return np.transpose(self.history, (1, 2, 0))
        else:
          return self.history

class Agent(object):
    def __init__(self, sess, min_action_set):
        self.sess = sess

        self.min_action_set = min_action_set

        scale = 10000
        self.max_step = 2000 * scale
        self.memory_size = 100 * scale

        self.batch_size = 32
        self.random_start = 30
        self.cnn_format = 'NCHW'
        self.discount = 0.99
        self.target_q_update_step = 1 * scale
        self.learning_rate = 0.00025
        self.learning_rate_minimum = 0.00025
        self.learning_rate_decay = 1.0
        self.learning_rate_decay_step = 5 * scale

        self.ep_end = 0.1
        self.ep_start = 1.
        self.ep_end_t = self.memory_size

        self.history_length = 4
        self.train_frequency = 4
        self.learn_start = 5 * scale

        self.min_delta = -1
        self.max_delta = 1

        self.double_q = False
        self.dueling = False

        self.screen_width  = 84
        self.screen_height = 84
        self.max_reward = 1.
        self.min_reward = -1.

        self.action_size = 4
        self.action_repeat = 4
        self.is_training = True
        self.dims = (self.screen_width, self.screen_height)

        self.env_name = 'Breakout-v0'
        self.env_type = 'detail'

        self.model_dir = 'checkpoints/'
        self.checkpoint_dir = self.model_dir #os.path.join('checkpoints', self.model_dir)
        self.weight_dir = 'weights'


        self.history = History()
        self.test_history = History()

        with tf.variable_scope('step'):
          self.step_op = tf.Variable(0, trainable=False, name='step')
          self.step_input = tf.placeholder('int32', None, name='step_input')
          self.step_assign_op = self.step_op.assign(self.step_input)

        self.build_dqn()

    def getSetting(self):
        """
        # TODO
            You can only modify these three parameters.
            Adding any other parameters are not allowed.
            1. action_repeat: number of time for repeating the same action 
            2. random_init_step: number of randomly initial steps
            3. screen_type: return 0 for RGB; return 1 for GrayScale
        """
        action_repeat = self.action_repeat
        screen_type = 0
        return action_repeat, screen_type

    def predict(self, s_t, test_ep=None):
        ep = test_ep 

        if random.random() < ep:
          action = random.randrange(self.action_size)
        else:
          action = self.q_action.eval({self.s_t: [s_t]})[0]

        return action

    def update_target_q_network(self):
        for name in self.w.keys():
            self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})

    def load_model(self):
        print(" [*] Loading checkpoints...")
        self.saver.restore(self.sess, 'best_model.ckpt')
        print(" [*] Load SUCCESS")

    def build_dqn(self):
        """
        # TODO
            You need to build your DQN here.
            And load the pre-trained model named as './best_model.ckpt'.
            For example, 
                saver.restore(self.sess, './best_model.ckpt')
        """
        self.w = {}
        self.t_w = {}

        #initializer = tf.contrib.layers.xavier_initializer()
        initializer = tf.truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu

        # training network
        with tf.variable_scope('prediction'):
          if self.cnn_format == 'NHWC':
            self.s_t = tf.placeholder('float32',
                [None, self.screen_height, self.screen_width, self.history_length], name='s_t')
          else:
            self.s_t = tf.placeholder('float32',
                [None, self.history_length, self.screen_height, self.screen_width], name='s_t')

          self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.s_t,
              32, [8, 8], [4, 4], initializer, activation_fn, self.cnn_format, name='l1')
          self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1,
              64, [4, 4], [2, 2], initializer, activation_fn, self.cnn_format, name='l2')
          self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2,
              64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='l3')

          shape = self.l3.get_shape().as_list()
          self.l3_flat = tf.reshape(self.l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

          if self.dueling:
            self.value_hid, self.w['l4_val_w'], self.w['l4_val_b'] = \
                linear(self.l3_flat, 512, activation_fn=activation_fn, name='value_hid')

            self.adv_hid, self.w['l4_adv_w'], self.w['l4_adv_b'] = \
                linear(self.l3_flat, 512, activation_fn=activation_fn, name='adv_hid')

            self.value, self.w['val_w_out'], self.w['val_w_b'] = \
              linear(self.value_hid, 1, name='value_out')

            self.advantage, self.w['adv_w_out'], self.w['adv_w_b'] = \
              linear(self.adv_hid, self.action_size, name='adv_out')

            # Average Dueling
            self.q = self.value + (self.advantage - 
              tf.reduce_mean(self.advantage, reduction_indices=1, keep_dims=True))
          else:
            self.l4, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, 512, activation_fn=activation_fn, name='l4')
            self.q, self.w['q_w'], self.w['q_b'] = linear(self.l4, self.action_size, name='q')

          self.q_action = tf.argmax(self.q, dimension=1)

          q_summary = []
          avg_q = tf.reduce_mean(self.q, 0)
          for idx in range(self.action_size):
            q_summary.append(tf.histogram_summary('q/%s' % idx, avg_q[idx]))
          self.q_summary = tf.merge_summary(q_summary, 'q_summary')

        # target network
        with tf.variable_scope('target'):
          if self.cnn_format == 'NHWC':
            self.target_s_t = tf.placeholder('float32', 
                [None, self.screen_height, self.screen_width, self.history_length], name='target_s_t')
          else:
            self.target_s_t = tf.placeholder('float32', 
                [None, self.history_length, self.screen_height, self.screen_width], name='target_s_t')

          self.target_l1, self.t_w['l1_w'], self.t_w['l1_b'] = conv2d(self.target_s_t, 
              32, [8, 8], [4, 4], initializer, activation_fn, self.cnn_format, name='target_l1')
          self.target_l2, self.t_w['l2_w'], self.t_w['l2_b'] = conv2d(self.target_l1,
              64, [4, 4], [2, 2], initializer, activation_fn, self.cnn_format, name='target_l2')
          self.target_l3, self.t_w['l3_w'], self.t_w['l3_b'] = conv2d(self.target_l2,
              64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='target_l3')

          shape = self.target_l3.get_shape().as_list()
          self.target_l3_flat = tf.reshape(self.target_l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

          if self.dueling:
            self.t_value_hid, self.t_w['l4_val_w'], self.t_w['l4_val_b'] = \
                linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_value_hid')

            self.t_adv_hid, self.t_w['l4_adv_w'], self.t_w['l4_adv_b'] = \
                linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_adv_hid')

            self.t_value, self.t_w['val_w_out'], self.t_w['val_w_b'] = \
              linear(self.t_value_hid, 1, name='target_value_out')

            self.t_advantage, self.t_w['adv_w_out'], self.t_w['adv_w_b'] = \
              linear(self.t_adv_hid, self.action_size, name='target_adv_out')

            # Average Dueling
            self.target_q = self.t_value + (self.t_advantage - 
              tf.reduce_mean(self.t_advantage, reduction_indices=1, keep_dims=True))
          else:
            self.target_l4, self.t_w['l4_w'], self.t_w['l4_b'] = \
                linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_l4')
            self.target_q, self.t_w['q_w'], self.t_w['q_b'] = \
                linear(self.target_l4, self.action_size, name='target_q')

          self.target_q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
          self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)

        with tf.variable_scope('pred_to_target'):
          self.t_w_input = {}
          self.t_w_assign_op = {}

          for name in self.w.keys():
            self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
            self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

        # optimizer
        with tf.variable_scope('optimizer'):
          self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
          self.action = tf.placeholder('int64', [None], name='action')

          action_one_hot = tf.one_hot(self.action, self.action_size, 1.0, 0.0, name='action_one_hot')
          q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')

          self.delta = self.target_q_t - q_acted

          self.global_step = tf.Variable(0, trainable=False)

          self.loss = tf.reduce_mean(clipped_error(self.delta), name='loss')
          self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
          self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
              tf.train.exponential_decay(
                  self.learning_rate,
                  self.learning_rate_step,
                  self.learning_rate_decay_step,
                  self.learning_rate_decay,
                  staircase=True))
          self.optim = tf.train.RMSPropOptimizer(
              self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)

        with tf.variable_scope('summary'):
          scalar_summary_tags = ['average.reward', 'average.loss', 'average.q', \
              'episode.max reward', 'episode.min reward', 'episode.avg reward', 'episode.num of game', 'training.learning_rate']

          self.summary_placeholders = {}
          self.summary_ops = {}

          for tag in scalar_summary_tags:
            self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
            self.summary_ops[tag]  = tf.scalar_summary("%s-%s/%s" % (self.env_name, self.env_type, tag), self.summary_placeholders[tag])

          histogram_summary_tags = ['episode.rewards', 'episode.actions']

          for tag in histogram_summary_tags:
            self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
            self.summary_ops[tag]  = tf.histogram_summary(tag, self.summary_placeholders[tag])

          self.writer = tf.train.SummaryWriter('./logs/%s' % self.model_dir, self.sess.graph)

        tf.initialize_all_variables().run()

        self.saver = tf.train.Saver(list(self.w.values()) + [self.step_op], max_to_keep=30)

        self.load_model()
        self.update_target_q_network()

    
    def play(self, screen, test_ep=None, render=False):
        if test_ep == None:
            test_ep = 0.05 #self.ep_end

        resized_screen = cv2.resize(cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)/255., self.dims)
        self.test_history.add(resized_screen)
        action = self.predict(self.test_history.get(), test_ep)
        #print ('action: ' + str(action))
        #cv2.imshow("123", resized_screen)
        #cv2.waitKey(0)
        return self.min_action_set[action]

"""
    def play(self, screen):
        ""
        # TODO
            The "action" is your DQN argmax ouput.
            The "min_action_set" is used to transform DQN argmax ouput into real action number.
            For example,
                 DQN output = [0.1, 0.2, 0.1, 0.6]
                 argmax = 3
                 min_action_set = [0, 1, 3, 4]
                 real action number = 4
        ""
        action = 0 # you can remove this line

        return self.min_action_set[action]
"""
