import time
import os
import random
from abc import ABCMeta, abstractmethod
import numpy as np
import tensorflow as tf

from state import State
from environment import Environment
from agents import HumanAgent, RandomAgent, TDAgent
from utils import extract_features


class BaseModel(metaclass=abc.ABCMeta):
    def __init__(self, sess, model_path, summary_path, checkpoint_path, hidden_sizes=None, restore_flag=False):
        self.sess = sess
        self.model_path = model_path
        self.summary_path = summary_path
        self.checkpoint_path = checkpoint_path
        self.hidden_sizes = hidden_sizes
        self.restore_flag = restore_flag

        self.build_graph()

    @abstractmethod
    def build_graph(self):
        pass

    def restore(self):
        latest_checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_path)
        self.saver.restore(self.sess, latest_checkpoint_path)

    def get_output(self, x):
        return self.sess.run(self.V, feed_dict={self.x: x, self.keep_prob: 1.0})

    def play(self):
        agents = [TDAgent(-1, self), HumanAgent(1)]
        env = Environment(agents, verbose=True)
        env.play()

    def test(self, n_episodes=100):
        td_wins = 0
        random_wins = 0
        for i in range(n_episodes):
            sgn = random.choice([-1, 1])
            agents = [TDAgent(sgn, model=self), RandomAgent(-sgn)]
            env = Environment(agents)
            env.play()

            if env.winner == sgn:
                td_wins += 1
            else:
                random_wins += 1

            print('Episode: {}, TD-Agent: {}, RandomAgent: {}'.format(i, td_wins, random_wins))

    def train(self, n_episodes=10000, val_period=1000, n_val=100):
        tf.train.write_graph(self.sess.graph_def, self.model_path, 'td_gammon.pb', as_text=False)
        summary_writer = tf.summary.FileWriter(logdir=self.summary_path,
                                               graph=self.sess.graph,
                                               filename_suffix=str(time.time()))

        agents = [TDAgent(-1, model=self), TDAgent(1, model=self)]
        for episode in range(n_episodes):
            if episode != 0 and episode % val_period == 0:
                self.test(n_episodes=n_val)

            state = State()
            i = random.randint(0, 1)
            agent = agents[i]
            state.sign = agent.sign
            x = extract_features(state)
            step = 0
            while not state.winner:
                move = agent.choose_move(state)
                state.update(move)
                i = (i + 1) % 2
                agent = agents[i]
                x_next = extract_features(state)
                V_next = self.get_output(x_next)
                feed_dict = {self.x: x, self.V_next: V_next}
                self.sess.run(self.train_op, feed_dict=feed_dict)
                x = x_next
                step += 1

            z = max(0, state.winner)

            _, global_step, summaries, _ = self.sess.run(
                [self.train_op,
                 self.global_step,
                 self.summaries_op,
                 self.reset_op],
                feed_dict={self.x: x,
                           self.V_next: np.array([[z]], dtype='float'),
                           self.keep_prob: keep_prob}
            )

            summary_writer.add_summary(summaries, global_step=global_step)
            print(f'Game: {episode} Winner: {z} in {step} turns')
            self.saver.save(self.sess, os.path.join(self.checkpoint_path, 'checkpoint'), global_step=global_step)

        summary_writer.close()
        self.test(n_episodes=1000)


class ModelTD(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    def build_graph(self):
        input_shape = TDAgent().feature_space_dim()
        self.x = tf.placeholder(shape=[1, input_shape], dtype=tf.float32)
        self.V_next = tf.placeholder(dtype=tf.float32)
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False)

        lambda_decay = tf.train.exponential_decay(0.9, self.global_step, 30000, 0.96, True)
        lamda = tf.maximum(0.1, lambda_decay, name='lambda')

        alpha_decay = tf.train.exponential_decay(0.1, self.global_step, 40000, 0.96, True)
        alpha = tf.maximum(0.01, alpha_decay, name='alpha')

        gamma = 0.99
        keep_prob = 0.2

        self.hidden_sizes = self.hidden_sizes if self.hidden_sizes else [80]
        h = hidden_sizes.pop(0)
        x = tf.layers.dense(self.x, h, tf.nn.relu)
        for h in hidden_sizes:
            x = tf.layers.dense(x, h, tf.nn.relu)
            x = tf.nn.dropout(x, keep_prob)

        self.V = tf.layers.dense(x, 1, tf.nn.sigmoid)

        delta = tf.reduce_mean(self.V_next - self.V)
        loss = tf.square(delta)

        with tf.variable_scope('game'):
            game_step = tf.Variable(1., name='game_step', trainable=False)
            game_step_op = game_step.assign_add(1)

            loss_sum = tf.Variable(0., name='loss_sum', trainable=False)
            loss_avg_ema = tf.train.ExponentialMovingAverage(decay=0.999)
            loss_sum_op = loss_sum.assign_add(loss)
            loss_avg_op = loss_sum / tf.maximum(game_step, 1.)

            loss_avg_ema_op = loss_avg_ema.apply([loss_avg_op])
            tf.summary.scalar('game/loss_avg_ema', loss_avg_ema.average(loss_avg_op))

            game_step_reset_op = game_step.assign(0.)
            loss_sum_reset_op = loss_sum.assign(0.)
            self.reset_op = tf.group(*[loss_sum_reset_op, game_step_reset_op])

        global_step_op = self.global_step.assign_add(1)

        tvars = tf.trainable_variables()
        grads = tf.gradients(self.V, tvars)

        # https://arxiv.org/pdf/1512.04087.pdf, algorithm 1
        apply_gradients = []
        with tf.variable_scope('apply_gradients'):
            for grad, var in zip(grads, tvars):
                with tf.variable_scope('trace'):
                    trace_var = tf.Variable(tf.zeros(grad.get_shape()), trainable=False, name='trace')
                    trace = trace_var.assign(gamma * lamda * trace_var + grad)
                grad_apply = var.assign_add(alpha * delta * trace)
                apply_gradients.append(grad_apply)

        with tf.control_dependencies([global_step_op, game_step_op, loss_sum_op, loss_avg_ema_op]):
            self.train_op = tf.group(*apply_gradients, name='train')

        self.summaries_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=1)
        self.sess.run(tf.global_variables_initializer())

        if self.restore_flag:
            self.restore()


class ModelTDOnline(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    def build_graph(self):
        input_shape = TDAgent().feature_space_dim()
        self.x = tf.placeholder(shape=[1, input_shape], dtype=tf.float32)
        self.V_next = tf.placeholder(dtype=tf.float32)
        self.V_old = tf.placeholder(dtype=tf.float32)
        self.V_trace = tf.placeholder(dtype=tf.float32)
        self.flag = tf.placeholder(dtype=tf.bool)

        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False)

        lambda_decay = tf.train.exponential_decay(0.9, self.global_step, 30000, 0.96, True)
        lamda = tf.maximum(0.1, lambda_decay, name='lambda')

        alpha_decay = tf.train.exponential_decay(0.1, self.global_step, 30000, 0.96, True)
        alpha = tf.maximum(0.01, alpha_decay, name='alpha')

        keep_prob = 0.2

        self.hidden_sizes = self.hidden_sizes if self.hidden_sizes else [80]
        h = hidden_sizes.pop(0)
        x = tf.layers.dense(self.x, h, tf.nn.relu)
        for h in hidden_sizes:
            x = tf.layers.dense(x, h, tf.nn.relu)
            x = tf.nn.dropout(x, keep_prob)

        self.V = tf.layers.dense(x, 1, tf.nn.sigmoid)

        delta = tf.reduce_mean(self.V_next - self.V)
        global_step_op = self.global_step.assign_add(1)

        tvars = tf.trainable_variables()
        grads = tf.gradients(self.V, tvars)

        apply_gradients = []
        apply_traces = []
        # https://arxiv.org/pdf/1512.04087.pdf, algorithm 2
        delta2 = self.V - self.V_old
        for grad, var in zip(grads, tvars):
            trace = tf.Variable(tf.zeros(grad.get_shape()), trainable=False, name='trace')
            trace_op = trace.assign(lamda * trace + grad - alpha * lamda * self.V_trace * grad)
            grad_apply = var.assign_add(alpha * (delta + delta2) * trace_op - alpha * delta2 * grad)
            apply_gradients.append(grad_apply)
            trace_apply = var.assign(trace_op)
            apply_traces.append(trace_apply)

        with tf.control_dependencies([global_step_op]):
            self.train_op = tf.group(*apply_gradients, name='train')
            self.assign_trace = tf.group(*apply_traces)

        self.summaries_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=1)
        self.sess.run(tf.global_variables_initializer())

        if self.restore_flag:
            self.restore()
