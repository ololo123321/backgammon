import os
import random
import logging
import time
import json
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

# from src.state import State
from src.state_pyx.state import State
from src.environment import Environment
from src.agents import HumanAgent, RandomAgent, TDAgent
from src import encoders


class LoggerMixin:
    def __init__(self, filename=None):
        logger_name = self.__class__.__name__
        self.logger = logging.getLogger(logger_name)
        formatter = logging.Formatter(
            fmt='%(asctime)s.%(msecs)03d %(name)-8s %(levelname)-8s %(message)s',
            datefmt='%y-%m-%d %H:%M:%S'
        )

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        self.logger.addHandler(console)

        if filename is not None:
            file_handler = logging.FileHandler(filename=filename)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        self.logger.setLevel(logging.DEBUG)


class BaseModel(ABC, LoggerMixin):
    def __init__(self, sess, config):
        """
        config = {
            "model": {
                "encoder": "TesauroEncoder",
                "params": {}
            },
            "training": {
                "model_dir": "/tmp/backgammon_agent",
                "num_games": 100000,
                "val_period": 1000,
                "save_period": 1000,
                "max_to_keep": 10
            },
            "validation": {
                "num_games": 100
            }
        }
        """
        filename = None
        if config is not None:
            model_dir = config.get("training", {}).get("model_dir")
            if model_dir is not None:
                filename = os.path.join(model_dir, "train.log")
        super().__init__(filename=filename)

        self.sess = sess
        self.config = config

        # inputs
        self.state_ph = None
        self.training_ph = None

        # tensors
        self.V = None
        self.V_next = None
        self.global_step = None
        self.loss = None

        # ops
        self.train_op = None
        self.reset_op = None
        self.summaries_op = None

    @abstractmethod
    def build(self):
        pass

    def train(self):
        summary_writer = tf.summary.FileWriter(logdir=self.config['training']['model_dir'])
        saver = tf.train.Saver(max_to_keep=self.config['training']['max_to_keep'])
        checkpoint_path = os.path.join(self.config['training']['model_dir'], 'model.ckpt')

        agents = [TDAgent(-1, model=self), TDAgent(1, model=self)]

        for episode in range(self.config['training']['num_games']):
            # print(f"episode {episode} starts")

            state = State()
            i = random.randint(0, 1)
            agent = agents[i]
            state.sign = agent.sign
            x = state.features
            step = 0
            t0 = time.time()
            while not state.is_game_over:
                # 1. Выбрать лучшее состояние из возможных
                assert state.sign == agent.sign
                # print('=' * 10, f"step {step} starts", '=' * 10)
                # print("current agent:", agent.sign)
                # print("current state:", state)
                state = agent.ply(state)  # выбрали наилучшее следующее состояние от лица текущего игрока
                # print("chosen state:", state)

                # 2. Развернуть состояние к противнику
                state = state.reversed  # развернули состояние к другому игроку
                # print("reversed state:", state)

                # 3. Сменить игрока
                i = (i + 1) % 2
                agent = agents[i]  # сменили игрока

                # 4. Получить вектор признакового описания нового состояния
                x_next = state.features

                # 5. Посчитать награду за переход на новое состояние
                v_next = self.get_output(x_next)

                # 6. Сделать шаг обучения
                feed_dict = {self.state_ph: x, self.V_next: v_next, self.training_ph: True}
                self.sess.run(self.train_op, feed_dict=feed_dict)

                x = x_next
                step += 1

            # Получить истинную вероятность победы игрока +1 (1, если он победил, 0 - иначе)
            z = max(0, state.winner)
            v_next = np.array([[z]], dtype=np.float32)

            ops = [self.train_op, self.global_step, self.summaries_op, self.reset_op]
            feed_dict = {self.state_ph: x, self.V_next: v_next, self.training_ph: True}
            _, global_step, summaries, _ = self.sess.run(ops, feed_dict=feed_dict)

            summary_writer.add_summary(summaries, global_step=global_step)
            t_game = round(time.time() - t0, 4)
            self.logger.debug(f'game: {episode}; winner: {z}; num turns: {step}; time elapsed: {t_game} sec.')

            if episode % self.config['training']['val_period'] == 0:
                self.logger.debug("evaluation with random agent starts")
                self.test(n_episodes=self.config['validation']['num_games'])

            if episode % self.config['training']['save_period'] == 0:
                self.logger.debug(f"saving model to {checkpoint_path}")
                saver.save(sess=self.sess, save_path=checkpoint_path, global_step=global_step)

        summary_writer.close()
        self.test(n_episodes=1000)

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

            self.logger.debug(f'Episode: {i}, TD-Agent: {td_wins}, RandomAgent: {random_wins}')

    def play(self):
        agents = [TDAgent(-1, self), HumanAgent(1)]
        env = Environment(agents, verbose=True)
        env.play()

    def restore(self, model_dir: str = None):
        # подгрузка конфига
        self.config = json.load(open(os.path.join(model_dir, 'config.json')))

        # построение вычислительного графа
        self.build()

        # подгрузка весов
        saver = tf.train.Saver()
        checkpoint_path = tf.train.latest_checkpoint(model_dir)
        self.logger.debug(f"restoring model from {checkpoint_path}")
        saver.restore(self.sess, checkpoint_path)

    def get_output(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run(self.V, feed_dict={self.state_ph: x, self.training_ph: False})

    def export_inference_graph(self, export_dir):
        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

        inputs = {
            "state": tf.saved_model.utils.build_tensor_info(self.state_ph),
            "training": tf.saved_model.utils.build_tensor_info(self.training_ph)
        }

        outputs = {
            self.V.name: tf.saved_model.utils.build_tensor_info(self.V)
        }

        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=outputs,
            method_name=tf.saved_model.PREDICT_METHOD_NAME
        )

        builder.add_meta_graph_and_variables(
            sess=self.sess,
            tags=[tf.saved_model.SERVING],
            signature_def_map={
                tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
            },
        )

        builder.save()


class ModelTD(BaseModel):
    def __init__(self, sess, config):
        super().__init__(sess=sess, config=config)

        self.lamda = None
        self.alpha = None
        self.gamma = None

    def build(self):
        d = State().features_dim
        self.state_ph = tf.placeholder(shape=[None, d], dtype=tf.float32, name="state_ph")
        self.training_ph = tf.placeholder(dtype=tf.bool, shape=None, name="training_ph")
        self.V_next = tf.placeholder(dtype=tf.float32, shape=None, name="V")

        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False)

        lambda_decay = tf.train.exponential_decay(0.9, self.global_step, 30000, 0.96, True)  # TODO: вынести в параметры
        self.lamda = tf.maximum(0.1, lambda_decay, name='lambda')

        alpha_decay = tf.train.exponential_decay(0.1, self.global_step, 40000, 0.96, True)  # TODO: вынести в параметры
        self.alpha = tf.maximum(0.01, alpha_decay, name='alpha')

        self.gamma = tf.constant(0.99)  # TODO: вынести в параметры

        enc_cls = getattr(encoders, self.config['model']['encoder'])
        enc = enc_cls(**self.config['model']['params'])
        x = enc(self.state_ph, training=self.training_ph)
        self.V = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(x)

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

        apply_gradients = self._get_grad_ops(tvars=tvars, grads=grads, delta=delta)

        with tf.control_dependencies([global_step_op, game_step_op, loss_sum_op, loss_avg_ema_op]):
            self.train_op = tf.group(*apply_gradients, name='train')

        self.summaries_op = tf.summary.merge_all()
        self.sess.run(tf.global_variables_initializer())

    def _get_grad_ops(self, tvars, grads, delta):
        # https://arxiv.org/pdf/1512.04087.pdf, algorithm 1
        apply_gradients = []
        with tf.variable_scope('apply_gradients'):
            for grad, var in zip(grads, tvars):
                with tf.variable_scope('trace'):
                    trace_var = tf.Variable(tf.zeros(grad.get_shape()), trainable=False, name='trace')
                    trace = trace_var.assign(self.gamma * self.lamda * trace_var + grad)
                grad_apply = var.assign_add(self.alpha * delta * trace)
                apply_gradients.append(grad_apply)
        return apply_gradients


class ModelTDOnline(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.V_old = None
        self.V_trace = None
        self.flag = None
        self.assign_trace_op = None

    def build(self):
        # TODO: копипаста из ModelTD!
        d = State().features_dim
        self.state_ph = tf.placeholder(shape=[None, d], dtype=tf.float32)
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

        hidden_sizes = [80]
        h = hidden_sizes.pop(0)
        x = tf.layers.dense(self.state_ph, h, tf.nn.relu)
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
            self.assign_trace_op = tf.group(*apply_traces)

        self.summaries_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=1)
        self.sess.run(tf.global_variables_initializer())


if __name__ == "__main__":
    def check():
        sess = tf.Session()
        model_dir = "/tmp/backgammon"
        config = {
            "model": {
                "hidden_dims": [80, 40],
                "dropout": 0.1
            },
            "training": {
                "model_dir": model_dir,
                "num_games": 100,
                "val_period": 10,
                "save_period": 10,
                "max_to_keep": 1
            },
            "validation": {
                "num_games": 10
            }
        }
        os.system(f'rm -r {model_dir} && mkdir {model_dir}')
        model = ModelTD(sess=sess, config=config)
        model.build()
        model.train()

    check()
