import numpy as np
import tensorflow as tf


def dense_layer(x, shape, activation, name, keep_prob=1):
    with tf.variable_scope(name):
        W = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
        b = tf.Variable(tf.constant(0.1, shape=shape[-1:]), name='bias')
        return tf.nn.dropout(activation(tf.matmul(x, W) + b, name='activation'), keep_prob)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_reward(x, weights=None):
    if weights:
        v1 = sigmoid(np.dot(weights['W1'].T, x.T) + weights['b1'].reshape(-1, 1))
        return sigmoid(np.dot(weights['W2'].T, v1) + weights['b2'])[0][0]
    return 0.5


def extract_features(state):
    """
    Такое простое признаковое описание пока даёт лучшие результаты
    """
    board_black = state.board if state.sign == 1 else [-x for x in state.board[::-1]]
    bar_black = state.bar if state.sign == 1 else {-k: v for k, v in state.bar.items()}
    features = []
    for sgn in [1, -1]:
        on_board = 0
        on_bar = bar_black[sgn]
        for p in board_black:
            v = abs(p) if np.sign(p) == sgn else 0
            on_board += v
            features.append(v)
        features.append(on_bar)
        features.append(15 - on_board - on_bar)
    features.append(int(state.sign == 1))
    return np.array(features).reshape(1, -1)


def choose_move_trained(state, weights=None):
    """
    Модель предсказывает вероятность выигрыша игрока +1.
    Соответственно, нужно получить вероятность противоположного события, если данный игрок -1.

    У аргумента должен быть метод get_moves() и атрибут sign
    """
    moves = state.get_moves()
    v_best = 0
    move_best = moves[0]
    for move in moves:
        state_new = state.clone()
        state_new.board, state_new.bar = move
        x = extract_features(state_new)
        v = get_reward(x, weights)
        if state_new.sign == -1:
            v = 1 - v
        if v > v_best:
            v_best = v
            move_best = move
    return move_best
