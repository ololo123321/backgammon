"""
Encoders of state. Take as input vectorized representation of states, and return approximation of value functions
"""

import tensorflow as tf


class TesauroEncoder(tf.keras.layers.Layer):
    """
    https://bkgm.com/articles/tesauro/tdl.html
    """
    def __init__(self, hidden_dim=80, activation=tf.nn.relu):
        super().__init__()
        self.dense = tf.keras.layers.Dense(hidden_dim, activation=activation)

    def call(self, x, **kwargs):
        x = self.dense(x)
        return x
