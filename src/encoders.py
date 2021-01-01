"""
Encoders of states. Take as input vectorized representation of states, and return new vectorized representation
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


class MLPEncoder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.dense_1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.dropout_1 = tf.keras.layers.Dropout(0.2)
        self.dense_2 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.dropout_2 = tf.keras.layers.Dropout(0.2)

    def call(self, x, training=None):
        x = self.dense_1(x)
        x = self.dropout_1(x, training=training)
        x = self.dense_2(x)
        x = self.dropout_2(x, training=training)
        return x
