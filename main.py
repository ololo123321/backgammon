import os
import sys
import tensorflow as tf
from argparse import ArgumentParser
from models import ModelTD as Model

if not os.path.exists('models'):
    os.makedirs('models')

if not os.path.exists('summaries'):
    os.makedirs('summaries')

if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

model_path = os.path.join(sys.path[0], 'models')
summary_path = os.path.join(sys.path[0], 'summaries')
checkpoint_path = os.path.join(sys.path[0], 'checkpoints')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--n_episodes', type=int)
    parser.add_argument('--val_period', type=int)
    parser.add_argument('--n_val', type=int)
    parser.add_argument('--restore_flag', action='store_true')
    parser.add_argument('--play', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(n_episodes=10000, val_period=1000, n_val=100)
    args = parser.parse_args()
    with tf.Session() as sess:
        model = Model(sess, model_path, summary_path, checkpoint_path, None, args.restore)
        if args.test:
            model.test(n_episodes=1000)
        elif args.play:
            model.play()
        else:
            model.train(n_episodes=args.n_episodes, val_period=args.val_period, n_val=args.n_val)
