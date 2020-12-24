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
    parser.add_argument('--n_episodes', type=int, default=10000)
    parser.add_argument('--val_period', type=int, val_period=500)
    parser.add_argument('--n_val', type=int, default=100)
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--play', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    print(args)

    with tf.Session() as sess:
        model = Model(
            sess=sess,
            model_path=model_path,
            summary_path=summary_path,
            checkpoint_path=checkpoint_path,
            hidden_sizes=None,
            restore_flag=args.restore
        )
        print("keep_prob:", model.keep_prob)
        if args.test:
            model.test(n_episodes=1000)
        elif args.play:
            model.play()
        else:
            model.train(n_episodes=args.n_episodes, val_period=args.val_period, n_val=args.n_val)
