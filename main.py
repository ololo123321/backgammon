import os
import tensorflow as tf
from argparse import ArgumentParser
from src.models import ModelTD as Model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_dir')
    parser.add_argument('--summaries_dir')
    parser.add_argument('--checkpoints_dir')
    parser.add_argument('--n_episodes', type=int, default=10000)
    parser.add_argument('--val_period', type=int, default=500)
    parser.add_argument('--n_val', type=int, default=100)
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--play', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    print(args)

    with tf.Session() as sess:
        model = Model(
            sess=sess,
            model_path=args.model_dir,
            summary_path=args.summaries_dir,
            checkpoint_path=args.checkpoints_dir,
            hidden_sizes=None,
            restore_flag=args.restore
        )
        if args.test:
            model.test(n_episodes=1000)
        elif args.play:
            model.play()
        else:
            if not os.path.exists(args.model_dir):
                os.makedirs(args.model_dir)
            if not os.path.exists(args.summaries_dir):
                os.makedirs(args.summaries_dir)
            if not os.path.exists(args.checkpoints_dir):
                os.makedirs(args.checkpoints_dir)

            model.train(n_episodes=args.n_episodes, val_period=args.val_period, n_val=args.n_val)
