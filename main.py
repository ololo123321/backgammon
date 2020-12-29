import os
import tensorflow as tf
from argparse import ArgumentParser
from src.models import ModelTD as Model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_dir')
    parser.add_argument('--n_episodes', type=int, default=10000)
    parser.add_argument('--val_period', type=int, default=1000)
    parser.add_argument('--n_val', type=int, default=100)
    parser.add_argument('--save_period', type=int, default=1000)
    parser.add_argument('--max_to_keep', type=int, default=3)
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--play', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    print(args)

    os.makedirs(args.model_dir, exist_ok=True)

    with tf.Session() as sess:
        model = Model(
            sess=sess,
            model_dir=args.model_dir,
            hidden_sizes=None,
            restore_flag=args.restore,
            max_to_keep=args.max_to_keep
        )
        if args.test:
            model.test(n_episodes=args.n_episodes)
        elif args.play:
            model.play()
        else:
            model.train(
                n_episodes=args.n_episodes,
                val_period=args.val_period,
                n_val=args.n_val,
                save_period=args.save_period
            )
