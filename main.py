import os
import json
import tensorflow as tf
from argparse import ArgumentParser
from src.models import ModelTD as Model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_dir')
    parser.add_argument('--hidden_dims', type=str, default='80')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--num_games_training', type=int, default=10000)
    parser.add_argument('--val_period', type=int, default=1000)
    parser.add_argument('--num_games_test', type=int, default=100)
    parser.add_argument('--save_period', type=int, default=1000)
    parser.add_argument('--max_to_keep', type=int, default=3)
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--play', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    print(args)

    os.makedirs(args.model_dir, exist_ok=True)

    if args.test or args.play:
        filename = None
    else:
        filename = os.path.join(args.model_dir, 'train.log')

    print("filename:", filename)

    hidden_dims = list(map(int, args.hidden_dims.split(',')))
    config = {
        "model": {
            "hidden_dims": hidden_dims,
            "dropout": args.dropout
        },
        "training": {
            "model_dir": args.model_dir,
            "num_games": args.num_games_training,
            "val_period": args.val_period,
            "save_period": args.save_period,
            "max_to_keep": args.max_to_keep
        },
        "validation": {
            "num_games": args.num_games_test
        }
    }
    print("config:", config)

    with open(os.path.join(args.model_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    sess = tf.Session()

    model = Model(sess=sess, config=config)
    model.build()

    if args.restore:
        model.restore()

    if args.test:
        model.test(n_episodes=args.num_games_test)
    elif args.play:
        model.play()
    else:
        model.train()
