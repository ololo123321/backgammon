import os
import json
import tensorflow as tf
from argparse import ArgumentParser
from src.models import ModelTD as Model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_dir')
    parser.add_argument('--encoder', type=str, default='TesauroEncoder')
    parser.add_argument('--enc_params', type=str, default='{}')
    parser.add_argument('--num_games_training', type=int, default=10000, required=False)
    parser.add_argument('--val_period', type=int, default=1000, required=False)
    parser.add_argument('--num_games_test', type=int, default=100, required=False)
    parser.add_argument('--save_period', type=int, default=1000, required=False)
    parser.add_argument('--max_to_keep', type=int, default=3, required=False)
    parser.add_argument('--restore', action='store_true', required=False)
    args = parser.parse_args()
    print(args)

    sess = tf.Session()

    if args.restore:
        print("restoring pretrained model...")
        model = Model(sess=sess, config=None)
        model.restore(model_dir=args.model_dir)
    else:
        print("building new model...")
        os.makedirs(args.model_dir, exist_ok=True)
        config = {
            "model": {
                "encoder": args.encoder,
                "params": json.loads(args.enc_params)
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

        model = Model(sess=sess, config=config)
        model.build()

    model.train()
