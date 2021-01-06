from argparse import ArgumentParser
from src.agents import TDAgent, KPlyAgent, HumanAgent
from src.environment import Environment


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--export_dir')
    parser.add_argument('--sign', type=int, choices=[-1, 1], default=1, required=False)
    parser.add_argument('--k', type=int, default=1, required=False)
    args = parser.parse_args()
    print(args)

    sgn = args.sign
    human = HumanAgent(sign=sgn)

    opponent = TDAgent.from_saved_model(sign=-sgn, export_dir=args.export_dir)
    opponent = KPlyAgent(sign=-sgn, k=args.k, agent=opponent)

    agents = [human, opponent]
    env = Environment(agents, verbose=True)
    env.play()
