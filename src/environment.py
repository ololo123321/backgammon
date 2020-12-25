import random
from .state import State


class Environment:
    def __init__(self, agents=None, verbose=False):
        self.agents = agents
        self.verbose = verbose
        self.state = State()
        self.winner = None

    def play(self):
        i = random.randint(0, 1)
        while not self.winner:
            agent = self.agents[i]
            self.state.sign = agent.sign
            if self.verbose:
                self.draw(agent)
            transitions = self.state.transitions
            self.state = agent.ply(transitions)
            self.winner = self.state.winner
            i = (i + 1) % 2

    @staticmethod
    def draw_row(content):
        white_space = ' ' * 5
        print(white_space.join(content))

    @staticmethod
    def draw_locations(upper=True):
        if upper:
            locations = [y + ' ' if len(y) == 1 else y for y in [str(x) for x in range(11, -1, -1)]]
        else:
            locations = [str(x) for x in range(12, 24)]
        Environment.draw_row(locations)

    @staticmethod
    def draw_side(side, agent, upper=True):
        height = max([abs(x) for x in side])
        for h in range(height):
            content = []
            for n in side[::-1]:
                v = agent.token if n > 0 else agent.opponent_token
                if upper:
                    if abs(n) > h:
                        content.append(v + ' ')
                    else:
                        content.append(' ' * 2)
                else:
                    if abs(n) >= height - h:
                        content.insert(0, v + ' ')
                    else:
                        content.insert(0, ' ' * 2)
            Environment.draw_row(content)

    def draw(self, agent):
        if agent.sign == 1:
            upper_side = [n for i, n in enumerate(self.state.board) if i < 12]
            lower_side = [n for i, n in enumerate(self.state.board) if i >= 12]
        else:
            upper_side = [n for i, n in enumerate(self.state.board[::-1]) if i < 12]
            lower_side = [n for i, n in enumerate(self.state.board[::-1]) if i >= 12]

        if agent.sign == 1:
            Environment.draw_locations(upper=True)
        else:
            Environment.draw_locations(upper=False)

        print('-' * 79)
        Environment.draw_side(upper_side, agent, upper=True)
        print()
        Environment.draw_side(lower_side, agent, upper=False)
        print('-' * 79)

        if agent.sign == 1:
            Environment.draw_locations(upper=False)
        else:
            Environment.draw_locations(upper=True)

        print('agent {} rolled ({}, {})'.format(agent.token, *self.state.roll))
        print('current board:', self.state.board)
        print('current bar:', self.state.bar)
        print('=' * 80)
