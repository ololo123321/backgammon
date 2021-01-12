import random
from tqdm import trange
# from .state import State
from .state_pyx.state import State


class Environment:
    def __init__(self, agents=None, state=None):
        self.agents = agents
        self.state = state if state is not None else State()

    def contest(self, num_episodes=100, verbose=False, print_fn=None):
        """
        Туринр двух агентов.
        :param num_episodes: число игр
        :param verbose: писать ли промежуточный результат
        :param print_fn: callable: str -> stdout. сделано так, чтоб сообщения можно было выводить как в консоль,
        так и в файл (нужно при обучении)
        :return:
        """
        print_fn = print_fn if print_fn is not None else print
        res = {-1: 0, 1: 0}
        for i in trange(num_episodes):
            winner = self.play()
            res[winner] += 1
            if verbose:
                print_fn(f"[episode {i + 1} / {num_episodes}] result: {res}")
        return res

    def play(self, verbose=False):
        i = random.randint(0, 1)
        # step = 0
        while not self.state.is_terminal:
            agent = self.agents[i]
            self.state.sign = agent.sign
            if verbose:
                self.draw(agent)
            s, _ = agent.ply(self.state)
            self.state = s.reversed
            i = (i + 1) % 2
            # step += 1
            # if step > 200:
            #     print(self.state)
            #     raise
        winner = self.state.winner
        self.state = State()
        return winner

    def draw(self, agent):
        if agent.sign == 1:
            upper_side = [n for i, n in enumerate(self.state.board.board) if i < 12]
            lower_side = [n for i, n in enumerate(self.state.board.board) if i >= 12]
        else:
            upper_side = [n for i, n in enumerate(self.state.board.board[::-1]) if i < 12]
            lower_side = [n for i, n in enumerate(self.state.board.board[::-1]) if i >= 12]

        if agent.sign == 1:
            self._draw_locations(upper=True)
        else:
            self._draw_locations(upper=False)

        print('-' * 79)
        self._draw_side(upper_side, agent, upper=True)
        print()
        self._draw_side(lower_side, agent, upper=False)
        print('-' * 79)

        if agent.sign == 1:
            self._draw_locations(upper=False)
        else:
            self._draw_locations(upper=True)

        print(f'agent {agent.token} rolled {self.state.roll}')
        print('current board:', self.state.board.board)
        print('current bar:', self.state.board.bar)
        print('=' * 80)

    @staticmethod
    def _draw_row(content):
        white_space = ' ' * 5
        print(white_space.join(content))

    def _draw_locations(self, upper=True):
        if upper:
            locations = [y + ' ' if len(y) == 1 else y for y in [str(x) for x in range(11, -1, -1)]]
        else:
            locations = [str(x) for x in range(12, 24)]
        self._draw_row(locations)

    def _draw_side(self, side, agent, upper=True):
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
            self._draw_row(content)
