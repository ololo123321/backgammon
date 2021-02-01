import random
from collections import defaultdict
from .state_pyx.state import State
from .utils import rolls_gen


class MCNode:
    def __init__(
            self,
            sign: int = 1,
            parent=None,
            state: State = None,
            r: float = None,
            c: float = 1.0,
            p: float = 1.0
    ):
        self.sign = sign
        self.parent = parent
        self.state = state
        self.state.sign = self.sign  # нужно для корректной работы extract_features

        self.r = r  # reward
        self.c = c  # exploration parameter
        self.p = p  #

        self.wins = 0
        self.visits = 0

        self.children = set()

        self.untried_moves = self.state.transitions

    @property
    def ucb(self) -> float:
        """
        http://web.stanford.edu/%7Esurag/posts/alphazero.html - отсюда взята формула
        """
        return self.wins / self.visits + self.c * self.r * self.visits ** 0.5 / self.visits

    @property
    def best_child(self):
        return max(self.children, key=lambda child: child.ucb)

    @property
    def is_fully_expanded(self) -> bool:
        """
        Попробованы все доступные действия
        p - вероятность того, что в течение одной симуляции
        выбор будет сделан из всех возможных ходов на этапе select
        """
        if len(self.untried_moves) == 0:
            return True
        elif random.random() < 1.0 - self.p:
            return True
        else:
            return False

    @property
    def is_terminal(self) -> bool:
        """
        Лист игрового дерева
        """
        return len(self.children) == 0

    def add_child(self, child):
        self.children.add(child)

    def update(self, result: int):
        self.visits += 1
        self.wins += result


class GameTreeNode:
    def __init__(self, sign=1, state=None, agent=None, r=None, p=1.0, k=2):
        self.sign = sign  # данный знак присваивается всем узлам дерева
        self.state = state
        self.agent = agent
        self.r = r  # награда за достижение данного узла
        self.p = p  # вероятность достигнуть данный узел. Единица для корня
        self.k = k

        self.children = []

        if self.agent is not None:
            self.agent.sign = sign
        if self.state is not None:
            self.state.sign = sign

    @property
    def expected_reward(self) -> float:
        if self.state.is_terminal:
            return 1.0
        nodes = [self]
        for _ in range(self.k):
            leaves_level = []
            for node in nodes:
                node.expand()
                leaves_level += node.children
            nodes = leaves_level
        # если дерево скошенное, то сумма может отличаться от единицы
        # p = sum(node.p for node in nodes)
        # assert round(p, 6) == 1.0, f"p actual: {p}, num nodes: {len(nodes)}, is terminal: {self.state.is_game_over}"
        r = sum(node.r * node.p for node in nodes)
        # assert 0 <= r <= 1.0, r
        return r

    def expand(self):
        if self.state.is_terminal:
            return
        # в одно состяоние можно прийти разными путями:
        # нарпимер, если выпадет (1,1) и (1,3), то можно походить одной фигурой 4 раза.
        board2prob = defaultdict(float)
        board2info = {}
        for roll in rolls_gen():
            state = self.state.copy
            state.roll = roll
            s, r = self.agent.ply(state)
            p = 2 / 36 if len(roll) == 2 else 1 / 36
            b = s.board.fingerprint
            board2info[b] = s, r
            board2prob[b] += p
        for b, (s, r) in board2info.items():
            p = board2prob[b] * self.p
            child = GameTreeNode(sign=self.sign * -1, state=s.reversed, agent=self.agent, r=r, p=p, k=self.k)
            self.children.append(child)
