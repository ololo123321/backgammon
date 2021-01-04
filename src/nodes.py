import random
from collections import defaultdict
from .state_pyx.state import State
from .utils import extract_features, get_reward


class MCNode:
    def __init__(
            self,
            sign: int = 1,
            move=None,
            parent=None,
            state: State = None,
            weights=None,
            c: float = 1.0,
            p: float = 1.0
    ):
        self.sign = sign
        self.move = move
        self.parent = parent
        self.state = state
        self.sign = self.sign  # нужно для корректной работы extract_features
        self.weights = weights
        self.c = c
        self.p = p

        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = self.state.transitions

    def ucb(self, child):
        """
        http://web.stanford.edu/%7Esurag/posts/alphazero.html - отсюда взята формула
        """
        x = extract_features(child.state)
        r = get_reward(x, self.weights)
        r = r if self.sign == 1 else 1 - r
        return child.wins / child.visits + self.c * r * self.visits ** 0.5 / child.visits

    @property
    def best_child(self):
        return max(self.children, key=lambda child: self.ucb(child))

    def add_child(self, move, state):
        child = MCNode(sign=self.sign, move=move, parent=self, state=state)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def update(self, result):
        self.visits += 1
        self.wins += result

    def fully_expanded(self):
        """
        Попробованы все доступные действия
        p - вероятность того, что в течение одной симуляции
        выбор будет сделан из всех возможных ходов на этапе selection
        """
        if not self.untried_moves:
            return True
        elif random.random() < 1 - self.p:
            return True
        return False

    def terminal(self):
        """
        Лист игрового дерева
        """
        return not self.children

    def get_moves(self):
        """
        Чтоб instance этого класса можно было сувать в choose_move
        """
        return self.untried_moves


class GameTreeNode:
    def __init__(self, sign=1, parent=None, state=None, agent=None, r=None, p=1.0, k=2):
        self.sign = sign  # данный знак присваивается всем узлам дерева
        self.parent = parent
        self.state = state
        self.agent = agent
        self.r = r  # награда за достижение данного узла
        self.p = p  # вероятность достигнуть данный узел. Единица для корня
        self.k = k

        self.children = []

    @property
    def expected_reward(self) -> float:
        nodes = [self]
        for _ in range(self.k):
            leaves_level = []
            for node in nodes:
                node.expand()
                leaves_level += node.children
            nodes = leaves_level
        r = sum(node.r * node.p for node in nodes)
        return r

    def expand(self):
        # в одно состяоние можно прийти разными путями:
        # нарпимер, если выпадет (1,1) и (1,3), то можно походить одной фигурой 4 раза.
        board2prob = defaultdict(float)
        board2info = {}
        for roll in self._rolls_gen():
            state = self.state.copy
            state.roll = roll
            s, r = self.agent.ply(state)
            r = r if self.sign == 1 else 1.0 - r
            p = 2/36 if len(roll) == 2 else 1/36
            b = s.board.fingerprint
            board2info[b] = s, r
            board2prob[b] += p
        for b, (s, r) in board2info.items():
            p = board2prob[b] * self.p
            child = GameTreeNode(sign=self.sign, parent=self, state=s, agent=self.agent, r=r, p=p, k=self.k)
            self.children.append(child)

    @staticmethod
    def _rolls_gen():
        for i in range(1, 7):
            for j in range(i, 7):
                if i == j:
                    yield (i,) * 4
                else:
                    yield i, j
