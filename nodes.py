import random
from utils import extract_features, get_reward, choose_move_trained


class MCNode:
    def __init__(self, sign=1, move=None, parent=None, state=None, weights=None, C=1, p=1):
        self.sign = sign
        self.move = move
        self.parent = parent
        self.state = state
        self.state = self.sign  # нужно для корректной работы extract_features
        self.weights = weights
        self.C = C
        self.p = p

        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = self.state.get_moves()

    def ucb(self, child):
        """
        http://web.stanford.edu/%7Esurag/posts/alphazero.html - отсюда взята формула
        """
        x = extract_features(child.state)
        r = get_reward(x, self.weights)
        r = r if self.sign == 1 else 1 - r
        return child.wins / child.visits + self.C * r * self.visits ** 0.5 / child.visits

    def choose_child(self):
        return sorted(self.children, key=lambda c: self.ucb(c))[-1]

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
    def __init__(self, sign=1, parent=None, state=None, weights=None, r=None, p=1, k=2):
        self.sign = sign  # данный знак присваивается всем узлам дерева
        self.parent = parent
        self.state = state
        self.weights = weights
        self.r = r  # награда за достижение данного узла
        self.p = p  # вероятность достигнуть данный узел. Единица для корня
        self.k = k

        self.children = []

    def expand(self):
        rolls = self.state.first_rolls + [[i] * 4 for i in range(1, 7)]
        for roll in rolls:
            state = self.state.clone()
            state.roll = roll
            move = choose_move_trained(state, self.weights)
            state.update(move)
            x = extract_features(state)
            r = get_reward(x, self.weights)
            r = r if self.sign == 1 else 1 - r
            p = 2/36 if len(roll) == 2 else 1/36
            p *= self.p
            child = GameTreeNode(sign=self.sign, parent=self, state=state, weights=self.weights, r=r, p=p, k=self.k)
            self.children.append(child)

    def expected_reward(self, move):
        state_c = self.state.clone()
        state_c.update(move)
        nodes = [GameTreeNode(sign=self.sign, state=state_c, weights=self.weights)]
        depth = self.k
        while depth:
            leaves_level = []
            for node in nodes:
                node.expand()
                leaves_level += node.children
            nodes = leaves_level
            depth -= 1
        # print(sum([node.r * node.p for node in nodes]), move)
        return sum([node.r * node.p for node in nodes]), move
