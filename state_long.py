import random
from collections import namedtuple

Node = namedtuple('Node', ['board', 'towers', 'depth'])


def roll_dice():
    r1, r2 = random.randint(1, 6), random.randint(1, 6)
    return (r1, r2) if r1 != r2 else (r1,) * 4


def is_valid_move(t, r, state, towers):
    end = t + r
    if 0 <= end <= 23:
        if state[end] >= 0:
            return True
        return False
    else:
        t_min = min(towers)
        if t_min >= 18 and (r == 24 - t or t == t_min):
            return True
        return False


def add_piece(board, p, towers):
    assert p <= 23, 'invalid position'
    assert board[p] >= 0, 'position is occupied'
    if board[p] >= 0:  # не противник
        board[p] += 1  # добавить шашку
    if board[p] == 1:  # если p не была в towers
        towers.append(p)


class State:
    """
    Класс состояния игры. Характеризуется:
    - выпавшими кубиками,
    - расположением фигур на доске,
    - съеденными фигурами
    """

    def __init__(self):
        self.board = [15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.roll = roll_dice()
        while len(self.roll) == 4:
            self.roll = roll_dice()
        self.towers = None
        self.winner = None
        self.sign = 1  # нужен только для определения победителя

    def reverse_state(self):
        self.board = [-x for x in self.board[::-1]]

    def get_moves_roll(self, roll):
        leaves = []
        max_depth = len(roll)

        def extend(node):
            if self.winner:
                return

            leaf_data = tuple(node.board), None, node.depth

            if node.depth == max_depth:
                leaves.append(Node(*leaf_data))
                return

            r = roll[node.depth]
            is_leaf = True
            for t in node.towers:
                if is_valid_move(t, r, node.board, node.towers):
                    is_leaf = False
                    board_, towers_ = node.board.copy(), node.towers.copy()
                    board_[t] -= 1
                    if board_[t] == 0:
                        towers_.remove(t)
                    end = t + r
                    if end <= 23:
                        add_piece(board_, end, towers_)
                    if len(towers_) == 0:
                        self.winner = self.sign
                        leaves.append(Node(*leaf_data))
                        return
                    extend(Node(board_, towers_, node.depth + 1))
            if is_leaf:
                leaves.append(Node(*leaf_data))

        root = Node(self.board, self.towers, 0)
        extend(root)
        if self.winner:
            return leaves[-1]
        return leaves if leaves else [Node(tuple(self.board), None, 0)]

    def get_moves(self):
        self.towers = [i for i, x in enumerate(self.board) if x > 0]
        leaves = self.get_moves_roll(self.roll)
        if self.winner:
            return [list(leaves.board)]
        if len(self.roll) == 2:
            leaves_ = self.get_moves_roll(self.roll[::-1])
            if self.winner:
                return [list(leaves_.board)]
            leaves += leaves_
        leaves = set(leaves)
        max_depth = max(leaf.depth for leaf in leaves)
        return [list(leaf.board) for leaf in leaves if leaf.depth == max_depth]

    def clone(self):
        s = State()
        for attr, value in self.__dict__().items():
            setattr(s, attr, value)
        return s

    def update(self, move):
        self.board = move  # обновление доски
        self.sign *= -1  # обновление знака текущего игрока
        self.reverse_state()  # разворот доски к новому игрока
        self.roll = roll_dice()  # обновление кубиков
