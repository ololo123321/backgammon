import random
from collections import namedtuple

Node = namedtuple('Node', ['board', 'bar', 'towers', 'depth'])


def roll_dice():
    r1, r2 = random.randint(1, 6), random.randint(1, 6)
    return (r1, r2) if r1 != r2 else (r1,) * 4


def is_valid_move(t, r, board, towers):
    end = t + r
    if end <= 23:
        if board[end] >= -1:  # <= 1 фигур противника
            return True
        return False
    else:
        t_min = min(towers)
        if t_min >= 18 and (r == 24 - t or t == t_min):
            return True
        return False


def add_piece(board, bar, p, towers):
    assert p <= 23, 'invalid position'
    assert board[p] >= -1, 'position is occupied'
    if bar[1] > 0:
        bar[1] -= 1  # убрать шашку с bar
    if board[p] >= 0:  # не противник
        board[p] += 1  # добавить шашку
    if board[p] == -1:  # одна шашка противника
        board[p] = 1  # поставить одну шашку
        bar[-1] += 1  # добавить одну шашку противника в bar
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
        self.board = [2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2]
        self.bar = {-1: 0, 1: 0}
        self.roll = roll_dice()
        while len(self.roll) == 4:
            self.roll = roll_dice()
        self.towers = None
        self.winner = None
        self.sign = 1  # нужен только для определения победителя

    def reverse_state(self):
        self.board = [-x for x in self.board[::-1]]
        self.bar = {-k: v for k, v in self.bar.items()}

    def get_moves_roll(self, roll):
        leaves = []
        max_depth = len(roll)

        def extend(node):
            if self.winner:
                return

            leaf_data = tuple(node.board), tuple(node.bar.items()), None, node.depth

            if node.depth == max_depth:
                leaves.append(Node(*leaf_data))
                return

            r = roll[node.depth]
            if node.bar[1]:
                board_, bar_, towers_ = node.board.copy(), node.bar.copy(), node.towers.copy()
                if node.board[r - 1] >= -1:
                    add_piece(board_, bar_, r - 1, towers_)
                    extend(Node(board_, bar_, towers_, node.depth + 1))
                else:
                    if max_depth == 2:
                        extend(Node(board_, bar_, towers_, node.depth + 1))
                    else:
                        leaves.append(Node(*leaf_data))
            else:
                is_leaf = True
                for t in node.towers:
                    if is_valid_move(t, r, node.board, node.towers):
                        is_leaf = False
                        board_, bar_, towers_ = node.board.copy(), node.bar.copy(), node.towers.copy()
                        board_[t] -= 1
                        if board_[t] == 0:
                            towers_.remove(t)
                        end = t + r
                        if end <= 23:
                            add_piece(board_, bar_, end, towers_)
                        if len(towers_) == 0:
                            self.winner = self.sign
                            leaves.append(Node(*leaf_data))
                            return
                        extend(Node(board_, bar_, towers_, node.depth + 1))
                if is_leaf:
                    leaves.append(Node(*leaf_data))

        root = Node(self.board, self.bar, self.towers, 0)
        extend(root)
        if self.winner:
            return leaves[-1]
        return leaves if leaves else [Node(tuple(self.board), tuple(self.bar.items()), None, 0)]

    def get_moves(self):
        self.towers = [i for i, x in enumerate(self.board) if x > 0]
        leaves = self.get_moves_roll(self.roll)
        if self.winner:
            return [(list(leaves.board), dict(leaves.bar))]
        if len(self.roll) == 2:
            leaves_ = self.get_moves_roll(self.roll[::-1])
            if self.winner:
                return [(list(leaves_.board), dict(leaves_.bar))]
            leaves += leaves_
        leaves = set(leaves)
        max_depth = max(leaf.depth for leaf in leaves)
        return [(list(leaf.board), dict(leaf.bar)) for leaf in leaves if leaf.depth == max_depth]

    def clone(self):
        s = State()
        for attr, value in vars(self).items():
            setattr(s, attr, value)
        return s

    def update(self, move):
        self.board, self.bar = move  # обновление доски
        self.sign *= -1  # обновление знака текущего игрока
        self.reverse_state()  # разворот доски к новому игрока
        self.roll = roll_dice()  # обновление кубиков


if __name__ == '__main__':
    """
    1 1 9
    1 2 15
    1 3 16
    1 4 14
    1 5 8
    1 6 10
    2 2 13
    2 3 17
    2 4 18
    2 5 8
    2 6 14
    3 3 13
    3 4 17
    3 5 9
    3 6 14
    4 4 12
    4 5 9
    4 6 14
    5 5 3
    5 6 7
    6 6 6
    """
    s = State()
    for i in range(1, 7):
        for j in range(i, 7):
            s.roll = i, j
            print(i, j, len(s.get_moves()))
