import random
from typing import List, Tuple, Union
from collections import namedtuple
from copy import deepcopy

ROLL_TYPE = Union[Tuple[int, int], Tuple[int, int, int, int]]


def roll_dice(first_move: bool = False) -> ROLL_TYPE:
    """
    Бросок кубика.
    Если выпадает дубль, то нужно ходить 4 раза выпавшее число.
    На первом ходе при дубле нужно перебрасывать:
    https://nardy-wiki.ru/short-nardi - раздел "Розыгрыш первого хода"
    """
    r1, r2 = random.randint(1, 6), random.randint(1, 6)
    if first_move:
        if r1 == r2:
            return roll_dice(first_move)
        else:
            return r1, r2
    else:
        if r1 == r2:
            return (r1,) * 4
        else:
            return r1, r2


class Board:
    def __init__(self, board=None, bar=None):
        if board is not None:
            self._board = board
        else:
            self._board = [2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2]  # доска

        if bar is not None:
            self._bar = bar
        else:
            self._bar = {-1: 0, 1: 0}  # выкинутые фигуры; 1 - свои, -1 - противника

        # индексы позиций со своими фигурами.
        # для большей эффективности удобно их посчитать один раз, пройдя цикл по всей доске,
        # а затем только обновлять этот список без проходов по всей доске
        self._towers = {i for i, x in enumerate(self._board) if x > 0}

        # хранение индекса первой башни, чтоб каждый раз его не пересчитывать за O(num towers)
        self._t_min = min(self._towers)

    @property
    def board(self):
        return self._board

    @property
    def bar(self):
        return self._bar

    @property
    def towers(self):
        return self._towers

    @property
    def state(self) -> Tuple:
        """
        hashable состояние доски
        """
        board = tuple(self._board)
        bar = self._bar[-1], self._bar[1]
        return board, bar

    def reverse(self):
        self._board = [-x for x in self._board[::-1]]
        self._bar = {-k: v for k, v in self._bar.items()}

    def is_valid_move(self, start: int, end: int) -> bool:
        """
        :param start:
        :param end:
        :return:
        """
        assert 0 <= start <= 23, f"invalid start position: {start}"
        assert self._bar[1] == 0, f'there is lock on moves due to figures on bar'

        # end = start + step
        step = end - start
        # перемещение фигуры на доске
        if end <= 23:
            # 1. на позиции start должна быть фигура
            # 2. на позиции end должна быть либо одна фигура противника, либо пусто, либо свои фигуры
            return (self._board[start] >= 1) and (self._board[end] >= -1)
        # выкидывание фигуры с доски
        else:
            # 1. все фигуры в зоне [18, 23]
            # 2.1. есть фигура на позиции 24 - step
            # 2.2. все фигуры расположены либо на позиции в зоне выкидывания,
            # соответствующей номеру на кубике, либо левее
            return (self._t_min >= 18) and ((step == 24 - start) or (start == self._t_min))

    def is_empty(self):
        """не осталось своих фигур"""
        return len(self._towers) == 0

    def move(self, start, end):
        """
        Предполагается, что валидность хода уже проверена
        :param start: позиция начала
        :param end: позиция конца. возможно
        :return:
        """
        self.remove_piece(start)
        if end <= 23:
            self.add_piece(end)

    def remove_piece(self, p):
        assert 0 <= p <= 23, f'invalid position: {p}'
        assert self._board[p] >= 1, f'position {p} is empty'
        self._board[p] -= 1  # убрать фигуру с позиции start
        # если на позиции start фигур не осталось, то удалить башню
        if self._board[p] == 0:
            self._towers.remove(p)
        if p == self._t_min:
            self._t_min = min(self._towers)  # TODO: сделать так, чтобы не надо было за O(n) обновлять минимум

    def add_piece(self, p: int):
        assert 0 <= p <= 23, f'invalid position: {p}'
        assert self._board[p] >= -1, f'position {p} is occupied'
        if self._bar[1] > 0:
            self._bar[1] -= 1  # убрать шашку с bar
        if self._board[p] >= 0:  # не противник
            self._board[p] += 1  # добавить шашку
        if self._board[p] == -1:  # одна шашка противника
            self._board[p] = 1  # поставить одну шашку
            self._bar[-1] += 1  # добавить одну шашку противника в bar
        if self._board[p] == 1:  # если p не была в towers
            self._towers.add(p)
            if p < self._t_min:
                self._t_min = p


class State:
    """
    Класс состояния игры. Характеризуется:
    - выпавшими кубиками,
    - доской
    - текущим игроком

    Интерфейс:
    s = State()
    next_states = s.moves(roll)
    best_move = moves[0]
    s.update(move)
    """

    def __init__(
            self,
            board: Board = None,
            roll: ROLL_TYPE = None,
            winner: int = None,
            sign: int = 1
    ):
        self.board = board if board is not None else Board()
        self.roll = roll if roll is not None else roll_dice(first_move=True)
        self.winner = winner
        self.sign = sign  # нужен только для определения победителя

    @property
    def moves(self) -> List[Board]:
        leaves = {}
        max_depth = len(self.roll)
        is_double = max_depth == 4

        # узел игрового дерева
        Node = namedtuple("Node", ["board", "depth"])

        def add_leaf(node):
            leaves[node.board.state] = node

        def extend(node: Node, roll: Tuple) -> bool:
            winner_found = False

            # сходили максимальное число раз
            if node.depth == max_depth:
                add_leaf(node)
                return winner_found

            # выбираем кубик для хода
            step = roll[node.depth]

            # если есть съеденные фигурки
            if node.board.bar[1]:
                board_copy = deepcopy(node.board)
                home_position = step - 1

                # если можно съеденную фигурку поставить на доску:
                if board_copy[home_position] >= -1:
                    board_copy.add_piece(home_position)
                    child = Node(board=board_copy, depth=node.depth + 1)
                    extend(node=child, roll=roll)
                else:
                    # если дубль, то мы уже не можем сделать ход
                    if is_double:
                        add_leaf(node)
                    else:
                        child = Node(board=node.board, depth=node.depth + 1)
                        extend(node=child, roll=roll)
            else:
                # пытаемся сделать ход с каждой башни
                is_leaf = True
                for start in node.board.towers:
                    end = start + step
                    if node.board.is_valid_move(start=start, end=end):
                        is_leaf = False
                        board_copy = deepcopy(node.board)
                        board_copy.move(start=start, end=end)
                        if board_copy.is_empty:
                            add_leaf(node)
                            winner_found = True
                            return winner_found
                        else:
                            child = Node(board=board_copy, depth=node.depth + 1)
                            extend(node=child, roll=roll)
                # нельзя сделать ни одного хода
                if is_leaf:
                    add_leaf(node)

        root = Node(board=self.board, depth=0)
        is_game_over = extend(node=root, roll=self.roll)
        if is_game_over:
            self.winner = self.sign
            return []
        elif len(self.roll) == 2:
            is_game_over = extend(node=root, roll=self.roll[::-1])
            if is_game_over:
                self.winner = self.sign
                return []
            else:
                if leaves:
                    max_depth = max(node.depth for node in leaves.values())
                    boards = [node.board for node in leaves.values() if node.depth == max_depth]
                    return boards
                else:
                    return []

    def clone(self):
        s = State()
        for attr, value in vars(self).items():
            setattr(s, attr, value)
        return s

    def update(self, move: Board):
        self.board = move  # обновление доски
        self.sign *= -1  # обновление знака текущего игрока
        self.board.reverse()  # разворот доски к новому игрока
        self.roll = roll_dice()  # обновление кубиков
