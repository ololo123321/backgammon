import random
from typing import List, Tuple, Union
from collections import namedtuple
from copy import deepcopy

import numpy as np

ROLL_TYPE = Union[Tuple[int, int], Tuple[int, int, int, int]]


def roll_dice(first_move: bool = False) -> ROLL_TYPE:
    """
    Бросок кубика.
    Если выпадает дубль, то нужно ходить 4 раза выпавшее число.
    На первом ходе при дубле нужно перебрасывать:
    https://nardy-wiki.ru/short-nardi - раздел "Розыгрыш первого хода"
    """
    r1, r2 = random.randint(1, 6), random.randint(1, 6)
    if r1 == r2:
        if first_move:
            return roll_dice(first_move)
        else:
            return (r1,) * 4
    else:
        return r1, r2


class ReprMixin:
    def __repr__(self):
        class_name = self.__class__.__name__
        params = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f'{class_name}({params})'


class Board(ReprMixin):
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

        # # хранение индекса первой башни, чтоб каждый раз его не пересчитывать за O(num towers)
        # self._t_min = min(self._towers)

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
    def fingerprint(self) -> Tuple:
        """
        минимальное хэшируемое представление доски.
        """
        board = tuple(self._board)
        bar = self._bar[-1], self._bar[1]
        return board, bar

    @property
    def reversed(self):
        board = [-x for x in self._board[::-1]]
        bar = {-k: v for k, v in self._bar.items()}
        b = Board(board=board, bar=bar)
        return b

    @property
    def is_empty(self) -> bool:
        """не осталось своих фигур"""
        return len(self._towers) == 0

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

    def remove_piece(self, p: int):
        """
        Удалить фигуру с позиции p
        :param p: номер позиции
        :return:
        """
        assert 0 <= p <= 23, f'invalid position: {p}'
        assert self._board[p] >= 1, f'position {p} is empty'
        self._board[p] -= 1  # убрать фигуру с позиции start
        # если на позиции start фигур не осталось, то удалить башню
        if self._board[p] == 0:
            self._towers.remove(p)
            # if self._towers:
            #     self._t_min = min(self._towers)  # TODO: сделать обновление минимума не за O(n)
            # else:
            #     self._t_min = -1

    def add_piece(self, p: int):
        assert 0 <= p <= 23, f'invalid position: {p}'
        assert self._board[p] >= -1, f'position {p} is occupied'
        if self._bar[1] > 0:
            self._bar[1] -= 1  # убрать шашку с bar
        if self._board[p] >= 0:  # не противник
            self._board[p] += 1  # добавить шашку
        elif self._board[p] == -1:  # одна шашка противника
            self._board[p] = 1  # поставить одну шашку
            self._bar[-1] += 1  # добавить одну шашку противника в bar
        if self._board[p] == 1:  # если p не была в towers
            assert p not in self._towers, f"p: {p}, towers: {self._towers}"
            self._towers.add(p)
            # if p < self._t_min:
            #     self._t_min = p

    @property
    def copy(self):
        return deepcopy(self)

    @property
    def _t_min(self):
        # TODO: сделать не за O(n)
        return min(self._towers)


class State(ReprMixin):
    """
    Класс состояния игры. Характеризуется:
    - выпавшими кубиками,
    - доской
    - текущим игроком

    Интерфейс:
    s = State()
    transitions = s.transitions
    s = choose_best_state(transitions)
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
    def transitions(self) -> List:
        """
        Список возможных переходов
        """
        if self.winner is not None:
            print(f"game over due to winner found: {self.winner}")
            return []

        leaves = {}
        max_depth = len(self.roll)
        is_double = max_depth == 4

        # узел игрового дерева
        Node = namedtuple("Node", ["board", "depth", "roll", "is_game_over"])

        def add_leaf(node: Node):
            leaves[node.board.fingerprint] = node

        def extend(node: Node):
            """
            Построение игрового дерева, определяющего все возможные варианты ходов.
            Условия завершения обхода:
            1. Достигнута максимальная глубина, определяемая выпавшими кубиками
            """
            # сходили максимальное число раз
            if node.depth == max_depth:
                add_leaf(node)
                return

            # выбираем кубик для хода
            step = node.roll[node.depth]

            # если есть съеденные фигурки
            board = node.board
            if board.bar[1]:
                board_copy = board.copy
                home_position = step - 1

                # если можно съеденную фигурку поставить на доску:
                if board_copy.board[home_position] >= -1:
                    board_copy.add_piece(home_position)
                    child = Node(board=board_copy, depth=node.depth + 1, roll=node.roll, is_game_over=False)
                    extend(child)
                else:
                    # если дубль, то мы уже не можем сделать ход
                    if is_double:
                        add_leaf(node)
                    else:
                        child = Node(board=board_copy, depth=node.depth + 1, roll=node.roll, is_game_over=False)
                        extend(child)
            else:
                # пытаемся сделать ход с каждой башни
                is_leaf = True
                for start in board.towers:
                    end = start + step
                    if board.is_valid_move(start=start, end=end):
                        is_leaf = False
                        board_copy = board.copy
                        board_copy.move(start=start, end=end)
                        if board_copy.is_empty:
                            # пустая доска ~ текущий игрок победил
                            child = Node(board=board_copy, depth=node.depth + 1, roll=node.roll, is_game_over=True)
                            add_leaf(child)
                            # return
                        else:
                            child = Node(board=board_copy, depth=node.depth + 1, roll=node.roll, is_game_over=False)
                            extend(child)
                # нельзя сделать ни одного хода
                if is_leaf:
                    add_leaf(node)

        root = Node(board=self.board, depth=0, roll=self.roll, is_game_over=False)
        extend(root)
        if len(self.roll) == 2:
            root = Node(board=self.board, depth=0, roll=self.roll[::-1], is_game_over=False)
            extend(root)

        # гарантируется непустота leaves
        # игрок обязан ходить максимально возможное количество раз
        max_depth = max(node.depth for node in leaves.values())
        res = []
        roll_next = roll_dice()
        for node in leaves.values():
            if node.is_game_over:
                s = State(board=node.board, roll=roll_next, winner=self.sign, sign=self.sign)
                res.append(s)
            elif node.depth == max_depth:
                s = State(board=node.board, roll=roll_next, winner=None, sign=self.sign)
                res.append(s)
        return res

    @property
    def features(self) -> np.ndarray:
        """
        Признаковое описание состояня. Строится от лица игрока +1 независимо от того, какой игрок ходит.
        """
        board_pos = self.board if self.sign == 1 else self.board.reversed
        features = []
        for sgn in [1, -1]:
            on_board = 0
            on_bar = board_pos.bar[sgn]
            for p in board_pos.board:
                v = abs(p) if np.sign(p) == sgn else 0
                on_board += v
                features.append(v)
            features.append(on_board)  # число фигур на доске
            features.append(on_bar)  # число съеденных фигур
            features.append(15 - on_board - on_bar)  # число выкинутых фигур
        features.append(int(self.sign == 1))  # кто ходит
        return np.array(features).reshape(1, -1)

    @property
    def features_dim(self):
        x = self.features
        return x.shape[1]

    @property
    def copy(self):
        """точная копия состояния"""
        return deepcopy(self)

    @property
    def reversed(self):
        return State(board=self.board.reversed, roll=self.roll, winner=self.winner, sign=self.sign * -1)


if __name__ == '__main__':

    def check():
        for i in range(1, 7):
            for j in range(i, 7):
                if i != j:
                    roll = i, j
                else:
                    roll = i, i, i, i
                ss = State(roll=roll)
                print(roll, len(ss.transitions))
    check()
