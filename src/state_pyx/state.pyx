import random
import numpy as np
cimport numpy as np


cdef tuple roll_dice(bint first_move = False):
    """
    Бросок кубика.
    Если выпадает дубль, то нужно ходить 4 раза выпавшее число.
    На первом ходе при дубле нужно перебрасывать:
    https://nardy-wiki.ru/short-nardi - раздел "Розыгрыш первого хода"
    """
    cdef int r1, r2
    r1, r2 = random.randint(1, 6), random.randint(1, 6)
    if r1 == r2:
        if first_move:
            return roll_dice(first_move)
        else:
            return (r1,) * 4
    else:
        return r1, r2


cdef class Board:
    cdef list _board
    cdef dict _bar
    cdef set _towers

    def __init__(self, list board = None, dict bar=None):
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

    property board:
        def __get__(self): return self._board

    property bar:
        def __get__(self): return self._bar

    property towers:
        def __get__(self): return self._towers

    property fingerprint:
        def __get__(self):
            """
            минимальное хэшируемое представление доски.
            """
            cdef tuple board
            cdef tuple bar
            board = tuple(self._board)
            bar = self._bar[-1], self._bar[1]
            return board, bar

    property reversed:
        def __get__(self):
            cdef list board
            cdef dict bar
            board = [-x for x in self._board[::-1]]
            bar = {-k: v for k, v in self._bar.items()}
            return Board(board=board, bar=bar)

    property is_empty:
        def __get__(self):
            """не осталось своих фигур"""
            return len(self._towers) == 0

    property copy:
        def __get__(self): return Board(board=self._board.copy(), bar=self._bar.copy())

    # TODO: сделать не за O(n)
    property _t_min:
        def __get__(self): return min(self._towers)

    cdef bint is_valid_move(self, int start, int end):
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

    cpdef void move(self, int start, int end):
        """
        Предполагается, что валидность хода уже проверена
        :param start: позиция начала
        :param end: позиция конца. возможно
        :return:
        """
        self.remove_piece(start)
        if end <= 23:
            self.add_piece(end)

    cdef remove_piece(self, int p):
        """
        Удалить фигуру с позиции p
        :param p: номер позиции
        :return:
        """
        self._board[p] -= 1  # убрать фигуру с позиции start
        # если на позиции start фигур не осталось, то удалить башню
        if self._board[p] == 0:
            self._towers.remove(p)

    cpdef add_piece(self, int p):
        if self._bar[1] > 0:
            self._bar[1] -= 1  # убрать шашку с bar
        if self._board[p] >= 0:  # не противник
            self._board[p] += 1  # добавить шашку
        elif self._board[p] == -1:  # одна шашка противника
            self._board[p] = 1  # поставить одну шашку
            self._bar[-1] += 1  # добавить одну шашку противника в bar
        if self._board[p] == 1:  # если p не была в towers
            self._towers.add(p)


cdef class Node:
    cdef public Board board
    cdef public int depth
    cdef public tuple roll
    cdef public bint is_game_over

    def __init__(self, board, depth, roll, is_game_over):
        self.board = board
        self.depth = depth
        self.roll = roll
        self.is_game_over = is_game_over


cdef dict add_leaf(Node node, dict leaves):
    leaves[node.board.fingerprint] = node
    return leaves


cdef dict extend(Node node, int max_depth, dict leaves, bint is_double):
    """
    Построение игрового дерева, определяющего все возможные варианты ходов.
    Условия завершения обхода:
    1. Достигнута максимальная глубина, определяемая выпавшими кубиками
    """
    cdef int step, start, end
    cdef bint is_leaf

    # сходили максимальное число раз
    if node.depth == max_depth:
        leaves = add_leaf(node=node, leaves=leaves)
        return leaves

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
            extend(node=child, max_depth=max_depth, leaves=leaves, is_double=is_double)
        else:
            # если дубль, то мы уже не можем сделать ход
            if is_double:
                leaves = add_leaf(node=node, leaves=leaves)
            else:
                child = Node(board=board_copy, depth=node.depth + 1, roll=node.roll, is_game_over=False)
                extend(node=child, max_depth=max_depth, leaves=leaves, is_double=is_double)
    else:
        # пытаемся сделать ход с каждой башни
        is_leaf = True
        for start in board.towers:
            end = start + step
            if board.is_valid_move(start=start, end=end):
                is_leaf = False
                board_copy = board.copy

                # попытка подебажить это:
                # E   AttributeError: 'src.state_pyx.state.Board' object has no attribute 'move'
                # пока лечится с помощью cpdef move
                # # board_copy = Board(board=node.board.board.copy(), bar=node.board.bar.copy())
                # # board_copy = Board()
                # print(type(board), type(board_copy))
                # print(board.move(start=start, end=end))
                # print(board_copy.move(start=start, end=end))
                # print('000')
                # board.move(start=start, end=end)
                # print('111')
                # board_copy.move(start=start, end=end)
                # print('222')

                board_copy.move(start=start, end=end)

                if board_copy.is_empty:
                    # пустая доска ~ текущий игрок победил
                    child = Node(board=board_copy, depth=node.depth + 1, roll=node.roll, is_game_over=True)
                    leaves = add_leaf(node=child, leaves=leaves)
                else:
                    child = Node(board=board_copy, depth=node.depth + 1, roll=node.roll, is_game_over=False)
                    extend(node=child, max_depth=max_depth, leaves=leaves, is_double=is_double)
        # нельзя сделать ни одного хода
        if is_leaf:
            leaves = add_leaf(node=node, leaves=leaves)
    return leaves


cdef class State:
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
    cdef public Board board
    cdef public tuple roll
    cdef public int winner
    cdef public int sign

    def __init__(self, board=Board(), roll=(), winner=0, sign=1):
        self.board = board
        self.roll = roll
        if len(self.roll) == 0:
            self.roll = roll_dice(first_move=True)
        self.winner = winner
        self.sign = sign  # нужен только для определения победителя

    @property
    def transitions(self):
        """
        Список возможных переходов
        """
        cdef dict leaves
        cdef int max_depth
        cdef bint is_double
        cdef list res

        if self.winner != 0:
            print(f"game over due to winner found: {self.winner}")
            return []

        leaves = {}
        max_depth = len(self.roll)
        is_double = max_depth == 4

        root = Node(board=self.board, depth=0, roll=self.roll, is_game_over=False)
        leaves = extend(node=root, max_depth=max_depth, leaves=leaves, is_double=is_double)
        if len(self.roll) == 2:
            root = Node(board=self.board, depth=0, roll=self.roll[::-1], is_game_over=False)
            leaves = extend(node=root, max_depth=max_depth, leaves=leaves, is_double=is_double)

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
                s = State(board=node.board, roll=roll_next, winner=0, sign=self.sign)
                res.append(s)
        return res

    @property
    def features(self):
        """
        Признаковое описание состояня. Строится от лица игрока +1 независимо от того, какой игрок ходит.
        """
        cdef int on_board, on_bar, v
        cdef list features
        cdef np.ndarray[int, ndim=2] x

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
        x = np.array(features).astype(np.int32).reshape(1, -1)
        return x

    @property
    def features_dim(self):
        x = self.features
        return x.shape[1]

    @property
    def reversed(self):
        return State(board=self.board.reversed, roll=self.roll, winner=self.winner, sign=self.sign * -1)

    @property
    def is_game_over(self):
        return self.winner != 0

    @property
    def copy(self):
        return State(board=self.board.copy, roll=self.roll, winner=self.winner, sign=self.sign)


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
    # check()
