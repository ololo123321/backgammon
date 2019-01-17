import numpy as np
import random
from multiprocessing import Pool
from abc import ABCMeta, abstractmethod

from state import State
from utils import extract_features, choose_move_trained, sigmoid
from nodes import MCNode, GameTreeNode


class BaseAgent(metaclass=ABCMeta):
    def __init__(self, sign=1):
        self.sign = sign
        self.token, self.opponent_token = ('x', 'o') if self.sign == 1 else ('o', 'x')

    @abstractmethod
    def choose_move(self, state):
        """
        state - объект с методом get_states(), возвращающий доступные действия в виде пар (board, bar)
        """
        pass


class RandomAgent(BaseAgent):
    def __init__(self, sign=1):
        super().__init__(sign)

    def choose_move(self, state):
        return random.choice(state.get_moves())


class HumanAgent(BaseAgent):
    """
    1,2 - обычный ход
    1,-1 - постановка фигуры в дом на позицию 1
    -1,1 - выкидывание фигуры с позиции 1
    всё остальное - Invalid input
    """
    def __init__(self, sign=1):
        super().__init__(sign)

    @staticmethod
    def transform_move(move):
        try:
            start, end = move.split(',')
            start, end = int(start), int(end)
            if (0 <= start <= 23 and -1 <= end <= 23) or (-1 <= start <= 23 and 0 <= end <= 23):
                return start, end
        except:
            return move

    @staticmethod
    def transform_turn(turn_str):
        turn = []
        for move in turn_str.split():
            turn.append(HumanAgent.transform_move(move))
        return turn

    @staticmethod
    def turn_format_is_valid(turn):
        if len(turn) <= 4 and all([isinstance(move, tuple) for move in turn]):
            return turn
        return False

    @staticmethod
    def run_move(s, move):
        start, end = move
        if 0 <= start <= 23 and 0 <= end <= 23:  # обычный ход
            s[start] -= 1
            if s[end] >= 0:
                s[end] += 1
            else:
                s[end] = 1
        elif 0 <= start <= 23 and end == -1:  # возврат
            if s[start] >= 0:
                s[start] += 1
            else:
                s[start] = 1
        elif start == -1 and 0 <= end <= 23:  # выкидывание
            s[end] -= 1
        return s

    @staticmethod
    def run_turn(turn, board):
        board_new = board.copy()
        while turn:
            move = turn.pop(0)
            HumanAgent.run_move(board_new, move)
        return tuple(board_new)

    def get_input(self):
        turn = input('Enter turn: start,end start,end: ')
        turn = HumanAgent.transform_turn(turn)
        if HumanAgent.turn_format_is_valid(turn):
            return turn
        print('Invalid input')
        return self.get_input()

    def choose_move(self, state):
        """
        states_bars не пуст
        """
        moves = state.get_moves()
        if len(moves) == 1 and moves[0] == (state.board, state.bar):
            print('No moves')
            return state.board, state.bar

        turn = self.get_input()
        board_new = HumanAgent.run_turn(turn, state.board)
        d = {tuple(board): bar for board, bar in moves}
        if board_new in d:
            bar = d[board_new]
            return list(board_new), bar

        print('Invalid turn')
        print('Current state:', state.board, state.bar)
        print('Possible states:')
        for move in moves:
            print(move)
        return self.choose_move(state)


class TDAgent(BaseAgent):
    def __init__(self, sign=1, model=None, weights=None):
        super().__init__(sign)
        self.model = model
        self.weights = weights

    @staticmethod
    def feature_space_dim():
        state = State()
        return extract_features(state).shape[1]

    def choose_move(self, state):
        """
        Модель предсказывает вероятность выигрыша игрока +1.
        Соответственно, нужно получить вероятность противоположного события, если данный игрок -1.

        У аргумента должен быть метод get_moves() и атрибут sign
        """
        moves = state.get_moves()
        v_best = 0
        move_best = moves[0]
        for move in moves:
            state_new = state.clone()
            state_new.board, state_new.bar = move
            x = extract_features(state_new)
            v = self.model.get_output(x)
            if state_new.sign == -1:
                v = 1 - v
            if v > v_best:
                v_best = v
                move_best = move
        return move_best


class MCAgentBase(BaseAgent):
    """
    Агент строит одно MC-дерево
    """
    def __init__(self, sign=1, weights=None, n_simulations=100, C=1, p=1):
        super().__init__(sign=sign)
        self.weights = weights
        self.n_simulations = n_simulations
        self.C = C
        self.p = p

    def mcts(self, rootstate):
        rootnode = MCNode(sign=self.sign, state=rootstate, weights=self.weights, C=self.C, p=self.p)
        for i in range(self.n_simulations):
            node = rootnode
            state = rootstate.clone()

            while node.fully_expanded() and not node.terminal():
                node = node.choose_child()
                state.update(node.move)

            if not node.fully_expanded():
                move = choose_move_trained(node.state, self.weights)
                state.update(move)
                node = node.add_child(move, state)

            while not state.winner:
                # move = choose_move_trained(state, self.weights)
                move = random.choice(state.get_moves())  # так сильно быстрее, но хуже
                state.update(move)

            while node:
                node.update(result=int(state.winner == rootstate.sign))
                node = node.parent
        return rootnode.children

    def choose_move(self, state):
        children = self.mcts(state)
        v_max = max([c.visits for c in children])
        return random.choice([c.move for c in children if c.visits == v_max])


class MCAgent(BaseAgent):
    """
    Агент строит несколько MC-деревьев, посещения детей корня суммируются
    """
    def __init__(self, sign=1, weights=None, n_simulations=100, C=1, n_trees=10, p=1):
        super().__init__(sign)
        self.weights = weights
        self.n_simulations = n_simulations
        self.C = C
        self.n_trees = n_trees
        self.p = p

    def choose_move(self, state):
        agent = MCAgentBase(weights=self.weights, n_simulations=self.n_simulations, C=self.C, p=self.p)
        with Pool() as pool:
            res = pool.map(agent.mcts, [state] * self.n_trees)

        visits = {}
        for children in res:
            visits = {str(c.move): (visits.get(str(c.move), (0, None))[0] + c.visits, c.move) for c in children}

        for k, (v, m) in visits.items():
            print(f'move: {k}, visits: {v}')
        v_max = max([v for v, m in visits.values()])
        return random.choice([move for str_move, (v, move) in visits.items() if v == v_max])


class KPlyAgent(BaseAgent):
    def __init__(self, sign=1, k=2, weights=None):
        super().__init__(sign)
        self.k = k
        self.weights = weights

    def choose_move(self, state):
        if self.k > 1:
            with Pool() as pool:
                res = pool.map(GameTreeNode(sign=self.sign, state=state, weights=self.weights, k=self.k-1).expected_reward,
                               state.get_moves())
            rewards = {}
            for r, move in res:
                rewards[str(move)] = r, move

            d = {r: move for str_move, (r, move) in rewards.items()}
            return d[max(d)]  # награда всегда максимизируется, так как считается с учётом self.sign
        return choose_move_trained(state, weights=self.weights)
