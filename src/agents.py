import random
from typing import List
from multiprocessing import Pool
from abc import ABC, abstractmethod
from collections import namedtuple

import numpy as np

from .state import Board, State
from .utils import choose_move_trained
from .nodes import MCNode, GameTreeNode


class BaseAgent(ABC):
    def __init__(self, sign=1):
        self.sign = sign
        self.token, self.opponent_token = ('x', 'o') if self.sign == 1 else ('o', 'x')

    @abstractmethod
    def ply(self, state: State) -> State:
        """
        https://en.wikipedia.org/wiki/Ply_(game_theory)
        given a list of possible states agent must chose one of them
        """
        pass


class RandomAgent(BaseAgent):
    def __init__(self, sign=1):
        super().__init__(sign)

    def ply(self, state: State) -> State:
        return random.choice(state.transitions)


class InvalidInput(Exception):
    """
    Ошибка парсинга атрибутов
    """


Move = namedtuple("Move", ["start", "end"])


class HumanAgent(BaseAgent):
    """
    1,2 - обычный ход
    1,-1 - постановка фигуры в дом на позицию 1
    -1,1 - выкидывание фигуры с позиции 1
    всё остальное - Invalid input
    """
    def __init__(self, sign=1):
        super().__init__(sign)

    def ply(self, state: State) -> State:
        transitions = state.transitions

        if len(transitions) == 1:
            print('Singe possible move is available')
            return transitions[0]

        turn = self._get_input()
        board2state = {s.board.fingerprint: s for s in transitions}
        board_new = self._run_turn(turn=turn, board=state.board)
        if board_new.fingerprint in board2state:
            return board2state[board_new.fingerprint]
        else:
            print('Invalid turn')
            print('Current state:', state.board.board, state.board.bar)
            print('Possible states:')
            for k in board2state.keys():
                print(k)
            print("Proposed state:", board_new.fingerprint)
            return self.ply(state)

    def _get_input(self):
        turn = input('Enter turn: start,end start,end: ')
        try:
            return self._transform_turn(turn)
        except InvalidInput:
            print(f'Invalid input: {turn}')
            return self._get_input()

    def _transform_turn(self, turn_str: str) -> List[Move]:
        turn = []
        moves = turn_str.split()
        if len(moves) > 4:
            raise InvalidInput(f"expected number of moves <= 4, got {len(moves)}")
        for move in moves:
            try:
                move = self._transform_move(move)
                turn.append(move)
            except InvalidInput as e:
                raise e
        return turn

    def _transform_move(self, move: str) -> Move:
        self._check_move(move)

        move = Move(*map(int, move.split(',')))

        if (0 <= move.start <= 23 and -1 <= move.end <= 23) or (-1 <= move.start <= 23 and 0 <= move.end <= 23):
            return move
        else:
            raise InvalidInput(f"invalid positions: start: {move.start}, end: {move.end}")

    @staticmethod
    def _run_turn(turn: List[Move], board: Board) -> Board:
        board_copy = board.copy
        for move in turn:
            if (0 <= move.start <= 23) and (0 <= move.end <= 23):
                board_copy.move(start=move.start, end=move.end)
            elif (move.start == -1) and (0 <= move.end <= 23):
                board_copy.remove_piece(move.end)
            elif (0 <= move.start <= 23) and (move.end == -1):
                board_copy.add_piece(move.start)
            else:
                raise
        return board_copy

    @staticmethod
    def _check_move(move: str):
        valid_positions = set(map(str, range(0, 24)))
        valid_positions.add('-1')
        try:
            start, end = move.split(',')
            assert start in valid_positions and end in valid_positions
        except:
            raise InvalidInput(f"unable to parse move {move}")


class TDAgent(BaseAgent):
    def __init__(self, sign=1, model=None, weights=None):
        super().__init__(sign)
        self.model = model
        self.weights = weights

    def ply(self, state: State) -> State:
        state.sign = self.sign  # для гарантии того, что self.sign и state.sign одинаковы
        transitions = state.transitions
        features = [s.features for s in transitions]
        x = np.concatenate(features, axis=0)  # [num_transitions, num_features]
        v = self.model.get_output(x)  # [num_transitions, 1]
        v = v.flatten()  # [num_transitions]

        # Модель предсказывает вероятность выигрыша игрока +1.
        # Соответственно, нужно получить вероятность противоположного события, если данный игрок -1.
        if self.sign == 1:
            idx_best = v.argmax()
        else:
            idx_best = v.argmin()

        s_best = transitions[idx_best]
        return s_best


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
