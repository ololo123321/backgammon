import random
from typing import List
from abc import ABC, abstractmethod
from collections import namedtuple, defaultdict
from copy import deepcopy

import tensorflow as tf
import numpy as np

from src.state_pyx.state import Board, State
from src.nodes import MCNode, GameTreeNode
from src.environment import Environment


TransitionInfo = namedtuple("TransitionInfo", ["state", "reward"])


class BaseAgent(ABC):
    def __init__(self, sign=1):
        self.sign = sign
        self.token, self.opponent_token = ('x', 'o') if self.sign == 1 else ('o', 'x')

    @abstractmethod
    def ply(self, state: State) -> TransitionInfo:
        """
        https://en.wikipedia.org/wiki/Ply_(game_theory)
        given a list of possible states agent must chose one of them
        """
        pass


class RandomAgent(BaseAgent):
    def __init__(self, sign=1):
        super().__init__(sign)

    def ply(self, state: State) -> TransitionInfo:
        s = random.choice(state.transitions)
        r = 0.5
        return TransitionInfo(state=s, reward=r)


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

    def ply(self, state: State) -> TransitionInfo:
        transitions = state.transitions

        if len(transitions) == 1:
            print('Singe possible move is available')
            s = transitions[0]
            r = 0.5
            return TransitionInfo(state=s, reward=r)

        turn = self._get_input()
        board2state = {s.board.fingerprint: s for s in transitions}
        board_new = self._run_turn(turn=turn, board=state.board)
        if board_new.fingerprint in board2state:
            s = board2state[board_new.fingerprint]
            r = 0.5
            return TransitionInfo(state=s, reward=r)
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
    def __init__(self, sign=1, model=None):
        super().__init__(sign)
        self.model = model

    @classmethod
    def from_saved_model(cls, sign, export_dir):
        model = SavedModelWrapper(export_dir=export_dir)
        return cls(sign=sign, model=model)

    @classmethod
    def from_checkpoint(cls, sign, model):
        return cls(sign=sign, model=model)

    def ply(self, state: State) -> TransitionInfo:
        """
        Награда = вероятность выигрыша игрока со знаком self.sign, т.е. от лица self.
        """
        state.sign = self.sign  # для гарантии того, что self.sign и state.sign одинаковы
        transitions = state.transitions
        features = [s.features for s in transitions]
        x = np.concatenate(features, axis=0)  # [num_transitions, num_features]
        v = self.model.get_output(x)  # [num_transitions, 1]
        v = v.flatten()  # [num_transitions]

        # Модель предсказывает вероятность выигрыша игрока +1.
        # Соответственно, нужно получить вероятность противоположного события, если данный игрок -1.
        if self.sign == -1:
            v = 1.0 - v

        i = v.argmax()
        s = transitions[i]
        r = v[i]
        return TransitionInfo(state=s, reward=r)


class SavedModelWrapper:
    def __init__(self, export_dir):
        self.predict_fn = tf.contrib.predictor.from_saved_model(export_dir)

    def get_output(self, x: np.ndarray) -> np.ndarray:
        return self.predict_fn({
            "state": x,
            "training": False
        })["value"]


class KPlyAgent(BaseAgent):
    """
    Класс-обёртка над агентом, помогающий находить оптимальный ход с учётом просмотра на k ходов вперёд.
    k = 0 -> жадная стратегия
    """
    def __init__(self, sign=1, k=1, agent=None):
        super().__init__(sign)
        assert k >= 0
        self.k = k
        self.agent = agent

    def ply(self, state: State) -> TransitionInfo:
        """
        1. получить все доступные состояния s
        2. для каждого доступного состояния s построить игровое дерево T глубины k
        3. для каждого игрового дерева T посчитать матожиадние награды по листьям:
           E[r] = sum(p(leaf) * reward(leaf) for leaf in leaves(T))
        """
        if self.k == 0:
            return self.agent.ply(state)

        transitions = state.transitions
        rewards = []
        for t in transitions:
            root = GameTreeNode(
                sign=self.sign * -1,  # следующий ход противника
                state=t.reversed,
                agent=self.agent,
                r=None,
                p=1.0,
                k=self.k
            )
            r = root.expected_reward
            rewards.append(r)
        # каждому доступному состоянию соответствует своё игровое дерево, в котором первый ход делает противник.
        # следовательно, на нечётных глубинах получаем награды от лица противника, а на чётных - свои.
        # при этом свои хочется максимизировать, а противника - минимизировать.
        if self.k % 2 == 0:
            r_best = max(rewards)
        else:
            r_best = min(rewards)
        i = rewards.index(r_best)
        return TransitionInfo(state=transitions[i], reward=r_best)


class KPlyAgentFused(KPlyAgent):
    def __init__(self, sign=1, k=1, agent=None, batch_size=65536):
        super().__init__(sign=sign, k=k, agent=agent)
        self.batch_size = batch_size

    # TODO: мат. ожиданиия наград немного отличаются от тех, что получаются с помощью KPlyAgent.ply
    def ply(self, state: State) -> TransitionInfo:
        if self.k == 0:
            return self.agent.ply(state)

        transitions_root = state.transitions
        rolls = list(self._rolls_gen())
        num_rolls = len(rolls)
        max_transitions = 500

        # Node = namedtuple("Node", ["state", "id_action", "reward", "prob"])
        class Node:
            def __init__(self, id, id_action, state, reward, prob_path, prob_roll):
                self.id = id
                self.id_action = id_action
                self.state = state
                self.reward = reward
                self.prob_path = prob_path
                self.prob_roll = prob_roll

        leaves = [
            Node(id=i, id_action=i, state=s.reversed, reward=0.0, prob_path=1.0, prob_roll=None)
            for i, s in enumerate(transitions_root)
        ]
        sign = self.sign * -1
        board2features = {}  # мемоизация фичей
        for depth in range(self.k):
            # t0 = time()
            num_leaves = len(leaves)
            # TODO: мб вернуться к списку с последующим решейпом, чтоб не аллоцировать большой тензор
            candidates = np.empty((num_leaves, num_rolls, max_transitions), dtype='O')
            mask = np.zeros_like(candidates, dtype=np.int32)
            m = -1
            # TODO: распараллелить рассчёт фичей, ибо это узкое место
            for i, leaf in enumerate(leaves):
                for j, roll in enumerate(rolls):
                    state = leaf.state.copy
                    state.roll = roll
                    transitions_roll = state.transitions
                    p = 2 / 36 if len(roll) == 2 else 1 / 36
                    m = max(m, len(transitions_roll))
                    for k, t in enumerate(transitions_roll):
                        candidates[i, j, k] = Node(
                            id=leaf.id,
                            id_action=leaf.id_action,
                            state=t,
                            reward=None,
                            prob_path=leaf.prob_path,
                            prob_roll=p
                        )
                        mask[i, j, k] = 1

                        # мемоизация фичей
                        b = t.board.fingerprint
                        if b not in board2features:
                            board2features[b] = t.features

            # print("m:", m)
            candidates = candidates[..., :m]
            mask = mask[..., :m]

            # assert len(candidates) == num_leaves * num_rolls * m

            # print(f"depth: {depth}, num leaves: {num_leaves}, max num transitions: {m}, "
            #       f"num candidates: {len(candidates)}, time elapsed: {time() - t0}")

            candidates_flat = candidates.flatten()
            num_features = candidates_flat[0].state.features_dim  # TODO: костыль
            if hasattr(self.agent, "model"):
                rewards = []
                for start in range(0, len(candidates_flat), self.batch_size):
                    # print(start)
                    end = start + self.batch_size
                    # t0 = time()
                    # features = [c.state.features for c in candidates[start:end]]
                    # features = [board2features[c.state.board.fingerprint] for c in candidates_flat[start:end]]
                    # x = np.concatenate(features, axis=0)  # [N, num_features]
                    candidates_batch = candidates_flat[start:end]
                    x = np.zeros((len(candidates_batch), num_features), dtype=np.float32)
                    for i, c in enumerate(candidates_batch):
                        if c is not None:
                            x[i] = board2features[c.state.board.fingerprint]

                    # print(f"features computed, time elapsed: {time() - t0}")
                    # t0 = time()
                    rewards_i = self.agent.model.get_output(x)  # [N, 1]
                    # print("predictions computed, time elapsed:", time() - t0)
                    rewards.append(rewards_i)
                rewards = np.concatenate(rewards, axis=0)
                if sign == -1:
                    rewards = 1.0 - rewards
            else:
                rewards = np.full(shape=(len(candidates),), fill_value=0.5)

            mask_flat = mask.flatten()  # [N]
            rewards = rewards.flatten()  # [N]
            rewards *= mask_flat  # [N]
            shape = num_leaves, num_rolls, m
            rewards = rewards.reshape(shape)  # [num_leaves, num_rolls, max_num_transitions]
            zz = rewards.argmax(-1)  # [num_leaves, num_rolls]

            candidates = np.array(candidates).reshape(shape)
            # leaves_new = []
            leaves_new = {}
            # num_collapses = 0
            id_leaf_new = 0
            for i, j in np.ndindex(num_leaves, num_rolls):  # удобный аналог вложенным циклам
                k = zz[i, j]
                leaf = candidates[i, j, k]

                # гарантируется уникальность досок в разрезе leaf.id
                leaf_fingerprint = leaf.id, leaf.state.board.fingerprint

                # две комбинации кубиков привели к одному состоянию
                if leaf_fingerprint in leaves_new:
                    # num_collapses += 1
                    # раскрытие скобок в выражении prob_path * (p_roll_1 + ... + p_roll_k)
                    leaves_new[leaf_fingerprint].prob_path += leaf.prob_path * leaf.prob_roll
                else:
                    r = rewards[i, j, k]
                    leaf = Node(
                        id=id_leaf_new,
                        id_action=leaf.id_action,
                        state=leaf.state.reversed,
                        reward=r,
                        prob_path=leaf.prob_path * leaf.prob_roll,
                        prob_roll=None
                    )
                    leaves_new[leaf_fingerprint] = leaf
                    id_leaf_new += 1

            # print("num collapses:", num_collapses)

            leaves = list(leaves_new.values())

            # обновление знака (для корректного рассчёта награды)
            sign *= -1

        d = defaultdict(float)
        # action2prob = defaultdict(float)
        # action2count = defaultdict(int)
        for leaf in leaves:
            d[leaf.id_action] += leaf.reward * leaf.prob_path
            # action2prob[leaf.id_action] += leaf.prob_path
            # action2count[leaf.id_action] += 1

        # assert max(action2count.values()) <= 21, max(action2count.values())

        # print(action2count)
        # print("prob:")
        # print(action2prob)
        # print("E[r]:")
        # print(d)

        if self.k % 2 == 0:
            i = max(d, key=d.get)
        else:
            i = min(d, key=d.get)

        return TransitionInfo(state=transitions_root[i], reward=d[i])

    @staticmethod
    def _rolls_gen():  # TODO: копипаста из GameTreeNode!
        for i in range(1, 7):
            for j in range(i, 7):
                if i == j:
                    yield (i,) * 4
                else:
                    yield i, j


class MCAgent(BaseAgent):
    """
    Агент строит одно MC-дерево
    """
    def __init__(self, sign=1, agent=None, num_simulations=100, c=1.0, p=1.0):
        super().__init__(sign=sign)
        self.agent = agent
        self.num_simulations = num_simulations
        self.c = c
        self.p = p

        self._opponent = deepcopy(agent)
        self._opponent.sign = sign * -1

    def ply(self, state: State) -> TransitionInfo:
        root = self._monte_carlo_tree_search(state)
        v = max(child.visits for child in root.children)
        s = random.choice([child.state for child in root.children if child.visits == v])
        return TransitionInfo(state=s, reward=v)

    def _monte_carlo_tree_search(self, state: State) -> MCNode:
        # TODO: reward
        # TODO: при построении игрового дерева учитывать разные комбинации кубиков
        root = MCNode(sign=self.sign, parent=None, state=state, r=0.5, c=self.c, p=self.p)
        for _ in range(self.num_simulations):
            node = self._select(root)
            node = self._expand(node)
            result = self._simulate(node)
            self._back_propagate(node=node, result=result)
        return root

    @staticmethod
    def _select(node: MCNode) -> MCNode:
        """
        выбрать лучший узел игрового дерева такой, из которого попробованы не все действия
        """
        while node.is_fully_expanded and not node.is_terminal:
            node = node.best_child
        return node

    def _expand(self, node: MCNode) -> MCNode:
        """
        попробовать новое действие
        """
        # TODO: одним вызовом модели получать награды для всех доступных состояний из данного.
        s, r = self.agent.ply(node.state)
        child = MCNode(sign=node.sign * -1, parent=node, state=s, r=r, c=self.c, p=self.p)
        node.add_child(child)
        return child

    def _simulate(self, node: MCNode) -> int:
        """
        доиграть игру из выбранного состояния
        """
        # TODO: делать несколько симуляций на одной итерации, чтобы учесть разные выпадения кубиков
        env = Environment(agents=[self.agent, self._opponent], state=node.state)
        winner = env.play(verbose=False)
        result = int(winner == self.sign)
        return result

    @staticmethod
    def _back_propagate(node: MCNode, result: int):
        """
        обновыть скоры всех узлов на пути к результату
        """
        while node is not None:
            node.update(result=result)
            node = node.parent


# class MCAgentParallel(BaseAgent):
#     """
#     Агент строит несколько MC-деревьев, посещения детей корня суммируются
#     """
#     def __init__(self, sign=1, weights=None, n_simulations=100, c=1.0, n_trees=10, p=1.0):
#         super().__init__(sign)
#         self.weights = weights
#         self.n_simulations = n_simulations
#         self.c = c
#         self.n_trees = n_trees
#         self.p = p
#
#     def ply(self, state: State) -> TransitionInfo:
#         agent = MCAgent(weights=self.weights, n_simulations=self.n_simulations, c=self.c, p=self.p)
#         with Pool() as pool:
#             res = pool.map(agent.mcts, [state] * self.n_trees)
#
#         visits = {}
#         for children in res:
#             visits = {str(c.move): (visits.get(str(c.move), (0, None))[0] + c.visits, c.move) for c in children}
#
#         for k, (v, m) in visits.items():
#             print(f'move: {k}, visits: {v}')
#         v_max = max([v for v, m in visits.values()])
#         return random.choice([move for str_move, (v, move) in visits.items() if v == v_max])


# if __name__ == '__main__':
#     # from src.state_pyx.state import State
#     # base_agent_ = RandomAgent()
#     # agent_ = KPlyAgent(agent=base_agent_, k=2)
#     # s_ = State()
#     # info = agent_.ply(s_)
#     # print(info)
#     import os
#     from src.models import ModelTD
#     from src.environment import Environment
#     sess = tf.Session()
#     model = ModelTD(sess=sess, config=None)
#     model.restore("/tmp/backgammon_agent")
#     export_dir = "/tmp/backgammon_agent_saved_model"
#     os.system(f'rm -r {export_dir}')
#     model.export_inference_graph(export_dir)
#     agent_1 = TDAgent.from_saved_model(sign=1, export_dir=export_dir)
#     agent_2 = TDAgent.from_saved_model(sign=-1, export_dir=export_dir)
#     agent_2 = KPlyAgentFused(sign=-1, k=1, agent=agent_2)
#     env = Environment(agents=[agent_1, agent_2])
#     res = env.contest(num_episodes=10, verbose=True)
#     print(res)
