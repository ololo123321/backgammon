import pytest
from collections import namedtuple

from src.state import roll_dice, Board, State


def test_roll_dice():
    roll = roll_dice(first_move=True)
    assert len(roll) == 2


@pytest.mark.parametrize("board, p, expected_board, expected_towers, expected_min_tower", [
    pytest.param(
        Board(board=[2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2]),
        0,
        [1, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2],
        {0, 11, 16, 18},
        0
    ),
    pytest.param(
        Board(board=[1, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2]),
        0,
        [0, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2],
        {11, 16, 18},
        11
    )
])
def test_board_remove_piece(board, p, expected_board, expected_towers, expected_min_tower):
    board.remove_piece(p)
    assert board.board == expected_board
    assert board._towers == expected_towers
    assert board._t_min == expected_min_tower


# TODO
def test_board_remove_piece_exception():
    assert True


@pytest.mark.parametrize("board, p, expected_board, expected_towers, expected_bar, expected_min_tower", [
    pytest.param(
        Board(board=[2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2]),
        0,
        [3, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2],
        {0, 11, 16, 18},
        {-1: 0, 1: 0},
        0,
        id="добавили на башню"
    ),
    pytest.param(
        Board(board=[2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2]),
        1,
        [2, 1, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2],
        {0, 1, 11, 16, 18},
        {-1: 0, 1: 0},
        0,
        id="сделали новую башню"
    ),
    pytest.param(
        Board(board=[0, 2, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2]),
        0,
        [1, 2, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2],
        {0, 1, 11, 16, 18},
        {-1: 0, 1: 0},
        0,
        id="сделали новую минимальную башню"
    ),
    pytest.param(
        Board(board=[0, 2, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2]),
        0,
        [1, 2, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2],
        {0, 1, 11, 16, 18},
        {-1: 0, 1: 0},
        0,
        id="сделали новую минимальную башню"
    ),
    pytest.param(
        Board(board=[2, -1, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2]),
        1,
        [2, 1, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2],
        {0, 1, 11, 16, 18},
        {-1: 1, 1: 0},
        0,
        id="съеди фигуру противника"
    )
])
def test_board_add_piece(board, p, expected_board, expected_towers, expected_bar, expected_min_tower):
    board.add_piece(p)
    assert board.board == expected_board
    assert board._towers == expected_towers
    assert board.bar == expected_bar
    assert board._t_min == expected_min_tower


# TODO
def test_board_add_piece_exception():
    assert True


@pytest.mark.parametrize("board, start, end, expected", [
    pytest.param(
        [2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2], 0, 1, True,
        id="ordinal move"
    ),
    pytest.param(
        [2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2], 0, 5, False,
        id="move on large opponent's tower"
    ),
    pytest.param(
        [2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2], 16, 18, True,
        id="move on own tower"
    ),
    pytest.param(
        [2, -1, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2], 0, 1, True,
        id="eat opponent's tower"
    ),
    pytest.param(
        [2, -1, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2], 18, 24, False,
        id="wrong drop piece"
    ),
    pytest.param(
        [0, -1, 0, 0, 0, -5, 0, -3, 0, 0, 0, 0, -5, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, -2], 18, 24, True,
        id="right drop piece"
    ),
    pytest.param(
        [0, -1, 0, 0, 0, -5, 0, -3, 0, 0, 0, 0, -5, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -2], 19, 25, True,
        id="right drop piece"
    ),
    pytest.param(
        [0, -1, 0, 0, 0, -5, 0, -3, 0, 0, 0, 0, -5, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, -2], 19, 25, False,
        id="wrong drop piece"
    ),
])
def test_board_is_valid_move(board, start, end, expected):
    b = Board(board=board)
    actual = b.is_valid_move(start=start, end=end)
    assert actual == expected


def test_board_move():
    assert True


Case = namedtuple("Case", ["board", "bar", "towers", "t_min"])


"""
II                               I 
11 10 9  8  7  6  5  4  3  2  1  0
x           o     o              x
x           o     o              x
x           o     o
x                 o
x                 o
-----------------------------------
o                 x
o                 x
o           x     x
o           x     x              o
o           x     x              o
12 13 14 15 16 17 18 19 20 21 22 23
            III   IV
"""


@pytest.mark.parametrize("board, roll, expected", [
    pytest.param(
        Board(),
        (1, 5),
        # первый ход 1 возможен с башен 1, 3, 4
        # первый ход 5 возможен с башен 2, 3
        # одной шашкой все ходы можно сходить с 1, 2, 3 башен
        # в одно и то же состояние ведут следующие пути:
        # * (11, 16) -> (16, 17); (16, 17) -> (11, 15)
        # таким образом, итоговое число ходов равно 2 * 3 + 3 - 1 = 8
        {
            (1, 1, 0, 0, 0, -5, 0, -3, 0, 0, 0, 4, -5, 0, 0, 0, 4, 0, 5, 0, 0, 0, 0, -2),  # (0, 1), (11, 16)
            (1, 1, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 2, 0, 5, 0, 0, 1, 0, -2),  # (0, 1), (16, 21)

            (2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 4, -5, 0, 0, 0, 3, 1, 5, 0, 0, 0, 0, -2),  # (16, 17), (11, 16)
            (2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 1, 1, 5, 0, 0, 1, 0, -2),  # (16, 17), (16, 21)

            (2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 4, -5, 0, 0, 0, 4, 0, 4, 1, 0, 0, 0, -2),  # (18, 19), (11, 16)
            (2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 2, 0, 4, 1, 0, 1, 0, -2),  # (18, 19), (16, 21)

            (1, 0, 0, 0, 0, -5, 1, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2),  # (0, 1), (1, 5)
            (2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 2, 0, 5, 0, 0, 0, 1, -2),  # (16, 17), (17, 22)
        }
    )
])
def test_state_moves(board, roll, expected):
    s = State(board=board, roll=roll)
    actual = {tuple(x.board.board) for x in s.transitions}
    # print(expected - actual)
    # print('123')
    # print(actual - expected)
    assert len(actual) == len(expected)
    assert actual == expected
