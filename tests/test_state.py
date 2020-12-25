import pytest

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


@pytest.mark.parametrize("board, roll, expected", [
    pytest.param(
        Board(),
        (1, 2),
        {
            ()
        }
    )
])
def test_state_moves(board, roll, expected):
    s = State(board=board, roll=roll)
    actual = {x.state for x in s.moves}
    assert actual == expected
