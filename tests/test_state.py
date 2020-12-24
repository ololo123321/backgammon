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


# @pytest.mark.parametrize("board, roll, expected", [
#     pytest.param(
#         Board(),
#         (1, 2),
#         {
#             ()
#         }
#     )
# ])
# def test_moves(board, roll, expected):
#     s = State(board=board, roll=roll)
#     actual = {x.state for x in s.moves}
#     assert actual == expected
