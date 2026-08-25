from src.model import walk_forward_cv


def test_walk_forward_cv_builds_expanding_splits():
    X = list(range(10))
    y = list(range(10))
    splits = walk_forward_cv(X, y, initial_train_size=6, step_size=2)

    assert splits == [
        (list(range(6)), [6, 7]),
        (list(range(8)), [8, 9]),
    ]
