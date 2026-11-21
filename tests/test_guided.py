"""Learned guidance for ALNS: removal sets, features, and a search whose result is re-costed by the original model."""
import numpy as np
import pytest

from ebus.alns import initial_solution
from ebus.guided import FEATURES, OPS, features, guided_alns, removal_set, wait_before
from ebus.model import Solution, cost_breakdown
from ebus.validate import check_solution
from test_ebus import small


class ConstantSurrogate:
    """Predicts a fixed change for every candidate; lets the tests control the surrogate."""
    def __init__(self, v):
        self.v = v

    def predict(self, X):
        return np.full(len(X), self.v)


@pytest.fixture(scope="module")
def inst():
    return small("S", 6 * 60, 9 * 60)


def test_removal_sets_are_valid_trip_sets(inst):
    st = initial_solution(inst, frozenset(range(inst.T)))
    rng = np.random.default_rng(0)
    for op in OPS:
        rem = removal_set(inst, st.plans, op, nq=8, rng=rng)
        assert rem and all(0 <= x < inst.n for x in rem)


def test_feature_vector_matches_names_and_waits_are_nonnegative(inst):
    st = initial_solution(inst, frozenset(range(inst.T)))
    wait = wait_before(inst, st.plans)
    assert (wait >= 0).all()
    f = features(inst, wait, {0, 1, 2}, "random", 0.5, 1.0)
    assert len(f) == len(FEATURES)
    assert f[OPS.index("random")] == 1.0 and f[OPS.index("related")] == 0.0


@pytest.mark.parametrize("selector", ["weights", "ucb"])
def test_guided_search_returns_valid_solution_not_worse_than_start(inst, selector):
    start = initial_solution(inst, frozenset(range(inst.T)))
    sol, stats = guided_alns(inst, time_limit=3, seed=0, selector=selector)
    assert not check_solution(inst, sol)
    assert cost_breakdown(inst, sol)["cost"] <= start.cost + 1e-6
    assert stats["full_evals"] <= stats["iterations"]


def test_surrogate_run_is_valid_and_counts_predictions(inst):
    sol, stats = guided_alns(inst, time_limit=3, seed=1, selector="ucb", surrogate=ConstantSurrogate(-1.0), k=3)
    assert not check_solution(inst, sol)
    assert stats["pred_improve"] > 0 and stats["pred_improve_true"] <= stats["pred_improve"]


def test_moves_log_records_recomputed_cost_change(inst):
    moves = []
    guided_alns(inst, time_limit=2, seed=2, selector="weights", moves=moves)
    assert moves and all(len(m["features"]) == len(FEATURES) for m in moves)
    assert all(np.isfinite(m["delta"]) for m in moves)
