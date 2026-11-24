import json

import pytest

import vrptw
from vrptw import alns_py, exact, instance as ins

HAS_CPP = vrptw._alns_cpp is not None
needs_cpp = pytest.mark.skipif(not HAS_CPP, reason="C++ kernel not built")


def test_solomon_files_and_best_known_values_are_complete():
    names = ins.names()
    assert len(names) == 56
    b = ins.bks()
    assert all(n.upper() in b for n in names)
    C101 = ins.load("C101")
    assert C101.n == 100 and C101.capacity == 200 and C101.ready[1] == 912


def test_checker_rejects_bad_solutions():
    I = ins.load("C101").subset(5)
    good = [[1, 2, 3, 4, 5]]
    assert not ins.check(I, [[1, 2, 3, 4]])["feasible"]            # a customer is missing
    assert not ins.check(I, [[5, 4, 3, 2, 1]])["feasible"]         # time windows broken
    big = ins.load("C101").subset(30)
    assert not ins.check(big, [list(range(1, 31))])["feasible"]    # capacity or time windows
    assert isinstance(ins.check(I, good)["distance"], float)


def test_python_alns_gives_feasible_routes():
    I = ins.load("R101").subset(25)
    r, d, v, ms = alns_py.solve(*I.args(), 150, 3)
    chk = ins.check(I, r)
    assert chk["feasible"] and abs(chk["distance"] - d) < 1e-6 and v == len(r)


@needs_cpp
def test_cpp_alns_gives_feasible_routes_on_all_classes():
    for name in ("C101", "R101", "RC101", "C201", "R201", "RC201"):
        I = ins.load(name)
        r, d, v, ms = vrptw.solve(I, iterations=300, seed=2)
        chk = ins.check(I, r)
        assert chk["feasible"] and abs(chk["distance"] - d) < 1e-6


@needs_cpp
def test_cpp_reaches_known_optimum_of_c101():
    I = ins.load("C101")
    r, d, v, ms = vrptw.solve(I, iterations=3000, seed=1)
    b = ins.bks()["C101"]
    assert v == b["vehicles"] and d <= b["distance"] + 0.05


@needs_cpp
def test_cpp_is_much_faster_than_python():
    I = ins.load("C101")
    tp = vrptw.solve(I, iterations=20, seed=1, backend="python")[3]
    tc = vrptw.solve(I, iterations=20, seed=1, backend="cpp")[3]
    assert tp / tc > 10


def test_exact_methods_agree_and_alns_matches_the_optimum():
    I = ins.load("R101").subset(8)
    a, b = exact.solve_cpsat(I, 30), exact.solve_scip(I, 30)
    assert a["optimal"] and b["optimal"]
    assert a["objective"] == pytest.approx(b["objective"], abs=1e-2)
    assert ins.check(I, a["routes"])["feasible"] and ins.check(I, b["routes"])["feasible"]
    r, d, v, ms = (vrptw.solve(I, 400, 1) if HAS_CPP else alns_py.solve(*I.args(), 300, 1))
    assert d + 1e4 * v == pytest.approx(a["objective"], abs=1e-2)


def test_highs_model_agrees_with_cpsat():
    pytest.importorskip("pyomo")
    pytest.importorskip("highspy")
    I = ins.load("RC101").subset(8)
    a, h = exact.solve_cpsat(I, 30), exact.solve_highs(I, 30)
    assert h["optimal"] and h["objective"] == pytest.approx(a["objective"], abs=1e-2)


def test_service_solves_and_reports_metrics():
    from fastapi.testclient import TestClient
    from vrptw.service import app

    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/solve", json={"instance": "C101", "customers": 20, "iterations": 100, "seed": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["feasible"] and body["solve_ms"] > 0 and sorted(x for rt in body["routes"] for x in rt) == list(range(1, 21))
    m = c.get("/metrics").json()
    assert m["runs"] >= 1 and m["infeasible_runs"] == 0 and m["latency_ms_p95"] >= m["latency_ms_p50"]
    assert c.post("/solve", json={"instance": "XYZ"}).status_code == 404
    assert c.post("/solve", json={"instance": "C101", "iterations": 10**7}).status_code == 422
    assert c.post("/solve", json={}).status_code == 422


@needs_cpp
def test_fast_insertion_test_gives_the_same_routes_as_the_naive_one():
    for name in ("C101", "R101", "RC201"):
        I = ins.load(name)
        a = vrptw.solve(I, iterations=200, seed=5, fast=True)
        b = vrptw.solve(I, iterations=200, seed=5, fast=False)
        assert a[0] == b[0] and a[1] == pytest.approx(b[1])
