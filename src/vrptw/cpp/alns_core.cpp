// Adaptive large neighbourhood search for the vehicle routing problem with time windows (VRPTW).
// The algorithm is the same as in alns_py.py (the Python reference); only the language differs.
// Cost = total distance + BIG * routes, so fewer vehicles always win (the usual Solomon objective order).
//
// Two ways to test an insertion: `naive` walks the whole route (O(route length) per position) and
// `fast` uses cached service-start times and latest start times, so one position costs O(1). Both give
// the same answers; `fast` is the default and `naive` stays for the comparison and the profile.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace py = pybind11;
using std::vector;
using Clock = std::chrono::steady_clock;

static const double BIG = 1.0e4;
static const double INF = std::numeric_limits<double>::infinity();
static const double EPS = 1e-9;

struct Inst {
    int n = 0;  // customers 1..n, depot 0
    double cap = 0;
    vector<double> dem, ready, due, serv;
    vector<vector<double>> d;
};

typedef vector<vector<int>> Sol;

struct Stats {
    bool on = false;
    double t_feasibility = 0, t_destroy = 0, t_repair_total = 0;
    long long feasibility_calls = 0;
};

// ---------------------------------------------------------------------------------------------
// route with cached times (fast variant)
struct Route {
    vector<int> v;
    double load = 0;
    vector<double> a;  // service start at each node
    vector<double> l;  // latest service start that keeps the rest of the route feasible
};

static void rebuild(const Inst& I, Route& r) {
    const int m = (int)r.v.size();
    r.a.assign(m, 0);
    r.l.assign(m, 0);
    r.load = 0;
    double t = 0;
    int prev = 0;
    for (int k = 0; k < m; ++k) {
        int c = r.v[k];
        t = std::max(t + I.d[prev][c], I.ready[c]);
        r.a[k] = t;
        t += I.serv[c];
        prev = c;
        r.load += I.dem[c];
    }
    for (int k = m - 1; k >= 0; --k) {
        int c = r.v[k];
        double next_latest = (k == m - 1) ? I.due[0] - I.d[c][0] : r.l[k + 1] - I.d[c][r.v[k + 1]];
        r.l[k] = std::min(I.due[c], next_latest - I.serv[c]);
    }
}

// insertion of c at position p of route r, O(1)
static inline bool fast_ok(const Inst& I, const Route& r, int p, int c) {
    if (r.load + I.dem[c] > I.cap + EPS) return false;
    const int m = (int)r.v.size();
    int u = p == 0 ? 0 : r.v[p - 1];
    double dep_u = p == 0 ? 0.0 : r.a[p - 1] + I.serv[u];
    double tc = std::max(dep_u + I.d[u][c], I.ready[c]);
    if (tc > I.due[c] + EPS) return false;
    double dep_c = tc + I.serv[c];
    if (p == m) return dep_c + I.d[c][0] <= I.due[0] + EPS;
    int w = r.v[p];
    double tw = std::max(dep_c + I.d[c][w], I.ready[w]);
    return tw <= r.l[p] + EPS;
}

// naive test: walk the whole route
static bool naive_ok(const Inst& I, const vector<int>& r, int pos, int c, double load) {
    if (load + I.dem[c] > I.cap + EPS) return false;
    double t = 0;
    int prev = 0;
    const int m = (int)r.size();
    for (int k = 0; k <= m; ++k) {
        int v = k == pos ? c : (k < pos ? r[k] : r[k - 1]);
        t = std::max(t + I.d[prev][v], I.ready[v]);
        if (t > I.due[v] + EPS) return false;
        t += I.serv[v];
        prev = v;
    }
    return t + I.d[prev][0] <= I.due[0] + EPS;
}

static double dist_of(const Inst& I, const Sol& s) {
    double tot = 0;
    for (const auto& r : s) {
        int prev = 0;
        for (int v : r) { tot += I.d[prev][v]; prev = v; }
        tot += I.d[prev][0];
    }
    return tot;
}

static double cost_of(const Inst& I, const Sol& s) { return dist_of(I, s) + BIG * s.size(); }

struct Ins { double delta = INF; int route = -1, pos = -1; };

// ---------------------------------------------------------------------------------------------
// repair: greedy (regret = false) or regret-2 (regret = true)
static void repair(const Inst& I, Sol& s, vector<int> U, bool regret, bool fast, Stats& st) {
    auto t_start = Clock::now();
    vector<Route> R;
    for (auto& r : s) { Route x; x.v = r; rebuild(I, x); R.push_back(std::move(x)); }
    vector<double> loads;
    for (auto& r : R) loads.push_back(r.load);

    auto best_insert = [&](int c, double* second) {
        Ins best;
        double sec = INF;
        for (int ri = 0; ri < (int)R.size(); ++ri) {
            const auto& rt = R[ri];
            double rb = INF;
            int rp = -1;
            const int m = (int)rt.v.size();
            for (int p = 0; p <= m; ++p) {
                int a = p == 0 ? 0 : rt.v[p - 1];
                int b = p == m ? 0 : rt.v[p];
                double delta = I.d[a][c] + I.d[c][b] - I.d[a][b];
                if (delta >= rb) continue;
                bool ok;
                if (st.on) {
                    auto t0 = Clock::now();
                    ok = fast ? fast_ok(I, rt, p, c) : naive_ok(I, rt.v, p, c, rt.load);
                    st.t_feasibility += std::chrono::duration<double>(Clock::now() - t0).count();
                    ++st.feasibility_calls;
                } else {
                    ok = fast ? fast_ok(I, rt, p, c) : naive_ok(I, rt.v, p, c, rt.load);
                }
                if (ok) { rb = delta; rp = p; }
            }
            if (rp >= 0) {
                if (rb < best.delta) { sec = best.delta; best = {rb, ri, rp}; }
                else if (rb < sec) sec = rb;
            }
        }
        if (second) *second = sec;
        return best;
    };

    while (!U.empty()) {
        int pick = -1;
        Ins pi;
        double pkey = -INF;
        for (int k = 0; k < (int)U.size(); ++k) {
            int c = U[k];
            double sec;
            Ins b = best_insert(c, &sec);
            double key;
            if (b.route < 0) key = regret ? 1.0e9 : -(2 * I.d[0][c] + BIG);
            else if (regret) key = (sec >= INF ? 1.0e8 : sec) - b.delta;
            else key = -b.delta;
            if (key > pkey) { pkey = key; pick = k; pi = b; }
        }
        int c = U[pick];
        U.erase(U.begin() + pick);
        if (pi.route < 0) {
            Route x;
            x.v = {c};
            rebuild(I, x);
            R.push_back(std::move(x));
        } else {
            R[pi.route].v.insert(R[pi.route].v.begin() + pi.pos, c);
            rebuild(I, R[pi.route]);
        }
    }
    s.clear();
    for (auto& r : R) s.push_back(std::move(r.v));
    if (st.on) st.t_repair_total += std::chrono::duration<double>(Clock::now() - t_start).count();
}

static void remove_empty(Sol& s) {
    s.erase(std::remove_if(s.begin(), s.end(), [](const vector<int>& r) { return r.empty(); }), s.end());
}

static Sol initial(const Inst& I, bool fast) {
    vector<int> order(I.n);
    std::iota(order.begin(), order.end(), 1);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return I.due[a] < I.due[b]; });
    Sol s;
    Stats st;
    repair(I, s, order, false, fast, st);
    return s;
}

struct Solver {
    const Inst& I;
    std::mt19937_64 rng;
    explicit Solver(const Inst& inst, unsigned long long seed) : I(inst), rng(seed) {}
    double uni() { return std::uniform_real_distribution<double>(0, 1)(rng); }
    int uni_int(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); }

    vector<int> destroy(Sol& s, int op, int q) {
        vector<int> removed;
        if (op == 0) {  // random
            vector<int> all;
            for (auto& r : s) for (int v : r) all.push_back(v);
            std::shuffle(all.begin(), all.end(), rng);
            for (int k = 0; k < q && k < (int)all.size(); ++k) removed.push_back(all[k]);
        } else if (op == 1) {  // worst: largest detour
            vector<std::pair<double, int>> gains;
            for (auto& r : s)
                for (int p = 0; p < (int)r.size(); ++p) {
                    int a = p == 0 ? 0 : r[p - 1], b = p + 1 == (int)r.size() ? 0 : r[p + 1], c = r[p];
                    gains.push_back({I.d[a][c] + I.d[c][b] - I.d[a][b], c});
                }
            std::sort(gains.begin(), gains.end(), [](auto& x, auto& y) { return x.first > y.first; });
            for (int k = 0; k < q && !gains.empty(); ++k) {
                int idx = (int)(std::pow(uni(), 3.0) * gains.size());  // biased towards the largest detours
                removed.push_back(gains[idx].second);
                gains.erase(gains.begin() + idx);
            }
        } else if (op == 2) {  // Shaw: related to a seed customer
            vector<int> all;
            for (auto& r : s) for (int v : r) all.push_back(v);
            int seed = all[uni_int(0, (int)all.size() - 1)];
            vector<std::pair<double, int>> rel;
            for (int c : all) {
                double r1 = I.d[seed][c] / 100.0 + std::fabs(I.ready[seed] - I.ready[c]) / 100.0 +
                            std::fabs(I.dem[seed] - I.dem[c]) / I.cap;
                rel.push_back({r1, c});
            }
            std::sort(rel.begin(), rel.end());
            for (int k = 0; k < q && k < (int)rel.size(); ++k) removed.push_back(rel[k].second);
        } else {  // empty the smallest route
            int best = 0;
            for (int ri = 1; ri < (int)s.size(); ++ri) if (s[ri].size() < s[best].size()) best = ri;
            removed = s[best];
        }
        std::sort(removed.begin(), removed.end());
        removed.erase(std::unique(removed.begin(), removed.end()), removed.end());
        for (auto& r : s) r.erase(std::remove_if(r.begin(), r.end(), [&](int v) { return std::binary_search(removed.begin(), removed.end(), v); }), r.end());
        remove_empty(s);
        return removed;
    }
};

static py::tuple run(const vector<double>& x, const vector<double>& y, const vector<double>& dem, const vector<double>& ready,
                     const vector<double>& due, const vector<double>& serv, double cap, int iterations,
                     unsigned long long seed, bool fast, bool profile) {
    auto t0 = Clock::now();
    Inst I;
    I.n = (int)x.size() - 1;
    I.cap = cap;
    I.dem = dem; I.ready = ready; I.due = due; I.serv = serv;
    I.d.assign(I.n + 1, vector<double>(I.n + 1, 0));
    for (int i = 0; i <= I.n; ++i)
        for (int j = 0; j <= I.n; ++j) I.d[i][j] = std::hypot(x[i] - x[j], y[i] - y[j]);
    Solver S(I, seed);
    Stats st;
    st.on = profile;
    Sol cur = initial(I, fast), best = cur;
    double cc = cost_of(I, cur), bc = cc;
    double T0 = 50.0, T = T0, cool = std::pow(0.01 / T0, 1.0 / std::max(1, iterations));
    vector<double> wd(4, 1.0), wr(2, 1.0), sd(4, 0.0), sr(2, 0.0);
    vector<int> ud(4, 0), ur(2, 0);
    int qmax = std::max(4, std::min(40, (int)(0.3 * I.n))), qmin = std::max(3, (int)(0.1 * I.n));
    for (int it = 0; it < iterations; ++it) {
        auto pick = [&](vector<double>& w) {
            double tot = std::accumulate(w.begin(), w.end(), 0.0), r = S.uni() * tot, a = 0;
            for (int k = 0; k < (int)w.size(); ++k) { a += w[k]; if (r <= a) return k; }
            return (int)w.size() - 1;
        };
        int od = pick(wd), orr = pick(wr);
        Sol cand = cur;
        int q = S.uni_int(qmin, qmax);
        auto td = Clock::now();
        vector<int> U = S.destroy(cand, od, q);
        if (st.on) st.t_destroy += std::chrono::duration<double>(Clock::now() - td).count();
        repair(I, cand, U, orr == 1, fast, st);
        double nc = cost_of(I, cand);
        double score = 0;
        if (nc < bc - 1e-9) { best = cand; bc = nc; score = 33; cur = cand; cc = nc; }
        else if (nc < cc - 1e-9) { cur = cand; cc = nc; score = 9; }
        else if (S.uni() < std::exp(-(nc - cc) / std::max(T, 1e-9))) { cur = cand; cc = nc; score = 3; }
        sd[od] += score; ++ud[od]; sr[orr] += score; ++ur[orr];
        if ((it + 1) % 100 == 0) {
            for (int k = 0; k < 4; ++k) { if (ud[k]) wd[k] = 0.9 * wd[k] + 0.1 * sd[k] / ud[k]; wd[k] = std::max(wd[k], 0.05); sd[k] = 0; ud[k] = 0; }
            for (int k = 0; k < 2; ++k) { if (ur[k]) wr[k] = 0.9 * wr[k] + 0.1 * sr[k] / ur[k]; wr[k] = std::max(wr[k], 0.05); sr[k] = 0; ur[k] = 0; }
        }
        T *= cool;
    }
    double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    py::dict prof;
    if (profile) {
        prof["feasibility_seconds"] = st.t_feasibility;
        prof["destroy_seconds"] = st.t_destroy;
        prof["repair_seconds"] = st.t_repair_total;
        prof["feasibility_calls"] = st.feasibility_calls;
        prof["total_seconds"] = ms / 1000.0;
    }
    return py::make_tuple(best, dist_of(I, best), (int)best.size(), ms, prof);
}

PYBIND11_MODULE(_alns_cpp, m) {
    m.doc() = "ALNS for VRPTW (C++17 kernel)";
    m.def("solve",
          [](vector<double> x, vector<double> y, vector<double> dem, vector<double> ready, vector<double> due, vector<double> serv,
             double cap, int iterations, unsigned long long seed, bool fast) {
              auto r = run(x, y, dem, ready, due, serv, cap, iterations, seed, fast, false);
              return py::make_tuple(r[0], r[1], r[2], r[3]);
          },
          py::arg("x"), py::arg("y"), py::arg("demand"), py::arg("ready"), py::arg("due"), py::arg("service"), py::arg("capacity"),
          py::arg("iterations"), py::arg("seed"), py::arg("fast") = true,
          "Solve a VRPTW instance; returns (routes, distance, vehicles, milliseconds)");
    m.def("profile",
          [](vector<double> x, vector<double> y, vector<double> dem, vector<double> ready, vector<double> due, vector<double> serv,
             double cap, int iterations, unsigned long long seed, bool fast) {
              return run(x, y, dem, ready, due, serv, cap, iterations, seed, fast, true);
          },
          py::arg("x"), py::arg("y"), py::arg("demand"), py::arg("ready"), py::arg("due"), py::arg("service"), py::arg("capacity"),
          py::arg("iterations"), py::arg("seed"), py::arg("fast") = true,
          "Same as solve, with the time spent in the feasibility test, destroy and repair; returns (routes, distance, vehicles, ms, profile)");
}
