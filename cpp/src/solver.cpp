#include "../include/solver.hpp"
#include <numeric>
#include <sstream>
#include <string>

using namespace std;
using namespace GiNaC;

bool Solver::var_compute(const exT &deps,
                         const map<ex, vector<double>, ex_is_less> &dWts,
                         const exmap &vars, double T, ex Dtv, ex dtv,
                         exmap &toret) const {
  bool err = false;
  exmap temp1, nvars;
  for (auto it = vars.begin(); it != vars.end(); ++it) {
    toret[it->first] =
        EM(it->second, deps.at(it->first).op(0), deps.at(it->first).op(1), Dtv,
           dtv, dWts.at(it->first), vars, T);
  }
  // XXX: Now compute the values in two half-steps.
  for (auto it = vars.begin(); it != vars.end(); ++it) {
    auto f = dWts.at(it->first).begin(), l = dWts.at(it->first).begin() + R / 2;
    nvars[it->first] =
        EM(it->second, deps.at(it->first).op(0), deps.at(it->first).op(1),
           Dtv / 2, dtv / 1, f, l, vars, T);
  }
  for (auto it = nvars.begin(); it != nvars.end(); ++it) {
    auto f = dWts.at(it->first).begin() + R / 2, l = dWts.at(it->first).end();
    nvars[it->first] =
        EM(it->second, deps.at(it->first).op(0), deps.at(it->first).op(1),
           Dtv / 2, dtv / 1, f, l, nvars,
           T + ex_to<numeric>((Dtv / 2).evalf()).to_double());
  }
  // XXX: Now do the final check
#ifdef DEBUG
  cout << toret << "\n";
  cout << nvars << "\n";
#endif // DEBUG
  vector<bool> errs;
  for (auto it = nvars.begin(); it != nvars.end(); ++it) {
    errs.push_back(
        abs(toret[it->first] - nvars[it->first]) / (nvars[it->first] + ε) <= ε);
  }
#ifdef DEBUG
  cout << "Dtv: " << Dtv << ", dtv: " << dtv << "\n";
  cout << vars << "\n";
  std::for_each(std::begin(errs), std::end(errs),
                [](bool i) { cout << i << " "; });
  cout << "\n";
#endif // DEBUG
  err = all_of(errs.begin(), errs.end(), [](bool i) { return i == true; });
  return err;
}

double Solver::default_compute(const exT &deps, const exmap &vars,
                               const map<ex, vector<double>, ex_is_less> &dWts,
                               exmap &toret, double T) const {
  double step = 0;
  ex Dtv = DEFAULT_STEP;
  ex dtv;
  bool err;
  while (true) {
    dtv = Dtv / R;
    err = var_compute(deps, dWts, vars, T, Dtv, dtv, toret);
    if (err) {
      step = ex_to<numeric>(Dtv.evalf()).to_double();
      break;
    } else {
      Dtv /= 2;
    }
  }
  return step;
}

double Solver::zstep(const ex &left, const lst &right, const exT &deps,
                     const exmap &vars, double T,
                     const map<ex, vector<double>, ex_is_less> &dWts,
                     exmap &toret, double Uz) const {
  /**
   * Computes the step size using the jump edges
   * left: symbol
   * right is 2 list of derivatives for left
   * deps is the list of derivatives
   */
  double step = 0;
  // XXX: fill in the algorithm
  symbol t("t");
  if (right.op(1) != 0)
    throw runtime_error("Rate cannot be a stochastic DE");
  if ((Uz == INF) || isnan(Uz)) {
    step = INF;
    toret = vars;
    return step;
  }
  ex zdt = right.op(0);
#ifdef ZDEBUG
  cout << "zdt: " << zdt << "\n";
#endif                    // ZDEBUG
  zdt = zdt.subs(vars);   // Substitution with current values
  zdt = zdt.subs(t == T); // Subs time t
  zdt = zdt.evalf();
#ifdef ZDEBUG
  cout << "zdt after subs: " << zdt << "\n";
  cout << "Uz: " << Uz << "\n";
#endif // ZDEBUG
  if (zdt == 0) {
    toret = vars;
    step = INF;
    return step;
  }

  // XXX: Now compute the step size
  ex L = abs(Uz - vars.at(left)).evalf();
#ifdef ZDEBUG
  cout << "Lz: " << L << "\n";
#endif // ZDEBUG
  int count = 0;
  ex Dtv;

  auto build_eq = [](ex f, ex L) {
    ex eq = L / f;
    return eq.evalf();
  };

  while (true) {
    ex Dt1 = build_eq(zdt, L), Dt2 = build_eq(zdt, -L);
#ifdef ZDEBUG
    cout << "Dt1z: " << Dt1 << " Dt2z: " << Dt2 << "\n";
#endif // ZDEBUG
    if (Dt1 > 0) {
      Dtv = Dt1;
    } else if (Dt2 > 0) {
      Dtv = Dt2;
    } else
      // XXX: This is the case where we have no solution at all!
      throw runtime_error("No real-positive root for z");

    // XXX: Now do the check and bound for the variables
    ex dtv = Dtv / R;
    bool err = var_compute(deps, dWts, vars, T, Dtv, dtv, toret);
    if (err) {
      step = ex_to<numeric>(Dtv.evalf()).to_double();
      break;
    } else {
      count += 1;
      if (count == iter_count)
        throw runtime_error("Too many iterations");
      L /= 2;
    }
  }
  return step;
}

double Solver::gstep(const ex &expr, const exT &deps, const exmap &vars,
                     const map<ex, vector<double>, ex_is_less> &dWts,
                     exmap &toret, double T) const {
  /**
   * Computes the step size using the guard
   */
#ifdef DEBUG
  cout << "guard expr: " << expr << "\n";
#endif
  double step = 0;
  symbol dt("Dt");

  // XXX: Now compute the ddeps
  exmap ddeps, dWt, ddWts;
  matrix dvars(1, vars.size());
  unsigned count = 0;
  for (const auto &v : vars) {
    symbol s = symbol{"d_" + ex_to<symbol>(v.first).get_name()};
    symbol s1 = symbol{"dWt_" + ex_to<symbol>(v.first).get_name()};
    ddeps[s] = deps.at(v.first).op(0) * dt + deps.at(v.first).op(1) * s1;
    dWt[v.first] = s1;
    ddWts[s1] =
        accumulate(dWts.at(v.first).begin(), dWts.at(v.first).end(), 0.0) *
        sqrt(dt / R);
    dvars(0, count) = s;
    ++count;
  }
  // XXX: The jacobian
  matrix jacobian(vars.size(), 1);
  count = 0;
  for (auto it = vars.begin(); it != vars.end(); ++it, ++count)
    jacobian(count, 0) = expr.diff(ex_to<symbol>(it->first));

#ifdef DEBUG
  cout << "the jacobian: " << jacobian << "\n";
  cout << "dvars: " << dvars << "\n";
#endif

  ex fp = (dvars * jacobian).evalm().op(0);
#ifdef DEBUG
  cout << "fp: " << fp << "\n";
#endif

  // XXX: The hessian
  matrix hessian(vars.size(), vars.size());
  unsigned i = 0, j = 0;
  for (auto it = vars.begin(); it != vars.end(); ++it, ++i) {
    j = 0;
    ex expr1 = expr.diff(ex_to<symbol>(it->first));
    for (auto it = vars.begin(); it != vars.end(); ++it, ++j) {
      hessian(i, j) = expr1.diff(ex_to<symbol>(it->first));
    }
  }
#ifdef DEBUG
  cout << "hessian:" << hessian << "\n";
#endif
  ex sp = 0.5 * (dvars * hessian * dvars.transpose()).evalm().op(0);
#ifdef DEBUG
  cout << "sp: " << sp << "\n";
#endif

  fp = fp.subs(ddeps);
  sp = sp.subs(ddeps);

  fp = fp.subs(vars);
  sp = sp.subs(vars);

  fp = fp.subs(symbol("t") == T);
  sp = sp.subs(symbol("t") == T);

  // XXX: Now expand and apply Ito's lemma
  fp = fp.expand();
  sp = sp.expand();

  // XXX: Now Ito's lemma

  // XXX: dt^2 == 0
  fp = fp.subs(pow(dt, 2) == 0);
  sp = sp.subs(pow(dt, 2) == 0);

  // XXX: Now the dwt*dt == 0
  for (const auto &i : dWt) {
    fp = fp.subs(i.second * dt * wild() == 0);
    fp = fp.subs(pow(i.second, 2) == dt);
    sp = sp.subs(i.second * dt * wild() == 0);
    sp = sp.subs(pow(i.second, 2) == dt);
  }

#ifdef DEBUG
  cout << "fp: " << fp << ", sp: " << sp << "\n";
#endif

  // XXX: Now substitute the uncorrelated Wiener processes.
  for (const auto &i : dWt)
    for (const auto &j : dWt)
      sp = i.first != j.first ? sp.subs(i.second * j.second * wild() == 0) : sp;

  // XXX: Now substitute the dWts.
  fp = fp.subs(ddWts).evalf();
  sp = sp.subs(ddWts).evalf();

#ifdef DEBUG
  cout << "fp: " << fp << ", sp: " << sp << "\n";
#endif
  // XXX: This is g[T]
  ex gv = expr.subs(vars);
  gv = gv.subs(symbol("t") == T);

  // Now compute the time step dt using the level crossing and the
  // quadratic
  if (fp + sp == 0) {
    toret = vars;
    return INF;
  }

  // XXX: Get the step-size within error constraints.
  ex L = gv;
  ex eq1, eq2, Dtv, root1, root2, dtv;
  bool err;
  count = 0;
  while (true) {
    root1 = build_eq_g(dt, fp, sp, L, T, eq1);
    root2 = build_eq_g(dt, fp, sp, -L, T, eq2);
#ifdef DEBUG
    cout << "Guard roots: " << root1 << "," << root2 << "\n";
#endif
    Dtv = min(root1, root2);
#ifdef DEBUG
    cout << "Guard Dtv: " << Dtv << "\n";
#endif
    step = ex_to<numeric>(Dtv.evalf()).to_double();
    if (step == INF) {
      toret = vars;
      break;
    }
    dtv = Dtv / R;
    err = var_compute(deps, dWts, vars, T, Dtv, dtv, toret);
    if (err) {
      step = ex_to<numeric>(Dtv.evalf()).to_double();
      break;
    } else {
      count += 1;
      if (count == iter_count)
        throw runtime_error("Too many iterations");
      L /= 2;
    }
  }
#ifdef DEBUG
  cout << "guard step returned: " << step << "\n";
  // exit(1);
#endif // DEBUG
  return step;
}

ex Solver::build_eq_g(const symbol &dt, const ex &fp, const ex &sp, const ex &L,
                      const double &T, ex &toret) const {
  ex root{INF}; // The default value
  ex f(fp + sp);
  f = f.expand().evalf();
#ifdef DEBUG
  cout << "build_eq_g, f: " << f << "\n";
#endif // DEBUG
  ex dtc = f.subs(sqrt(dt) * wild() == 0).collect(dt).coeff(dt, 1);
#ifdef DEBUG
  cout << "dtc: " << dtc << "\n";
#endif // DEBUG
  ex eq = pow((-dtc * dt - L), 2) - pow((f - dtc * dt), 2);
  eq = eq.expand().collect(dt);
  ex a = eq.coeff(dt, 2);
  ex b = eq.coeff(dt, 1);
  ex c = eq.coeff(dt, 0);
  ex D = pow(b, 2) - (4 * a * c);
#ifdef DEBUG
  cout << "a:" << a << ", b:" << b << ", D:" << D << ", c: " << c << "\n";
#endif // DEBUG
  if ((D >= 0) && (a != 0)) {
    ex root1 = (-b + sqrt(D)) / (2 * a);
    ex root2 = (-b - sqrt(D)) / (2 * a);
    if ((root1 > 0) && (root2 > 0))
      root = min(root1, root2);
    else if (root1 > 0)
      root = root1;
    else if (root2 > 0)
      root = root2;
    // XXX: One of them will be negative as such.
    // else
    // throw runtime_error("Could not find a real-positive root for guard");
  }
#ifdef DEBUG
  cout << "guard root: " << root << "\n";
#endif // DEBUG
  toret = eq;
  return root;
}

ex Solver::EM(const ex &init, const ex &f, const ex &g, const ex &Dt,
              const ex &dt, const std::vector<double> &dWts, const exmap &vars,
              const double T) const {
  return EMP(init, f, g, Dt, dt, std::accumulate(dWts.begin(), dWts.end(), 0),
             vars, T);
}

template <typename InputIt>
ex Solver::EM(const ex &init, const ex &f, const ex &g, const ex &Dt,
              const ex &dt, InputIt first, InputIt last, const exmap &vars,
              const double T) const {
  return EMP(init, f, g, Dt, dt, std::accumulate(first, last, 0), vars, T);
}

ex Solver::EMP(const ex &init, const ex &f, const ex &g, const ex &Dt,
               const ex &dt, double dWts_sum, const exmap &vars,
               const double T) const {
  ex res = 0;
  // Build the map for substitution
  ex f1 = f.subs(vars);
  ex f2 = f1.subs(symbol("t") == T);
  ex g1 = g.subs(vars);
  ex g2 = g1.subs(symbol("t") == T);
  res = (init + f2 * Dt + g2 * dWts_sum * sqrt(dt)).evalf();
  return res;
}
