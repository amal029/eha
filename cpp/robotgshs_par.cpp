#include "matplotlibcpp.h"
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <ginac/ginac.h>
#include <iostream>
#include <numeric>
#include <random>

using namespace std;
using namespace GiNaC;

namespace plt = matplotlibcpp;

// The enumeration for the states of the system
enum STATES { MOVE = 0, INNER = 1, OUTTER = 2, CT = 3, NCT = 4 };

typedef std::map<ex, lst, ex_is_less> exT;
typedef std::map<STATES, exT> derT;

// Initialize the random number generator
std::random_device rd{};
std::mt19937 gen{rd()};
std::normal_distribution<> d{0, 1};

// The standard uniform distribution for jump edges
std::uniform_real_distribution<> dis(0.0, 1.0);

// Give the prints operation for the cout for derivatives
template <typename T>
ostream &operator<<(ostream &out, const std::map<ex, T> &var) {
  for (auto it = var.begin(); it != var.end(); ++it) {
    out << it->first << ":" << it->second << "\n";
  }
  return out;
}

// The constants
int v = 4;
double wv = 0.1;
double e = 1e-1;

struct Solver {
  double default_compute(const exT &deps, const exmap &vars,
                         const map<ex, vector<double>, ex_is_less> &dWts,
                         double T = 0) const {
    double step = 0;
    ex Dtv = DEFAULT_STEP;
    while (true) {
      ex dtv = Dtv / R;
      exmap toret;
      bool err = var_compute(deps, dWts, vars, T, Dtv, dtv, toret);
      if (err) {
        break;
      } else {
        Dtv /= 2;
      }
    }
    return step;
  }

  double zstep(const ex &left, const lst &right, const exT &deps,
               const exmap &vars, double T,
               const map<ex, vector<double>, ex_is_less> &dWts, exmap &toret,
               double Uz = NAN) const {
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
    if (isinf(Uz) || isnan(Uz)) {
      step = INFINITY;
      toret = vars;
      return step;
    }
    ex zdt = right.op(0);
    zdt = zdt.subs(vars);   // Substitution with current values
    zdt = zdt.subs(t == T); // Subs time t
    if (zdt == 0) {
      toret = vars;
      step = INFINITY;
      return step;
    }

    // XXX: Now compute the step size
    ex L = (Uz - vars.at(left));
    int count = 0;
    ex Dtv;

    auto build_eq = [](ex f, ex K) {
      ex eq = f / K;
      return eq.evalf();
    };

    while (true) {
      ex Dt1 = build_eq(zdt, L), Dt2 = build_eq(zdt, -L);
      if (Dt1 > 0) {
        Dtv = Dt1;
      } else if (Dt2 > 0) {
        Dtv = Dt2;
      } else {
        // XXX: This is the case where we have no solution at all!
        toret = vars;
        step = INFINITY;
        break;
      }
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

  bool var_compute(const exT &deps,
                   const map<ex, vector<double>, ex_is_less> &dWts,
                   const exmap &vars, double T, ex Dtv, ex dtv,
                   exmap &toret) const {
    bool err = false;
    exmap temp1, nvars;
    for (auto it = vars.begin(); it != vars.end(); ++it) {
      toret[it->first] =
          EM(it->second, deps.at(it->first).op(0), deps.at(it->first).op(1),
             Dtv, dtv, dWts.at(it->first), vars, T);
    }
    // XXX: Now compute the values in two half-steps.
    for (auto it = vars.begin(); it != vars.end(); ++it) {
      vector<double> v1(dWts.at(it->first).begin(),
                        dWts.at(it->first).begin() + R / 2);
      nvars[it->first] = EM(it->second, deps.at(it->first).op(0),
                            deps.at(it->first).op(1), Dtv, dtv, v1, vars, T);
    }
    for (auto it = nvars.begin(); it != nvars.end(); ++it) {
      vector<double> v1(dWts.at(it->first).begin() + R / 2,
                        dWts.at(it->first).end());
      nvars[it->first] = EM(it->second, deps.at(it->first).op(0),
                            deps.at(it->first).op(1), Dtv, dtv, v1, vars, T);
    }
    // XXX: Now do the final check
    vector<bool> errs;
    for (auto it = nvars.begin(); it != nvars.end(); ++it) {
      errs.push_back(abs(toret[it->first] -
                         nvars[it->first] / (nvars[it->first] + ε)) <= ε);
    }
    err = all_of(errs.begin(), errs.end(), [](bool i) { return i == true; });
    return err;
  }

  double gstep(const ex &expr, const exT &deps, const exmap &vars,
               const map<ex, vector<double>, ex_is_less> &dWts,
               double T = 0) const {
    /**
     * Computes the step size using the guard
     */
    double step = 0;
    symbol dt("dt");

    // XXX: Now compute the ddeps
    exmap ddeps, dWt;
    matrix dvars(1, vars.size());
    unsigned count = 0;
    for (const auto &v : vars) {
      symbol s = symbol{"d_" + ex_to<symbol>(v.first).get_name()};
      symbol s1 = symbol{"dWt_" + ex_to<symbol>(v.first).get_name()};
      ddeps[s] = deps.at(v.first).op(0) * dt + deps.at(v.first).op(1) * s1;
      dWt[v.first] = s1;
      dvars(0, count) = s;
      ++count;
    }

    // XXX: The jacobian
    matrix jacobian(vars.size(), 1);
    count = 0;
    for (auto it = vars.begin(); it != vars.end(); ++it, ++count)
      jacobian(count, 0) = expr.diff(ex_to<symbol>(it->first));

    ex fp = (dvars*jacobian).evalm();

    // XXX: The hessian
    matrix hessian(vars.size(), vars.size());
    unsigned i = 0, j = 0;
    for (auto it = vars.begin(); it != vars.end(); ++it, ++i) {
      ex expr1 = expr.diff(ex_to<symbol>(it->first));
      for (auto it = vars.begin(); it != vars.end(); ++it, ++j) {
        hessian(i, j) = expr1.diff(ex_to<symbol>(it->first));
      }
    }
    ex sp = 0.5*(dvars*hessian*dvars.transpose()).evalm();

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

    return step;
  }
  ex EM(const ex &init, const ex &f, const ex &g, const ex &Dt, const ex &dt,
        const std::vector<double> &dWts, const exmap &vars,
        const double T) const {
    ex res = 0;
    // Build the map for substitution
    ex f1 = f.subs(vars);
    ex f2 = f.subs(symbol("t") == T);
    ex g1 = g.subs(vars);
    ex g2 = g.subs(symbol("t") == T);
    res = (init + f2 * Dt +
           g2 * std::accumulate(dWts.begin(), dWts.end(), 0) * sqrt(dt))
              .evalf();
    return res;
  }

  static int p;
  static int R;

private:
  // XXX: This will have the required private data
  double ε = 1e-3;
  int iter_count = 50;
  double DEFAULT_STEP = 1;
};

int Solver::p = 3;
int Solver::R = std::pow(2, p);

// This is the robot x, y movement
double HIOA1(const symbol &x, const symbol &y, const symbol &z,
             const symbol &th, const derT &ders, const exmap &vars, bool &ft1,
             const STATES &cs, STATES &ns, exmap &toret,
             std::map<ex, vector<double>, ex_is_less> &dWts) {

  double step = 0;
  ex xval = vars.at(x), yval = vars.at(y), zval = vars.at(z);
  ex thval = vars.at(th);
  // XXX: The state machine
  switch (cs) {
  case MOVE: {
    if (xval * xval + yval * yval - v * v <= -e) {
      // XXX: Inter-transition
      ns = INNER;
      step = 0;
      ft1 = true;
      toret = vars;
    } else if (xval * xval + yval * yval - v * v >= e) {
      // XXX: Inter-transition
      ns = OUTTER;
      step = 0;
      ft1 = true;
      toret = vars;
    } else {
      // XXX: This is the Intra-transition
      ns = cs;
      ft1 = false;
    }
    break;
  }
  case INNER: {
    if ((xval * xval + yval * yval - v * v >= -e) &&
        (xval * xval + yval * yval - v * v <= e)) {
      ns = MOVE, ft1 = true, step = 0, toret = vars;
    } else if (xval * xval + yval * yval - v * v >= e) {
      ns = OUTTER, ft1 = true, step = 0, toret = vars;
    } else {
      ns = INNER, ft1 = false;
      cout << "inside INNER"
           << "\n";
      // XXX: Euler-Maruyama for step
    }
    break;
  }
  case OUTTER: {
    if ((xval * xval + yval * yval - v * v <= e) &&
        (xval * xval + yval * yval - v * v >= -e)) {
      ns = MOVE, ft1 = true, step = 0, toret = vars;
    } else if (xval * xval + yval * yval - v * v <= -e) {
      ns = INNER, ft1 = true, toret = vars, step = 0;
    } else {
      // XXX: Euler-Maruyama step
      ns = cs;
      ft1 = false;
    }
    break;
  }
  default:
    throw runtime_error("Unknown state entered for XY");
  }
  return step;
}

// This is the angle movement
double HIOA2(const symbol &x, const symbol &y, const symbol &z,
             const symbol &th, const derT &ders, const exmap &vars, bool &ft2,
             STATES &cs, STATES &ns, exmap &toret,
             std::map<ex, vector<double>, ex_is_less> &dWts) {

  double step = 0;
  ex xval = vars.at(x), yval = vars.at(y), zval = vars.at(z);
  ex thval = vars.at(th);
  switch (cs) {
  case CT: {
    cout << "Inside CT"
         << "\n";
    static double Uz = 0.0;
    Uz = ft2 ? -log(dis(gen)) : Uz;
    if ((xval * xval + yval * yval - v * v <= e) &&
        (xval * xval + yval * yval - v * v >= -e)) {
      ns = NCT;
      ft2 = true;
      step = 0;
      toret = vars;
      toret[z] = 0;
    } else if (abs(Uz - zval) <= e) {
      ns = CT;
      toret = vars;
      toret[z] = 0;
      toret[th] = thval - Uz;
      ft2 = true;
      step = 0;
    } else {
      ns = cs;
      ft2 = false;
      // XXX: Euler-Maruyama compute step
    }
    break;
  }
  case NCT: {
    if ((xval * xval + yval * yval - v * v <= -e) ||
        (xval * xval + yval * yval - v * v >= e)) {
      step = 0;
      ns = CT;
      ft2 = true;
      toret = vars;
      toret[z] = 0;
      toret[th] = atan(yval / xval);
    } else {
      ns = cs;
      ft2 = false;
      // XXX: Compute step using EM
    }
    break;
  }
  default:
    throw runtime_error("Unknown state entered for θ");
  }
  return step;
}

int main(void) {
  // Initialize the random seed
  srand(0);

  // The variables in the system
  symbol x("x"), y("y"), z("z"), th("th");

  // Initialise a map of the derivatives
  std::map<STATES, std::map<ex, lst, ex_is_less>> ders;
  ders[MOVE] = {{x, {-1 * v * wv * sin(th), ex{0}}},
                {y, {v * wv * cos(th), ex{0}}},
                {th, {ex{0}, ex{0}}},
                {z, {ex{0}, ex{0}}}};
  ders[INNER] = {{x, {v * cos(th), ex{v}}},
                 {y, {v * sin(th), ex{v}}},
                 {th, {ex{0}, ex{0}}},
                 {z, {ex{0}, ex{0}}}};
  ders[OUTTER] = {{x, {v * cos(th), ex{v}}},
                  {y, {v * sin(th), ex{v}}},
                  {th, {ex{0}, ex{0}}},
                  {z, {ex{0}, ex{0}}}};
  ders[CT] = {{x, {ex{0}, ex{0}}},
              {y, {ex{0}, ex{0}}},
              {th, {ex{2 * v}, ex{0}}},
              {z, {x * x + y * y + th * th * th, ex{0}}}};
  ders[NCT] = {{x, {ex{0}, ex{0}}},
               {y, {ex{0}, ex{0}}},
               {th, {ex{wv}, ex{0}}},
               {z, {ex{0}, ex{0}}}};

  // Just printing the derivatives for each state in the system
  // for (int i = MOVE; i <= NCT; ++i)
  //   cout << i << "->" << ders[(STATES)i] << "\n";

  // The intial values
  ex xval = 1;
  ex yval = 1;
  ex zval = 0;
  ex thval = atan(yval / xval);

  cout << xval << " " << yval << " " << zval << "\n";
  cout << thval << "\n";


  // The variable map
  exmap vars;

  // Now call the HIOA with the initial values
  bool ft1 = true;
  bool ft2 = true;
  double time = 0;

  // XXX: The simulation end time
  double SIM_TIME = 1.2;

  // The current and the next state variables
  STATES cs1, cs2, ns1, ns2;

  // XXX: This is returning a copy of the vector tor
  auto randn = [](std::vector<double> &tor) {
    for (auto it = tor.begin(); it != tor.end(); ++it)
      *it = d(gen);
  };

  // Set the current state
  if (abs(xval * xval + yval * yval - v * v) <= e)
    cs1 = MOVE;
  else if (xval * xval + yval * yval - v * v <= -e)
    cs1 = INNER;
  else if (xval * xval + yval * yval - v * v >= e)
    cs1 = OUTTER;
  else
    throw runtime_error("Do not know the starting state for cs1");

  cs2 = CT; // always startinng from CT

  // XXX: These are the values returned by the HIOAs
  exmap toret1{{x, 0.0}, {y, 0}, {z, 0}, {th, 0}};
  exmap toret2 = toret1;

  // XXX: Plotting vectors
  std::vector<double> xs{ex_to<numeric>(xval).to_double()};
  std::vector<double> ys{ex_to<numeric>(yval).to_double()};
  std::vector<double> zs{ex_to<numeric>(zval).to_double()};
  std::vector<double> ths{ex_to<numeric>(thval.evalf()).to_double()};
  std::vector<double> ts{0};

  cout << "H1"
       << "\n";

  // Now run until completion
  std::map<ex, std::vector<double>, ex_is_less> dWts;

  // print the state here
  cout << "Entering the loop"
       << "\n";
  while (time <= SIM_TIME) {

    // Generate the sample path for the Euler-Maruyama step
    dWts[x] = std::vector<double>(Solver::R, 0);
    randn(dWts[x]); // Updating the vector sample path
    dWts[y] = std::vector<double>(Solver::R, 0);
    randn(dWts[y]);
    dWts[th] = std::vector<double>(Solver::R, 0);
    randn(dWts[th]);
    dWts[z] = std::vector<double>(Solver::R, 0);

    // XXX: Set the variables
    vars[x] = xval;
    vars[y] = yval;
    vars[z] = zval;
    vars[th] = thval;

    cout << vars[x] << " " << vars[y] << " " << vars[z] << " " << vars[th] << "\n";

    // Calling the HIOAs
    double d1 = HIOA1(x, y, z, th, ders, vars, ft1, cs1, ns1, toret1, dWts);
    double d2 = HIOA2(x, y, z, th, ders, vars, ft2, cs2, ns2, toret2, dWts);

    // The lockstep execution step
    if (!ft1 && !ft2) {
      // XXX: Intra-Intra
      if (d2 <= d1) {
        // XXX: Compute Euler-Maruyama
      } else if (d1 <= d2) {
        // XXX: Compute Euler-Maruyama
      }
    } else if (!ft1 && ft2) {
      // XXX: Intra-Inter
      thval = toret2[th];
      zval = toret2[z];
    } else if (ft1 and !ft2) {
      // XXX: Inter-Intra
      xval = toret1[x];
      yval = toret1[y];
    } else if (ft1 && ft2) {
      // XXX: Inter-Inter
      xval = toret1[x];
      yval = toret1[y];
      thval = toret2[th];
      zval = toret2[z];
    }

    // XXX: Set the next state
    cs1 = ns1;
    cs2 = ns2;

    // XXX: Increment the timer
    time += std::min(d1, d2);

    // FIXME: DEBUG
    time += 0.5;
    ts.push_back(time);

    // XXX: Append to plot later on
    xs.push_back(ex_to<numeric>(vars[x].evalf()).to_double());
    ys.push_back(ex_to<numeric>(vars[y].evalf()).to_double());
    zs.push_back(ex_to<numeric>(vars[z].evalf()).to_double());
    ths.push_back(ex_to<numeric>(vars[th].evalf()).to_double());
  }

  // XXX: Now we can plot the values
  std::vector<double> xy2(xs.size(), 0);
  for (int i = 0; i < xs.size(); ++i)
    xy2[i] = xs[i]*xs[i] + ys[i]*ys[i];

  plt::plot(ts, xy2);
  plt::show();

  return 0;
}
