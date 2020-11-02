#include <iostream>
#include <ginac/ginac.h>
#include <cstdlib>
#include <exception>
#include <random>
#include "matplotlibcpp.h"
#include <numeric>

using namespace std;
using namespace GiNaC;

namespace plt = matplotlibcpp;

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

// Information about the Wiener process
int p = 2;
int R = std::pow(2, p);

// The enumeration for the states of the system
enum STATES { MOVE = 0, INNER = 1, OUTTER = 2, CT = 3, NCT = 4 };

// The constants
int v = 4;
double wv = 0.1;
double e = 1e-1;

struct Solver {
  double zstep() const {
    /**
     * Computes the step size using the jump edges
     */
    double step = 0;
    // XXX: fill in the algorithm
    return step;
  }

  double gstep() const {
    /**
     * Computes the step size using the guard
     */
    double step = 0;
    return step;
  }
  double EM(double init, ex f, ex g, double Dt, double dt,
            const std::vector<double> &dWts,
            const std::map<ex, double, ex_is_less> &vars,
            const double T) const {
    double res = 0;
    // Build the map for substitution
    exmap v;
    for (auto it = vars.begin(); it != vars.end(); ++it) {
      v[it->first] = ex{it->second};
    }
    v[symbol("t")] = ex{T};
    f = f.subs(v);
    g = g.subs(v);
    res = ex_to<numeric>(
              (init + f * Dt +
               g * std::accumulate(dWts.begin(), dWts.end(), 0) * std::sqrt(dt))
                  .evalf())
              .to_double();
    return res;
  }

private:
  // XXX: This will have the required provate data
  double ε = 1e-3;
  int iter_count = 50;
  double DEFAULT_STEP = 1;
  int p = 3;
  int R = std::pow(2, p);
};

// This is the robot x, y movement
double HIOA1(const symbol &x, const symbol &y, const symbol &z,
             const symbol &th,
             const std::map<STATES, std::map<string, lst>> &ders,
             std::map<ex, double, ex_is_less> &vars, bool &ft1,
             const STATES &cs, STATES &ns,
             std::map<ex, double, ex_is_less> &toret,
             std::map<string, vector<double>> &dWts) {

  double step = 0;
  double xval = vars[x];
  double yval = vars[y];
  double thval = vars[th];
  double zval = vars[z];
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
    cout << "inside INNER"
         << "\n";
    if ((xval * xval + yval * yval - v * v >= -e) &&
        (xval * xval + yval * yval - v * v <= e)) {
      ns = MOVE;
      ft1 = true;
      step = 0;
      toret = vars;
    } else if (xval * xval + yval * yval - v * v >= e) {
      ns = OUTTER;
      ft1 = true;
      step = 0;
      toret = vars;
    } else {
      ns = INNER;
      ft1 = false;
      // XXX: Euler-Maruyama for step
    }
    break;
  }
  case OUTTER: {
    if ((xval * xval + yval * yval - v * v <= e) &&
        (xval * xval + yval * yval - v * v >= -e)) {
      ns = MOVE;
      ft1 = true;
      step = 0;
      toret = vars;
    } else if (xval * xval + yval * yval - v * v <= -e) {
      ns = INNER;
      ft1 = true;
      toret = vars;
      step = 0;
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
             const symbol &th,
             const std::map<STATES, std::map<string, lst>> &ders,
             std::map<ex, double, ex_is_less> &vars, bool &ft2, STATES &cs,
             STATES &ns, std::map<ex, double, ex_is_less> &toret,
             std::map<string, vector<double>> &dWts) {

  double step = 0;
  double xval = vars[x];
  double yval = vars[y];
  double thval = vars[th];
  double zval = vars[z];
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
    } else if (abs(zval - Uz) <= e) {
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
  std::map<STATES, std::map<string, lst>> ders;
  ders[MOVE] = {{x.get_name(), {-1 * v * wv * sin(th), ex{0}}},
                {y.get_name(), {v * wv * cos(th), ex{0}}},
                {th.get_name(), {ex{0}, ex{0}}},
                {z.get_name(), {ex{0}, ex{0}}}};
  ders[INNER] = {{x.get_name(), {v * cos(th), ex{v}}},
                 {y.get_name(), {v * sin(th), ex{v}}},
                 {th.get_name(), {ex{0}, ex{0}}},
                 {z.get_name(), {ex{0}, ex{0}}}};
  ders[OUTTER] = {{x.get_name(), {v * cos(th), ex{v}}},
                  {y.get_name(), {v * sin(th), ex{v}}},
                  {th.get_name(), {ex{0}, ex{0}}},
                  {z.get_name(), {ex{0}, ex{0}}}};
  ders[CT] = {{x.get_name(), {ex{0}, ex{0}}},
              {y.get_name(), {ex{0}, ex{0}}},
              {th.get_name(), {ex{2 * v}, ex{0}}},
              {z.get_name(), {x * x + y * y + th * th * th, ex{0}}}};
  ders[NCT] = {{x.get_name(), {ex{0}, ex{0}}},
               {y.get_name(), {ex{0}, ex{0}}},
               {th.get_name(), {ex{wv}, ex{0}}},
               {z.get_name(), {ex{0}, ex{0}}}};

  // Just printing the derivatives for each state in the system
  // for (int i = MOVE; i <= NCT; ++i)
  //   cout << i << "->" << ders[(STATES)i] << "\n";

  // The intial values
  double xval = 1;
  double yval = 1;
  double zval = 0;
  double thval = atan(yval / xval);

  // The variable map
  std::map<ex, double, ex_is_less> vars;

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

  // print the state here
  cout << "Entering the loop"
       << "\n";

  // XXX: These are the values returned by the HIOAs
  std::map<ex, double, ex_is_less> toret1{{x, 0.0}, {y, 0}, {z, 0}, {th, 0}};
  std::map<ex, double, ex_is_less> toret2 = toret1;

  // XXX: Plotting vectors
  std::vector<double> xs{xval};
  std::vector<double> ys{yval};
  std::vector<double> zs{zval};
  std::vector<double> ths{thval};
  std::vector<double> ts{0};

  // Now run until completion
  std::map<string, std::vector<double>> dWts;
  while (time <= SIM_TIME) {

    // Generate the sample path for the Euler-Maruyama step
    dWts[x.get_name()] = std::vector<double>(R, 0);
    randn(dWts[x.get_name()]); // Updating the vector sample path
    dWts[y.get_name()] = std::vector<double>(R, 0);
    randn(dWts[y.get_name()]);
    dWts[th.get_name()] = std::vector<double>(R, 0);
    randn(dWts[th.get_name()]);
    dWts[z.get_name()] = std::vector<double>(R, 0);

    // XXX: Set the variables
    vars[x] = xval;
    vars[y] = yval;
    vars[z] = zval;
    vars[th] = thval;

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
    xs.push_back(vars[x]);
    ys.push_back(vars[y]);
    zs.push_back(vars[z]);
    ths.push_back(vars[th]);
  }

  // XXX: Now we can plot the values
  std::vector<double> xy2(xs.size(), 0);
  for (int i = 0; i < xs.size(); ++i)
    xy2[i] = xs[i]*xs[i] + ys[i]*ys[i];

  plt::plot(ts, xy2);
  plt::show();

  return 0;
}
