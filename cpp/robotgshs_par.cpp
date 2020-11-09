#include "./include/solver.hpp"
#include "matplotlibcpp.h"

using namespace std;
using namespace GiNaC;

namespace plt = matplotlibcpp;

// The enumeration for the states of the system
enum STATES { MOVE = 0, INNER = 1, OUTTER = 2, CT = 3, NCT = 4 };

typedef std::map<ex, lst, ex_is_less> exT;
typedef std::map<STATES, exT> derT;

// Initialize the random number generator
// std::random_device rd{};
// std::mt19937 gen{rd()}; // Usually use the random device rd()
std::mt19937 gen{7}; // Usually use the random device rd()
std::normal_distribution<> d{0, 1};

// The standard uniform distribution for jump edges
std::uniform_real_distribution<> dis(0.0, 1.0);

// The constants
int v = 4;
double wv = 0.1;
double e = 1e-1;

double __compute(const exmap &vars,
                 const std::map<ex, vector<double>, ex_is_less> &dWts,
                 const derT &ders, const STATES location, const lst&& guards,
                 const Solver &s, exmap &toret, double t = 0,
                 const lst &&zs = {}, double Uz = NAN) {
  double T = 0.0;
  // XXX: Now call the rate and the guard computation for from the
  // solver.
  exT DM(ders.at(location));
  std::map<double, exmap> Dts;
  exmap toretz;
  for(auto const &z: zs){
    double Dz = s.zstep(z, DM[z], DM, vars, t, dWts, toretz, Uz);
    Dts[Dz] = std::move(toretz);
  }
  exmap toretg; // This will be passed back
  for (const auto &i : guards) {
    double Dt = s.gstep(i, DM, vars, dWts, toretg, t);
    Dts[Dt] = std::move(toretg);
  }
  // XXX: Now get the smallest step size
  vector<double> k;
  for (const auto &i : Dts) {
    k.push_back(i.first);
  }
  T = (k.size() > 1) ? *std::min_element(k.begin(), k.end())
                     : k.size() > 0 ? k[0] : INF;
  if (T == INF) {
    T = s.default_compute(DM, vars, dWts, toret, t);
  } else
    toret = std::move(Dts[T]);
  return T;
}

// This is the robot x, y movement
double HIOA1(const symbol &x, const symbol &y, const symbol &z,
             const symbol &th, const derT &ders, const exmap &vars, bool &ft1,
             const STATES &cs, STATES &ns, exmap &toret,
             const std::map<ex, vector<double>, ex_is_less> &dWts,
             const Solver &s, const double time) {

  double step = 0;
  ex xval = vars.at(x), yval = vars.at(y), zval = vars.at(z);
  ex thval = vars.at(th);
  // XXX: The state machine
  switch (cs) {
  case MOVE: {
    if (xval * xval + yval * yval - v * v <= -e) {
      // XXX: Inter-transition
      ns = INNER, step = 0;
      ft1 = true, toret = vars;
    } else if (xval * xval + yval * yval - v * v >= e) {
      // XXX: Inter-transition
      ns = OUTTER, step = 0;
      ft1 = true, toret = vars;
    } else {
      // XXX: This is the Intra-transition
      ns = cs, ft1 = false;
      step = __compute(vars, dWts, ders, cs, {}, s, toret, time);
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
      ns = cs, ft1 = false;
      ex g = pow(x, 2) + pow(y, 2) - std::pow(v, 2);
      // XXX: Euler-Maruyama for step
      step = __compute(vars, dWts, ders, cs, {g}, s, toret, time);
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
      ex g = pow(x, 2) + pow(y, 2) - std::pow(v, 2);
      // XXX: Euler-Maruyama for step
      step = __compute(vars, dWts, ders, cs, {g}, s, toret, time);
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
             std::map<ex, vector<double>, ex_is_less> &dWts, const Solver &s,
             const double time) {

  double step = 0;
  ex xval = vars.at(x), yval = vars.at(y), zval = vars.at(z);
  ex thval = vars.at(th);
  switch (cs) {
  case CT: {
    static double Uz = 0.0;
    Uz = ft2 ? -log(dis(gen)) : Uz;
    if ((xval * xval + yval * yval - v * v <= e) &&
        (xval * xval + yval * yval - v * v >= -e)) {
      ns = NCT, ft2 = true;
      step = 0, toret = vars;
      toret[z] = 0, toret[th] = atan(yval / xval);
    } else if (abs(Uz - zval).evalf() <= e) {
      ns = CT, toret = vars;
      toret[z] = 0, toret[th] = thval - Uz;
      ft2 = true, step = 0;
    } else {
      ns = cs, ft2 = false;
      // XXX: Euler-Maruyama compute step
      ex g = pow(x, 2) + pow(y, 2) - std::pow(v, 2);
      // XXX: Euler-Maruyama for step
      step = __compute(vars, dWts, ders, cs, {g}, s, toret, time, {z}, Uz);
      // cout << "step: " << step << " toret2:" << toret << "\n";
    }
    break;
  }
  case NCT: {
    if ((xval * xval + yval * yval - v * v <= -e) ||
        (xval * xval + yval * yval - v * v >= e)) {
      step = 0, ns = CT;
      ft2 = true, toret = vars;
      toret[z] = 0, toret[th] = atan(yval / xval);
    } else {
      ns = cs, ft2 = false;
      // XXX: Compute step using EM
      // XXX: Euler-Maruyama for step
      step = __compute(vars, dWts, ders, cs, {}, s, toret, time);
    }
    break;
  }
  default:
    throw runtime_error("Unknown state entered for θ");
  }
  return step;
}

int F(int argc, char **argv) {
  double SIM_TIME = 1.2;
  Solver::DEFAULT_STEP = 1e-1;
  if (argc > 1)
    SIM_TIME = std::atof(argv[1]);

  // XXX: Initialise the solver
  const Solver s{};

  // The variables in the system
  symbol x("x"), y("y"), z("z"), th("th");

  // Initialise a map of the derivatives
  std::map<STATES, std::map<ex, lst, ex_is_less>> ders;
  ders[MOVE] = {{x, {v * wv * -sin(th), ex{0}}},
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

  // The intial values
  ex xval = 1, yval = 1, zval = 0, thval = atan(yval / xval);

  // The variable map
  exmap vars;

  // Now call the HIOA with the initial values
  bool ft1 = true, ft2 = true;
  double time = 0;

  // The current and the next state variables
  STATES cs1, cs2, ns1, ns2;

  auto randn = [](std::vector<double> &tor) {
    for (auto it = tor.begin(); it != tor.end(); ++it)
      *it = d(gen);
  };

  // Set the current state
  if (abs(xval * xval + yval * yval - v * v).evalf() <= e)
    cs1 = MOVE;
  else if (xval * xval + yval * yval - v * v <= -e)
    cs1 = INNER;
  else if (xval * xval + yval * yval - v * v >= e)
    cs1 = OUTTER;
  else
    throw runtime_error("Do not know the starting state for cs1");

  cs2 = CT; // always startinng from CT

  // XXX: These are the values returned by the HIOAs
  exmap toret1{{x, 0.0}, {y, 0}, {z, 0}, {th, 0}}, toret2 = toret1;

  // XXX: Plotting vectors
  std::vector<double> xs{ex_to<numeric>(xval).to_double()};
  std::vector<double> ys{ex_to<numeric>(yval).to_double()};
  std::vector<double> zs{ex_to<numeric>(zval).to_double()};
  std::vector<double> ths{ex_to<numeric>(thval.evalf()).to_double()};
  std::vector<double> ts{0};

  // Now run until completion
  std::map<ex, std::vector<double>, ex_is_less> dWts;

  // XXX: Print the values of variables at time
  auto tostate = [](STATES s) -> std::string {
    switch (s) {
    case 0:
      return "MOVE";
    case 1:
      return "INNER";
    case 2:
      return "OUTTER";
    case 3:
      return "CT";
    case 4:
      return "NCT";
    default:
      break;
    }
  };
  // print the state here
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

#ifndef TIME
    cout << time << ":"
         << " L1: " << tostate(cs1) << " L2: " << tostate(cs2) << " ";
    std::for_each(std::begin(vars), std::end(vars), [](const auto &i) {
      cout << i.first << ":" << i.second << "  ";
    });
    cout << "\n";
#endif // TIME

    // Calling the HIOAs
    double d1 =
        HIOA1(x, y, z, th, ders, vars, ft1, cs1, ns1, toret1, dWts, s, time);
    double d2 =
        HIOA2(x, y, z, th, ders, vars, ft2, cs2, ns2, toret2, dWts, s, time);

    // The lockstep execution step
    if (!ft1 && !ft2) {
      // XXX: Intra-Intra
      if (d2 <= d1) {
        double T = d2;
        // XXX: Compute Euler-Maruyama
        xval = s.EM(vars[x], ders[cs1][x].op(0), ders[cs1][x].op(1), ex(T),
                    ex(T) / s.R, dWts[x], vars, time);
        yval = s.EM(vars[y], ders[cs1][y].op(0), ders[cs1][y].op(1), ex(T),
                    ex(T) / s.R, dWts[y], vars, time);
        thval = toret2[th], zval = toret2[z];
        // cout << "d2 <= d1 " << toret2 << "\n";
      } else if (d1 <= d2) {
        double T = d1;
        xval = toret1[x], yval = toret1[y];
        // XXX: Compute Euler-Maruyama
        thval = s.EM(vars[th], ders[cs2][th].op(0), ders[cs2][th].op(1), ex(T),
                     ex(T) / s.R, dWts[th], vars, time);
        zval = s.EM(vars[z], ders[cs2][z].op(0), ders[cs2][z].op(1), ex(T),
                    ex(T) / s.R, dWts[z], vars, time);
      }
    } else if (!ft1 && ft2) {
      // XXX: Intra-Inter
      thval = toret2[th], zval = toret2[z];
    } else if (ft1 and !ft2) {
      // XXX: Inter-Intra
      xval = toret1[x], yval = toret1[y];
    } else if (ft1 && ft2) {
      // XXX: Inter-Inter
      xval = toret1[x], yval = toret1[y], thval = toret2[th];
      zval = toret2[z];
    }

    // XXX: Set the next state
    cs1 = ns1, cs2 = ns2;

    // XXX: Increment the timer
    time += std::min(d1, d2);
    ts.push_back(time);

    // XXX: Append to plot later on
    xs.push_back(ex_to<numeric>(vars[x].evalf()).to_double());
    ys.push_back(ex_to<numeric>(vars[y].evalf()).to_double());
    zs.push_back(ex_to<numeric>(vars[z].evalf()).to_double());
    ths.push_back(ex_to<numeric>(vars[th].evalf()).to_double());
  }
#ifndef TIME
  cout << "TOTAL SIM COUNT: " << ts.size() << "\n";
#endif // TIME

  // XXX: Now we can plot the values
  std::vector<double> xy2(xs.size(), 0);
  for (int i = 0; i < xs.size(); ++i)
    xy2[i] = xs[i] * xs[i] + ys[i] * ys[i];

#ifndef TIME
  plt::plot(ts, xy2);
  plt::show();
#endif // TIME
  return 0;
}

#ifdef TIME
#include <chrono>
#endif // TIME
int main(int argc, char *argv[])
{
#ifdef TIME
  auto t1 = std::chrono::high_resolution_clock::now();
#endif // TIME
  F(argc, argv);
#ifdef TIME
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout
      << "F() took "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
      << " milliseconds\n";
#endif // TIME
  return 0;
}
