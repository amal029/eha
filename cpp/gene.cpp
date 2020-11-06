#include "./include/solver.hpp"
#include "matplotlibcpp.h"

using namespace GiNaC;

namespace plt = matplotlibcpp;
// The enumeration for the states of the system
enum STATES { X0 = 0, X1 = 1, D = 2 };
typedef std::map<STATES, exT> derT;

// Initialize the random number generator
std::random_device rd{};
std::mt19937 gen{rd()}; // Usually use the random device rd()
// std::mt19937 gen{rd()}; // Usually use the random device rd()

std::normal_distribution<> d{0, 1};

// The standard uniform distribution for jump edges
std::uniform_real_distribution<> dis(0, 1);

// Error constant
const double e = 1e-1;
const int kp = 1;
const float ku = 0.001, kd = 0.01, kb = 0.01;

double __compute(const exmap &vars,
                 const std::map<ex, std::vector<double>, ex_is_less> &dWts,
                 const derT &ders, const STATES location, lst guards,
                 const Solver &s, exmap &toret, double t = 0,
                 const symbol *z = nullptr, double Uz = NAN) {
  double T = 0.0;
  // XXX: Now call the rate and the guard computation for from the
  // solver.
  exT DM(ders.at(location));
  std::map<double, exmap> Dts;
  if (z != nullptr) {
    exmap toretz;
    double Dz = s.zstep(*z, DM[*z], DM, vars, t, dWts, toretz, Uz);
    Dts[Dz] = std::move(toretz);
  }
  for (const auto &i : guards) {
    exmap toretg; // This will be passed back
    double Dt = s.gstep(i, DM, vars, dWts, toretg, t);
    Dts[Dt] = std::move(toretg);
  }
  // XXX: Now get the smallest step size
  std::vector<double> k;
  for (const auto &i : Dts) {
    k.push_back(i.first);
  }
  T = (k.size() > 1) ? *std::min_element(k.begin(), k.end())
                     : k.size() > 0 ? k[0] : INF;
  if (T == INF) {
    T = s.default_compute(DM, vars, dWts, toret, t);
  } else {
    toret = std::move(Dts[T]);
  }
  return T;
}

double HIOA1(const symbol &x, const symbol &z, const derT &ders,
             const exmap &vars, const STATES &cs, STATES &ns, exmap &toret,
             const std::map<ex, std::vector<double>, ex_is_less> &dWts,
             const Solver &s, const double time, bool &ft1) {
  double step = 0;
  ex xval = vars.at(x);
  ex zval = vars.at(z);
  static double Uz = 0.0;
  Uz = ft1 ? -log(dis(gen)) : Uz;
  switch (cs) {
  case X0: {
    if ((xval >= 1) && abs(zval - Uz) <= e) {
      ft1 = true;
      ns = X1, toret = vars, toret[z] = 0, step = 0;
    } else {
      ns = cs, ft1 = false;
      step = __compute(vars, dWts, ders, cs, {}, s, toret, time, &z, Uz);
    }
    break;
  }
  case X1: {
    if (abs(zval - Uz) <= e) {
      ns = X0, step = 0, toret = vars, toret[z] = 0;
    } else {
      ns = cs, ft1 = false;
      step = __compute(vars, dWts, ders, cs, {}, s, toret, time, &z, Uz);
    }
    break;
  }
  default:
    throw std::runtime_error("Entered an unknown state for HIOA1");
  }
  return step;
}

double HIOA2(const symbol &x, const symbol &z, const derT &ders,
             const exmap &vars, const STATES &cs, STATES &ns, exmap &toret,
             const std::map<ex, std::vector<double>, ex_is_less> &dWts,
             const Solver &s, const double time, bool &ft2) {
  double step = 0;
  switch (cs) {
  case D: {
    ns = cs, ft2 = false;
    step = __compute(vars, dWts, ders, cs, {}, s, toret, time);
    break;
  }
  default:
    throw std::runtime_error("Unknown location for HIOA2");
  }
  return step;
}

int main(int argc, char *argv[]) {
  double SIM_TIME = 2000;
  Solver::DEFAULT_STEP = 1;
  Solver::ε = 1e-3;

  // Solver
  const Solver s{};

  // Symbols
  symbol x("x"), z("z");

  std::map<STATES, std::map<ex, lst, ex_is_less>> ders;
  ders[X0] = {{x, {kp, 0}}, {z, {kb * x, 0}}};
  ders[X1] = {{x, {0, 0}}, {z, {ku, 0}}};
  ders[D] = {{x, {-kd * x, 0}}, {z, {0, 0}}};

  // initial values
  ex xval = 0, zval = 0;

  // vars
  exmap vars{{x, xval}, {z, zval}};

  // time
  double time = 0;

  // states
  STATES cs1, ns1, cs2, ns2;

  auto randn = [](std::vector<double> &tor) {
    for (auto it = tor.begin(); it != tor.end(); ++it)
      *it = d(gen);
  };

  // initialise the state
  cs1 = X0;
  cs2 = D;

  // XXX: These are the values returned by the HIOAs
  exmap toret1{{x, xval}, {z, zval}}, toret2 = toret1;

  // Plotting vectors
  std::vector<double> xs{ex_to<numeric>(toret1[x]).to_double()};
  std::vector<double> zs{ex_to<numeric>(toret1[z]).to_double()};
  std::vector<double> ts{time};

  // Now run until completion
  std::map<ex, std::vector<double>, ex_is_less> dWts;
  dWts[x] = std::vector<double>(Solver::R, 0);
  dWts[z] = std::vector<double>(Solver::R, 0);

  auto tostate = [](STATES s) -> std::string {
    switch (s) {
    case X0:
      return "X0";
    case X1:
      return "X1";
    case D:
      return "D";
    }
  };
  bool ft1 = true, ft2 = true;
  // Now run the simulation
  while (time <= SIM_TIME) {
    // Generate the sample path for the Euler-Maruyama step
    randn(dWts[x]); // Updating the vector sample path

    // XXX: Set the variables
    vars[x] = xval;
    vars[z] = zval;

    // Calling the HIOAs
    double d1 = HIOA1(x, z, ders, vars, cs1, ns1, toret1, dWts, s, time, ft1);
    double d2 = HIOA2(x, z, ders, vars, cs2, ns2, toret2, dWts, s, time, ft2);

    // Update the next state
    cs1 = ns1, cs2 = ns2;

    // XXX: Update the variables.
    if (!ft1 && !ft2) {
      double T;
      ex x1, x2;
      if (d2 <= d1) {
        T = d2;
        x1 = s.EM(vars[x], ders[cs1][x].op(0), ders[cs1][x].op(1), T, (T / s.R),
                  dWts[x], vars, time);
        zval = s.EM(vars[z], ders[cs1][z].op(0), ders[cs1][z].op(1), T,
                    (T / s.R), dWts[z], vars, time);
	x2 = toret2[x];
      } else if (d1 <= d2) {
        T = d1;
        x2 = s.EM(vars[x], ders[cs2][x].op(0), ders[cs2][x].op(1), T, (T / s.R),
                  dWts[x], vars, time);
	x1 = toret1[x];
	zval = toret1[z];
      }
      // Combine
      xval = x1 + x2 - vars[x];
    } else if (ft1 && !ft2) {
      zval = toret1[z], xval = toret1[x];
    } else
      throw std::runtime_error("HIOA2 cannot take a transition");

    // XXX: Increment the timer
    time += std::min(d1, d2);
    ts.push_back(time);

    // XXX: Append to plot later on
    xs.push_back(ex_to<numeric>(xval.evalf()).to_double());
    zs.push_back(ex_to<numeric>(zval.evalf()).to_double());

    // Print things
    std::cout << time << ":"
              << " L1: " << tostate(cs1) << " L2:" << tostate(cs2) << " ";
    std::for_each(std::begin(vars), std::end(vars), [](const auto &i) {
      std::cout << i.first << ":" << i.second << "  ";
    });
    std::cout << "\n";
  }
  std::cout << "TOTAL SIM COUNT: " << ts.size() << "\n";

  // Plot
  plt::plot(ts, xs);
  plt::show();
  return 0;
}
