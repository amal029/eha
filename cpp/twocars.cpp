#include "./include/solver.hpp"
#include "matplotlibcpp.h"

using namespace GiNaC;

namespace plt = matplotlibcpp;
// The enumeration for the states of the system
enum STATES { C = 0, K = 1, B = 2 };
typedef std::map<STATES, exT> derT;

// Initialize the random number generator
std::random_device rd{};
std::mt19937 gen{rd()}; // Usually use the random device rd()

std::normal_distribution<> d{0, 1};

// Error constant
const double e = 1e-1;
const int d0 = 10, d1 = 7, d2 = 4, d3 = 1;
const double vx1 = 10, v2 = 5, a1 = 4;

double __compute(const exmap &vars,
                 const std::map<ex, std::vector<double>, ex_is_less> &dWts,
                 const derT &ders, const STATES location, const lst&& guards,
                 const Solver &s, exmap &toret, double t = 0,
                 const lst &&zs = {}, double Uz = NAN) {
  double T = 0.0;
  // XXX: Now call the rate and the guard computation for from the
  // solver.
  exT DM(ders.at(location));
  std::map<double, exmap> Dts;
  exmap toretz;
  for(auto const &z : zs){
    double Dz = s.zstep(z, DM[z], DM, vars, t, dWts, toretz, Uz);
    Dts[Dz] = std::move(toretz);
  }
  exmap toretg; // This will be passed back
  for (const auto &i : guards) {
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

double HIOA1(const symbol &x1, const symbol &x2, const symbol &v1, const derT &ders,
             const exmap &vars, const STATES &cs, STATES &ns, exmap &toret,
             const std::map<ex, std::vector<double>, ex_is_less> &dWts,
             const Solver &s, const double time) {
  double step = 0;
  ex x1val = vars.at(x1);
  ex x2val = vars.at(x2);
  switch (cs) {
  case C: {
    if (abs((x2val - x1val) - d2).evalf() <= e) {
      ns = K, toret = vars, step = 0;
    } else {
      ns = cs;
      ex g = x2 - x1 - d2;
      step = __compute(vars, dWts, ders, cs, {g}, s, toret, time);
    }
    break;
  }
  case K: {
    if (abs((x2val - x1val) - d1).evalf() <= e) {
      ns = C, step = 0, toret = vars;
    } else if (abs((x2val - x1val) - d3).evalf() <= e) {
      ns = B, toret = vars, step = 0;
    }
    else {
      ns = cs;
      ex g1 = x2 - x1 - d1, g2 = x2 - x1 - d3;
      step = __compute(vars, dWts, ders, cs, {g1, g2}, s, toret, time);
    }
    break;
  }
  case B: {
    if (abs(x2val - x1val - d0) <= e) {
      ns = C, step = 0, toret = vars;
    } else {
      ns = cs;
      ex g = x2 - x1 - d0;
      step = __compute(vars, dWts, ders, cs, {g}, s, toret, time);
    }
    break;
  }
  default:
    throw std::runtime_error("Entered an unknown state for HIOA");
  }
  return step;
}

int F(void) {
  double SIM_TIME = 20;

  // Solver
  const Solver s{};
  s.DEFAULT_STEP = 1;
  s.ε = 1e-3;

  // Symbols
  symbol x1("x1"), x2("x2"), v1("v1");

  std::map<STATES, std::map<ex, lst, ex_is_less>> ders;
  ders[C] = {{x1, {vx1, 1}}, {x2, {v2, 0}}, {v1, {0, 0}}};
  ders[K] = {{x1, {v2, 1}}, {x2, {v2, 0}}, {v1, {0, 0}}};
  ders[B] = {{x1, {v1, 0}}, {x2, {v2, 0}}, {v1, {-a1, 0}}};

  // initial values
  ex x1val = 1, x2val = 6, v1val = 0.5;

  // vars
  exmap vars{{x1, x1val}, {x2, x2val}, {v1, v1val}};

  // time
  double time = 0;

  // states
  STATES cs1, ns1;

  auto randn = [](std::vector<double> &tor) {
    for (auto it = tor.begin(); it != tor.end(); ++it)
      *it = d(gen);
  };

  // initialise the state
  if ((x2val - x1val) > d2)
    cs1 = C;
  else if (((x2val - x1val) > d3) && ((x2val - x1val) < d1))
    cs1 = K;
  else
    cs1 = B;

  // XXX: These are the values returned by the HIOAs
  exmap toret1{{x1, x1val}, {x2, x2val}, {v1, v1val}}; 

  // Plotting vectors
  std::vector<double> x1s{ex_to<numeric>(toret1[x1]).to_double()};
  std::vector<double> x2s{ex_to<numeric>(toret1[x2]).to_double()};
  std::vector<double> v1s{ex_to<numeric>(toret1[v1]).to_double()};
  std::vector<double> ts{time};

  // Now run until completion
  std::map<ex, std::vector<double>, ex_is_less> dWts;
  dWts[x1] = std::vector<double>(Solver::R, 0);
  dWts[x2] = std::vector<double>(Solver::R, 0);
  dWts[v1] = std::vector<double>(Solver::R, 0);

  auto tostate = [](STATES s) -> std::string {
    switch (s) {
    case C:
      return "C";
    case K:
      return "K";
    case B:
      return "B";
    }
  };

  // Now run the simulation
  while (time <= SIM_TIME) {
    // Generate the sample path for the Euler-Maruyama step
    randn(dWts[x1]); // Updating the vector sample path

    // XXX: Set the variables
    vars[x1] = x1val, vars[x2] = x2val, vars[v1] = v1val;

    // Calling the HIOAs
    double d = HIOA1(x1, x2, v1, ders, vars, cs1, ns1, toret1, dWts, s, time);

    // Update the next state
    cs1 = ns1;

    // Update the variables
    x1val = toret1[x1];
    x2val = toret1[x2];
    v1val = toret1[v1];

    // XXX: Increment the timer
    time += d;
    ts.push_back(time);

    // XXX: Append to plot later on
    x1s.push_back(ex_to<numeric>(x1val.evalf()).to_double());
    x2s.push_back(ex_to<numeric>(x2val.evalf()).to_double());
    v1s.push_back(ex_to<numeric>(v1val.evalf()).to_double());

#ifndef TIME
    // Print things
    std::cout << time << ":"
              << " L1: " << tostate(cs1) << " ";
    std::for_each(std::begin(vars), std::end(vars), [](const auto &i) {
      std::cout << i.first << ":" << i.second << "  ";
    });
    std::cout << "\n";
#endif // TIME
  }

#ifndef TIME
  std::cout << "TOTAL SIM COUNT: " << ts.size() << "\n";
  // Plot
  // plt::style("ggplot");
  plt::plot(ts, x1s);
  plt::plot(ts, x2s);
  plt::xlabel("Time (sec)");
  plt::ylabel("$x1(t), x2(t)$ (units)");
  plt::tight_layout();
  plt::show();
#endif // TIME
  return 0;
}

#ifdef TIME
#include <chrono>
#endif
int main(void)
{
#ifdef TIME
  auto t1 = std::chrono::high_resolution_clock::now();
#endif // TIME
  F();
#ifdef TIME
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout
    << "F() took "
    << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
    << " milliseconds\n";
#endif // TIME
  return 0;
}

