#include "./include/solver.hpp"
#include "matplotlibcpp.h"

using namespace GiNaC;

namespace plt = matplotlibcpp;
// The enumeration for the states of the system
enum STATES { S1 = 0, S2 = 1, S3 = 2 };
typedef std::map<STATES, exT> derT;

// Initialize the random number generator
std::random_device rd{};
// std::mt19937 gen{4907}; // Usually use the random device rd()
std::mt19937 gen{rd()}; // Usually use the random device rd()

std::normal_distribution<> d{0, 1};

// The standard uniform distribution for jump edges
std::uniform_real_distribution<> dis(0.0, 1.0);

// Error constant
const double e = 1e-1;

double __compute(const exmap &vars,
                 const std::map<ex, std::vector<double>, ex_is_less> &dWts,
                 const derT &ders, const STATES location, const lst&& guards,
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

double HIOA(const symbol &x, const derT &ders, const exmap &vars,
            const STATES &cs, STATES &ns, exmap &toret,
            const std::map<ex, std::vector<double>, ex_is_less> &dWts,
            const Solver &s, const double time) {
  double step = 0;
  ex xval = vars.at(x);
  switch (cs) {
  case S1: {
    if (abs(cos(xval)).evalf() <= e) {
      ns = S3, toret = vars, step = 0;
    } else if (cos(xval) >= e) {
      ns = S2, step = 0, toret = vars;
    } else {
      ns = cs;
      ex g = cos(x);
      step = __compute(vars, dWts, ders, cs, {g}, s, toret, time);
    }
    break;
  }
  case S2: {
    if (abs(cos(xval)).evalf() <= e) {
      ns = S3, step = 0, toret = vars;
    } else if (cos(xval) <= -e) {
      ns = S1, step = 0, toret = vars;
    } else {
      ex g = cos(x);
      ns = cs;
      step = __compute(vars, dWts, ders, cs, {g}, s, toret, time);
    }
    break;
  }
  case S3: {
    ns = cs;
    step = __compute(vars, dWts, ders, cs, {}, s, toret, time);
    break;
  }
  default:
    throw std::runtime_error("Entered an unknown state!");
  }
  return step;
}

int F(void) {
  double SIM_TIME = 20;
  Solver::DEFAULT_STEP = 1;
  Solver::ε = 1e-5;

  // Solver
  const Solver s{};

  // Symbols
  symbol x("x");

  std::map<STATES, std::map<ex, lst, ex_is_less>> ders;
  ders[S1] = {{x, {-0.01, 1}}};
  ders[S2] = {{x, {0.01, -1}}};
  ders[S3] = {{x, {0, 0}}};

  // initial values
  ex xval = 0.5;

  // vars
  exmap vars {{x, xval}};

  // time
  double time = 0;

  // states
  STATES cs, ns;

  auto randn = [](std::vector<double> &tor) {
    for (auto it = tor.begin(); it != tor.end(); ++it)
      *it = d(gen);
  };

  // initialise the state
  if (xval < (Pi / 2).evalf())
    cs = S2;
  else
    cs = S1;

  // XXX: These are the values returned by the HIOAs
  exmap toret{{x, xval}};

  // Plotting vectors
  std::vector<double> xs{ex_to<numeric>(toret[x]).to_double()};
  std::vector<double> ts{time};

  // Now run until completion
  std::map<ex, std::vector<double>, ex_is_less> dWts;
  dWts[x] = std::vector<double>(Solver::R, 0);

  auto tostate = [](STATES s) -> std::string {
    switch (s) {
    case S1:
      return "S1";
    case S2:
      return "S2";
    case S3:
      return "S3";
    default:
      break;
    }
  };
  // Now run the simulation
  while (time <= SIM_TIME) {
    // Generate the sample path for the Euler-Maruyama step
    randn(dWts[x]); // Updating the vector sample path

    // XXX: Set the variables
    vars[x] = xval;

    // Calling the HIOAs
    double d = HIOA(x, ders, vars, cs, ns, toret, dWts, s, time);

    // Update the next state
    cs = ns;

    // Udpate the values
    xval = toret[x];

    // XXX: Increment the timer
    time += d;
    ts.push_back(time);

    // XXX: Append to plot later on
    xs.push_back(ex_to<numeric>(vars[x].evalf()).to_double());

#ifndef TIME
    // Print things
    std::cout << time << ":"
              << " L: " << tostate(cs) << " ";
    std::for_each(std::begin(vars), std::end(vars), [](const auto &i) {
      std::cout << i.first << ":" << i.second << "  ";
    });
    std::cout << "\n";
#endif // TIME
  }
#ifndef TIME
  // std::cout << ts[ts.size()-1] << "," << xs[xs.size()-1] << "\n";
  std::cout << "TOTAL SIM COUNT: " << ts.size() << "\n";
  // Plot
  plt::plot(ts, xs);
  plt::show();
#endif // TIME
  return 0;
}

#ifdef TIME
#include <chrono>
#endif
int main(int argc, char *argv[])
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
