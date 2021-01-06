#include "./include/solver.hpp"
#include "libInterpolate/Interpolate.hpp"
#include "matplotlibcpp.h"
#include <cstddef>
#include <vector>

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
                 const derT &ders, const STATES location, const lst &&guards,
                 const Solver &s, exmap &toret, double t = 0,
                 const lst &&zs = {}, double Uz = NAN) {
  double T = 0.0;
  // XXX: Now call the rate and the guard computation for from the
  // solver.
  exT DM(ders.at(location));
  std::map<double, exmap> Dts;
  exmap toretz;
  for (auto const &z : zs) {
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
      : k.size() > 0 ? k[0]
                     : INF;
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

int F(std::vector<std::vector<double>> &xss,
      std::vector<std::vector<double>> &tss) {
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
  exmap vars{{x, xval}};

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
  xss.push_back(xs);
  tss.push_back(ts);
#ifdef PRINT
  // Plot
  plt::plot(ts, xs);
  plt::ylabel("$x(t)$ (units)");
  plt::xlabel("Time (sec)");
  plt::tight_layout();
  plt::show();
#endif // PRINT
#endif // TIME
  return 0;
}

#ifdef TIME
#include <chrono>
#endif
int main(int argc, char *argv[]) {
#ifdef TIME
  auto t1 = std::chrono::high_resolution_clock::now();
#endif // TIME
  std::vector<std::vector<double>> x1ss{}, tss{};
  constexpr size_t N = 31;
  constexpr double tn = 1.96; // with 2 samples and df = 1
  constexpr double msize = 20;
  for (size_t i = 0; i < N; ++i)
    F(x1ss, tss);
  // XXX: Make the interpolators for only x2ss
  std::vector<_1D::LinearInterpolator<double>> interps1;
  for (size_t i = 0; i < N; ++i) {
    // XXX: Initialise the first time
    interps1.push_back(_1D::LinearInterpolator<double>());
    // XXX: Use them
    interps1[i].setData(tss[i], x1ss[i]);
  }
  // XXX: We will go in 1 second intervals
  // XXX: The mean of the values
  std::vector<double> meanx1(msize, 0);
  for (size_t i = 0; i < msize; ++i) {
    double ux1 = 0;
    for (size_t j = 0; j < N; ++j)
      ux1 += interps1[j](i);
    ux1 = ux1 / N;
    // XXX: Means for each time point
    meanx1[i] = ux1;
  }
  // XXX: Now the standard deviation
  std::vector<double> sigmax1(msize, 0);
  for (size_t i = 0; i < msize; ++i) {
    double sdx1 = 0;
    for (size_t j = 0; j < N; ++j) {
      sdx1 += (meanx1[i] - interps1[j](i)) * (meanx1[i] - interps1[j](i));
    }
    sdx1 /= N;
    // XXX: The standard deviations
    sdx1 = sqrt(sdx1);
    sigmax1[i] = sdx1;
  }
  // XXX: Now we can do the confidence interval computation
  std::vector<double> x1CI{};
  // XXX: Now the confidence interval at 95%
  for (size_t i = 0; i < msize; ++i) {
    x1CI.push_back(tn * sigmax1[i] / sqrt(N));
  }
  // XXX: Finally mean ± CI
  std::vector<double> mplusCIx1(msize, 0), mminusCIx1(msize, 0);
  for (size_t i = 0; i < msize; ++i) {
    mplusCIx1[i] = meanx1[i] + x1CI[i];
    mminusCIx1[i] = meanx1[i] - x1CI[i];
  }
  // XXX: Now plot it
  std::vector<double> time(msize, 0);
  std::iota(time.begin(), time.end(), 0);
  plt::plot(time, meanx1, {{"label", "mean"}});
  plt::plot(time, mplusCIx1, {{"label", "CI 95% upper bound"}});
  plt::plot(time, mminusCIx1, {{"label", "CI 95% lower bound"}});
  plt::xlabel("Time (seconds)");
  plt::ylabel("$x(t)$ (units)");
  plt::grid();
  plt::legend();
  plt::tight_layout();
  plt::savefig("driver.pdf", {{"bbox_inches", "tight"}});
  plt::show();

#ifdef TIME
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout
      << "F() took "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
      << " milliseconds\n";
#endif // TIME
  return 0;
}
