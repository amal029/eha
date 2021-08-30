#include "./include/solver.hpp"
#include "boost/iterator/detail/facade_iterator_category.hpp"
#include "libInterpolate/Interpolate.hpp"
#include "libInterpolate/Interpolators/_1D/InterpolatorBase.hpp"
#include "libInterpolate/Interpolators/_1D/LinearInterpolator.hpp"
#include "matplotlibcpp.h"
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace GiNaC;

// namespace plt = matplotlibcpp;
// The enumeration for the states of the system
enum STATES { C = 0, K = 1, B = 2 };
typedef std::map<STATES, exT> derT;

// Initialize the random number generator
std::random_device rd{};
std::mt19937 gen{rd()}; // Usually use the random device rd()
// std::mt19937 gen{7};// Usually use the random device rd()

std::normal_distribution<> d{0, 1};

// Error constant
const double e = 1e-1;
const int d0 = 10, d1 = 7, d2 = 4, d3 = 1;
const double vx1 = 10, v2 = 5, a1 = 4;

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

double HIOA1(const symbol &x1, const symbol &x2, const symbol &v1,
             const derT &ders, const exmap &vars, const STATES &cs, STATES &ns,
             exmap &toret,
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
    } else {
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

int F(std::vector<std::vector<double>> &x1ss,
      std::vector<std::vector<double>> &x2ss,
      std::vector<std::vector<double>> &tss) {
  double SIM_TIME = 20;

  // Solver
  const Solver s{};
  s.DEFAULT_STEP = 1;
  s.u = 1e-3;

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
#ifdef PRINT
  plt::plot(ts, x1s);
  plt::plot(ts, x2s);
  plt::xlabel("Time (sec)");
  plt::ylabel("$x1(t), x2(t)$ (units)");
  plt::tight_layout();
  plt::show();
#endif // PRINT
#endif // TIME
  // XXX: Append for computing the CI
  x1ss.push_back(std::move(x1s));
  x2ss.push_back(std::move(x2s));
  tss.push_back(std::move(ts));
  return 0;
}

#ifdef TIME
#include <chrono>
#endif
int main(void) {
#ifdef TIME
  auto t1 = std::chrono::high_resolution_clock::now();
#endif // TIME
  std::vector<std::vector<double>> x1ss{}, x2ss{}, tss{};
  constexpr size_t N = 31;
  constexpr double tn = 1.96; // with 2 samples and df = 1
  for (auto i = 0; i < N; ++i)
    F(x1ss, x2ss, tss);

  // XXX: The simulation time
  constexpr size_t msize = 20;

  // XXX: Make the interpolators for only x2ss
  std::vector<_1D::LinearInterpolator<double>> interps1;
  std::vector<_1D::LinearInterpolator<double>> interps2;
  for (size_t i = 0; i < N; ++i) {
    // XXX: Initialise the first time
    interps1.push_back(_1D::LinearInterpolator<double>());
    interps2.push_back(_1D::LinearInterpolator<double>());
    // XXX: Use them
    interps1[i].setData(tss[i], x1ss[i]);
    interps2[i].setData(tss[i], x2ss[i]);
  }

  // XXX: We will go in 1 second intervals
  // XXX: The mean of the values
  std::vector<double> meanx1(msize, 0), meanx2(msize, 0);
  for (size_t i = 0; i < msize; ++i) {
    double ux1 = 0, ux2 = 0;
    for (size_t j = 0; j < N; ++j)
      ux1 += interps1[j](i), ux2 += interps2[j](i);
    ux1 = ux1 / N, ux2 = ux2 / N;
    // XXX: Means for each time point
    meanx1[i] = ux1, meanx2[i] = ux2;
  }
  // DEBUG
  // std::for_each(std::begin(meanx2), std::end(meanx2),
  //               [](double d) { std::cout << d << "\t"; });
  // std::cout << "\n";

  // XXX: Now the standard deviation
  std::vector<double> sigmax1(msize, 0), sigmax2(msize, 0);
  for (size_t i = 0; i < msize; ++i) {
    double sdx1 = 0, sdx2 = 0;
    for (size_t j = 0; j < N; ++j) {
      sdx1 += (meanx1[i] - interps1[j](i)) * (meanx1[i] - interps1[j](i));
      sdx2 += (meanx2[i] - interps2[j](i)) * (meanx2[i] - interps2[j](i));
    }
    sdx1 /= N, sdx2 /= N;
    // XXX: The standard deviations
    sdx1 = sqrt(sdx1), sdx2 = sqrt(sdx2);
    sigmax1[i] = sdx1, sigmax2[i] = sdx2;
  }
  // DEBUG
  // std::for_each(std::begin(sigmax2), std::end(sigmax2),
  //               [](double d) { std::cout << d << "\t"; });
  // std::cout << "\n";

  // XXX: Now we can do the confidence interval computation
  std::vector<double> x1CI{}, x2CI{};
  // XXX: Now the confidence interval at 95%
  for (size_t i = 0; i < msize; ++i) {
    x1CI.push_back(tn * sigmax1[i] / sqrt(N));
    x2CI.push_back(tn * sigmax2[i] / sqrt(N));
  }
  // DEBUG
  // std::for_each(std::begin(x2CI), std::end(x2CI),
  //               [](double d) { std::cout << d << "\t"; });
  // std::cout << "\n";

  // XXX: Finally mean ± CI
  std::vector<double> mplusCIx1(msize, 0), mminusCIx1(msize, 0),
      mplusCIx2(msize, 0), mminusCIx2(msize, 0);
  for (size_t i = 0; i < msize; ++i) {
    mplusCIx1[i] = meanx1[i] + x1CI[i];
    mminusCIx1[i] = meanx1[i] - x1CI[i];
    mplusCIx2[i] = meanx2[i] + x2CI[i];
    mminusCIx2[i] = meanx2[i] - x2CI[i];
  }
  std::vector<double> time(msize, 0);
  std::iota(time.begin(), time.end(), 0);
  // plt::plot(time, meanx2, {{"label", "mean"}});
  // plt::plot(time, mplusCIx2, {{"label", "CI 95% upper bound"}});
  // plt::plot(time, mminusCIx2, {{"label", "CI 95% lower bound"}});
  // plt::xlabel("Time (sec)");
  // plt::ylabel("$x1(t)$ (units)");
  // plt::grid();
  // plt::legend();
  // plt::tight_layout();
  // plt::savefig("twocarsx1.pdf", {{"bbox_inches", "tight"}});
  // plt::show();

  // plt::plot(time, meanx1, {{"label", "mean"}});
  // plt::plot(time, mplusCIx1, {{"label", "CI 95% upper bound"}});
  // plt::plot(time, mminusCIx1, {{"label", "CI 95% lower bound"}});
  // plt::xlabel("Time (sec)");
  // plt::ylabel("$x2(t)$ (units)");
  // plt::grid();
  // plt::legend();
  // plt::tight_layout();
  // plt::savefig("twocarsx2.pdf", {{"bbox_inches", "tight"}});
  // plt::show();

  // XXX: Write to the file
  const std::string fName = "twocars.csv";
  std::ofstream ostrm(fName);
  ostrm << "time,meanx1,mplusCIx1,mminusCIx1,meanx2,mplusCIx2,mminusCIx2\n";
  for (size_t i = 0; i < msize; ++i) {
    // XXX: Plot everything
    ostrm << time[i] << "," << meanx1[i] << "," << mplusCIx1[i] << ","
          << mminusCIx1[i] << "," << meanx2[i] << "," << mplusCIx2[i] << ","
          << mminusCIx2[i] << "\n";
  }
  ostrm.flush();
  ostrm.close();

#ifdef TIME
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout
      << "F() took "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
      << " milliseconds\n";
#endif // TIME
  return 0;
}
