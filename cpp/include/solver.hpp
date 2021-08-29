/**
 * This is the header file for the stochastic differential equation solver.
 * @author: Avinash Malik
 * @date: Thu  5 Nov 2020 19:59:35 NZDT
 */
#pragma once

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <float.h>
#include <ginac/ginac.h>
#include <iostream>
#include <numeric>
#include <random>
#include <iterator>

#define INF DBL_MAX

// using namespace GiNaC;
namespace G = GiNaC;

typedef std::map<G::ex, G::lst, G::ex_is_less> exT;

struct Solver {
  double default_compute(
      const exT &deps, const G::exmap &vars,
      const std::map<G::ex, std::vector<double>, G::ex_is_less> &dWts,
      G::exmap &toret, double T = 0) const;

  double zstep(const G::ex &left, const G::lst &right, const exT &deps,
               const G::exmap &vars, double T,
               const std::map<G::ex, std::vector<double>, G::ex_is_less> &dWts,
               G::exmap &toret, double Uz = NAN) const;

  double gstep(const G::ex &expr, const exT &deps, const G::exmap &vars,
               const std::map<G::ex, std::vector<double>, G::ex_is_less> &dWts,
               G::exmap &toret, double T = 0) const;

  G::ex EM(const G::ex &init, const G::ex &f, const G::ex &g, const G::ex &Dt,
           const G::ex &dt, const std::vector<double> &dWts,
           const G::exmap &vars, const double T) const;

  template <typename InputIt>
  G::ex EM(const G::ex &init, const G::ex &f, const G::ex &g, const G::ex &Dt,
           const G::ex &dt, InputIt first, InputIt last, const G::exmap &vars,
           const double T) const;

  static int p;
  static int R;
  static double DEFAULT_STEP;
  static double u; 
private:
  // XXX: This will have the required private data
  const int iter_count = 50;
  bool
  var_compute(const exT &deps,
              const std::map<G::ex, std::vector<double>, G::ex_is_less> &dWts,
              const G::exmap &vars, double T, const G::ex &Dtv, const G::ex &dtv,
              G::exmap &toret) const;

  G::ex build_eq_g(const G::symbol &dt, const G::ex &fp, const G::ex &sp,
                   const G::ex &L, const double &T) const;

  G::ex EMP(const G::ex &init, const G::ex &f, const G::ex &g, const G::ex &Dt,
            const G::ex &dt, double dWtss, const G::exmap &vars,
            const double T) const;
};

int Solver::p = 3;
int Solver::R = std::pow(2, p);
double Solver::u = 1e-3;
double Solver::DEFAULT_STEP = 1.0;
