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

#define INF DBL_MAX

using namespace GiNaC;

typedef std::map<ex, lst, ex_is_less> exT;

struct Solver {
  double default_compute(const exT &deps, const exmap &vars,
                         const std::map<ex, std::vector<double>, ex_is_less> &dWts,
                         exmap &toret, double T = 0) const;

  double zstep(const ex &left, const lst &right, const exT &deps,
               const exmap &vars, double T,
               const std::map<ex, std::vector<double>, ex_is_less> &dWts, exmap &toret,
               double Uz = NAN) const;

  double gstep(const ex &expr, const exT &deps, const exmap &vars,
               const std::map<ex, std::vector<double>, ex_is_less> &dWts, exmap &toret,
               double T = 0) const;

  ex EM(const ex &init, const ex &f, const ex &g, const ex &Dt, const ex &dt,
        const std::vector<double> &dWts, const exmap &vars,
        const double T) const;

  static int p;
  static int R;
  static double DEFAULT_STEP;

private:
  // XXX: This will have the required private data
  const double ε = 1e-2;
  const int iter_count = 50;
  bool var_compute(const exT &deps,
                   const std::map<ex, std::vector<double>, ex_is_less> &dWts,
                   const exmap &vars, double T, ex Dtv, ex dtv,
                   exmap &toret) const;
  ex build_eq_g(const symbol &dt, const ex &fp, const ex &sp, const ex &L,
                const double &T, ex &toret) const;
};

int Solver::p = 3;
int Solver::R = std::pow(2, p);
double Solver::DEFAULT_STEP = 1.0;
