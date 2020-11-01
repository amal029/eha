#include <iostream>
#include <ginac/ginac.h>
#include <cstdlib>
#include <exception>
#include <random>

using namespace std;
using namespace GiNaC;

// Give the prints operation for the cout for derivatives
ostream& operator << (ostream &out, const std::map<string, lst> &var) {
  for (auto it = var.begin(); it != var.end(); ++it) {
    out << it->first << ":" << it->second << "\n";
  }
  return out;
}

// Information about the Wiener process
int p = 2;
int R = std::pow(2, p);

// The enumeration for the states of the system
enum STATES {MOVE=0, INNER=1, OUTTER=2, CT=3, NCT=4};

// The constants
int v = 4;
double wv = 0.1;
double e = 1e-1;

// Now the HIOA itself.
double HIOA1(const symbol &x, const symbol &y, const symbol &z,
             const symbol &th,
             const std::map<STATES, std::map<string, lst>> &ders,
             std::map<string, double> &vars, bool &ft1, const STATES &cs,
             STATES &ns, std::map<string, double> &toret) {

  double step = 0;
  ft1 = false;

  return step;
}

double HIOA2(const symbol &x, const symbol &y, const symbol &z,
             const symbol &th,
             const std::map<STATES, std::map<string, lst>> &ders,
             std::map<string, double> &vars, bool &ft2, STATES &cs, STATES &ns,
             std::map<string, double> &toret) {

  double step = 0;
  ft2 = false;

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
  std::map<string, double> vars;

  // Now call the HIOA with the initial values
  bool ft1 = true;
  bool ft2 = true;
  double time = 0;

  // XXX: The simulation end time
  double SIM_TIME = 1.2;

  // The current and the next state variables
  STATES cs1, cs2, ns1, ns2;

  // Initialize the random number generator
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<> d{0, 1};

  // XXX: This is returning a copy of the vector tor
  auto randn = [&gen, &d](std::vector<double> &tor) {
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
  std::map<string, double> toret1{{"x", 0.0}, {"y", 0}, {"z", 0}, {"th", 0}};
  std::map<string, double> toret2 = toret1;

  // XXX: Plotting vectors
  std::list<double> xs{xval};
  std::list<double> ys{yval};
  std::list<double> zs{zval};
  std::list<double> ths{thval};

  // Now run until completion
  while (time <= SIM_TIME) {

    // Generate the sample path for the Euler-Maruyama step
    std::map<string, std::vector<double>> dWts;
    dWts[x.get_name()] = std::vector<double>(R, 0);
    randn(dWts[x.get_name()]); // Updating the vector sample path
    dWts[y.get_name()] = std::vector<double>(R, 0);
    randn(dWts[y.get_name()]);
    dWts[th.get_name()] = std::vector<double>(R, 0);
    randn(dWts[th.get_name()]);
    dWts[z.get_name()] = std::vector<double>(R, 0);

    // XXX: Set the variables
    vars["x"] = xval;
    vars["y"] = yval;
    vars["z"] = zval;
    vars["th"] = thval;

    // Calling the HIOAs
    double d1 = HIOA1(x, y, z, th, ders, vars, ft1, cs1, ns1, toret1);
    double d2 = HIOA2(x, y, z, th, ders, vars, ft2, cs2, ns2, toret2);

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
      thval = toret2["th"];
      zval = toret2["z"];
    }
    else if (ft1 and !ft2){
      // XXX: Inter-Intra
      xval = toret1["x"];
      yval = toret1["y"];
    }
    else if (ft1 && ft2){
     // XXX: Inter-Inter 
      xval = toret1["x"];
      yval = toret1["y"];
      thval = toret2["th"];
      zval = toret2["z"];
    }

    // XXX: Set the next state
    cs1 = ns1;
    cs2 = ns2;

    // XXX: Increment the timer
    time += std::min(d1, d2);

    //XXX: Append to plot later on
    xs.push_back(vars["x"]);
    ys.push_back(vars["y"]);
    zs.push_back(vars["z"]);
    ths.push_back(vars["th"]);
  }

  // XXX: Now we can plot the values

  return 0;
}
