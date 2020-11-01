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
double HIOA1(const symbol &x, const symbol &y, const symbol &z, const symbol &th,
	    const std::map<STATES, std::map<string, lst>> &ders,
	    std::map<string, double> &vals, bool &ft1, const STATES &cs,
	    STATES &ns) {

  double step = 0;
  ft1 = false;

  return step;
}

double HIOA2(const symbol &x, const symbol &y, const symbol &z, const symbol &th,
	    const std::map<STATES, std::map<string, lst>> &ders,
	    std::map<string, double> &vals, bool &ft2,
	    STATES &cs, STATES &ns) {

  double step = 0;
  ft2 = false;

  return step;
}


int main() {
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
	      {th.get_name(), {ex{2*v}, ex{0}}},
	      {z.get_name(), {x*x+y*y+th*th*th, ex{0}}}};
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
  double thval = atan(yval/xval);
  // The variable map
  std::map<string, double> vars{{x.get_name(), xval},
			       {y.get_name(), yval},
			       {z.get_name(), zval},
                               {th.get_name(), thval}};
  // Now call the HIOA with the initial values
  bool ft1 = true;
  bool ft2 = true;
  double time = 0;
  double SIM_TIME = 1.2;

  // The current and the next state variables
  STATES cs1, cs2, ns1, ns2;

  // Initialize the random number generator
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<> d{0, 1};

  // XXX: This is returning a copy of the vector tor
  auto randn = [&gen, &d](std::vector<double> &tor){
    for (auto it = tor.begin(); it != tor.end(); ++it)
      *it = d(gen);
  };

  // Set the current state
  if (abs(xval*xval + yval*yval - v*v) <= e)
    cs1 = MOVE;
  else if(xval*xval + yval*yval - v*v <= -e)
    cs1 = INNER;
  else if(xval*xval + yval*yval - v*v >= e)
    cs1 = OUTTER;
  else
    throw runtime_error("Do not know the starting state for cs1");

  cs2 = CT;			// always startinng from CT

  // print the state here

  // Now run until completion
  while (time <= SIM_TIME) {

    // Generate the sample path for the Euler-Maruyama step
    std::map<string, std::vector<double>> dWts;
    dWts[x.get_name()] = std::vector<double>(R, 0);
    randn(dWts[x.get_name()]);	// Updating the vector sample path
    dWts[y.get_name()] = std::vector<double>(R, 0);
    randn(dWts[y.get_name()]);
    dWts[th.get_name()] = std::vector<double>(R, 0);
    randn(dWts[z.get_name()]);
    dWts[z.get_name()] = std::vector<double>(R, 0);

    // Calling the HIOAs
    double d1 = HIOA1(x, y, z, th, ders, vars, ft1, cs1, ns1);
    double d2 = HIOA2(x, y, z, th, ders, vars, ft2, cs2, ns2);


    // You will need to increment the timer here at the end
    time += std::min(d1, d2);
  }

  return 0;
}
