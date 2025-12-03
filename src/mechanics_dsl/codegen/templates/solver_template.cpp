#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>

// Use standard math functions
using std::sin; using std::cos; using std::tan; 
using std::exp; using std::log; using std::sqrt;
using std::pow; using std::abs;

// {{PARAMETERS}}

// State dimension
const int DIM = {{STATE_DIM}};

// Equations of Motion
void equations(const std::vector<double>& y, std::vector<double>& dydt, double t) {
{{STATE_UNPACK}}
    
{{EQUATIONS}}
}

// RK4 Solver Step
void rk4_step(std::vector<double>& y, double t, double dt) {
    std::vector<double> k1(DIM), k2(DIM), k3(DIM), k4(DIM), temp_y(DIM);
    std::vector<double> dydt(DIM);

    // k1
    equations(y, dydt, t);
    for(int i=0; i<DIM; i++) k1[i] = dt * dydt[i];

    // k2
    for(int i=0; i<DIM; i++) temp_y[i] = y[i] + 0.5 * k1[i];
    equations(temp_y, dydt, t + 0.5 * dt);
    for(int i=0; i<DIM; i++) k2[i] = dt * dydt[i];

    // k3
    for(int i=0; i<DIM; i++) temp_y[i] = y[i] + 0.5 * k2[i];
    equations(temp_y, dydt, t + 0.5 * dt);
    for(int i=0; i<DIM; i++) k3[i] = dt * dydt[i];

    // k4
    for(int i=0; i<DIM; i++) temp_y[i] = y[i] + k3[i];
    equations(temp_y, dydt, t + dt);
    for(int i=0; i<DIM; i++) k4[i] = dt * dydt[i];

    // Update
    for(int i=0; i<DIM; i++) {
        y[i] += (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]) / 6.0;
    }
}

int main() {
    // Initial Conditions
    std::vector<double> y = { {{INITIAL_CONDITIONS}} };
    double t = 0.0;
    double dt = 0.001; // Fixed step size - consider making this configurable
    double t_end = 10.0;
    int steps = static_cast<int>(t_end / dt);
    int log_interval = 10; // Log every 10 steps to save space

    std::ofstream file("{{SYSTEM_NAME}}_results.csv");
    file << "{{CSV_HEADER}}\n";
    file << std::fixed << std::setprecision(6);

    std::cout << "Simulating {{SYSTEM_NAME}}..." << std::endl;

    for(int step=0; step<=steps; step++) {
        if(step % log_interval == 0) {
            file << t;
            for(double val : y) file << "," << val;
            file << "\n";
        }

        rk4_step(y, t, dt);
        t += dt;
    }

    std::cout << "Simulation complete. Data saved to {{SYSTEM_NAME}}_results.csv" << std::endl;
    return 0;
}
