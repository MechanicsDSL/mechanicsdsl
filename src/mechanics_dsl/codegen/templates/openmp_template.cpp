/*
 * MechanicsDSL - OpenMP Parameter Sweep
 * System: {{SYSTEM_NAME}}
 * * Scans a range of initial conditions or parameters in parallel.
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <omp.h>

using std::sin; using std::cos; using std::tan; 
using std::exp; using std::log; using std::sqrt;
using std::pow; using std::abs;

// Base Parameters
// {{PARAMETERS}}

const int DIM = {{STATE_DIM}};

// Equations (Pass parameters if sweeping them)
void equations(const std::vector<double>& y, std::vector<double>& dydt, double t) {
{{STATE_UNPACK}}
{{EQUATIONS}}
}

void rk4_step(std::vector<double>& y, double t, double dt) {
    std::vector<double> k1(DIM), k2(DIM), k3(DIM), k4(DIM), temp_y(DIM);
    std::vector<double> dydt(DIM);
    // ... Standard RK4 implementation ...
    equations(y, dydt, t);
    for(int i=0; i<DIM; i++) k1[i] = dt * dydt[i];
    for(int i=0; i<DIM; i++) temp_y[i] = y[i] + 0.5 * k1[i];
    equations(temp_y, dydt, t + 0.5 * dt);
    for(int i=0; i<DIM; i++) k2[i] = dt * dydt[i];
    for(int i=0; i<DIM; i++) temp_y[i] = y[i] + 0.5 * k2[i];
    equations(temp_y, dydt, t + 0.5 * dt);
    for(int i=0; i<DIM; i++) k3[i] = dt * dydt[i];
    for(int i=0; i<DIM; i++) temp_y[i] = y[i] + k3[i];
    equations(temp_y, dydt, t + dt);
    for(int i=0; i<DIM; i++) k4[i] = dt * dydt[i];
    for(int i=0; i<DIM; i++) y[i] += (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]) / 6.0;
}

int main() {
    // SWEEP CONFIGURATION
    const int N_STEPS = 1000;
    double start_val = 0.0;
    double end_val = 3.14;
    double step_size = (end_val - start_val) / N_STEPS;

    std::ofstream outfile("sweep_results.csv");
    outfile << "param,final_state\n";

    // PARALLEL LOOP
    #pragma omp parallel for
    for(int i=0; i<N_STEPS; i++) {
        double param = start_val + i * step_size;
        
        // Setup individual simulation
        // Modifying initial condition y[0] based on sweep parameter
        std::vector<double> y = { {{INITIAL_CONDITIONS}} };
        y[0] = param; // OVERRIDE initial position with sweep param

        double t = 0;
        double dt = 0.01;
        double t_max = 10.0;

        while(t < t_max) {
            rk4_step(y, t, dt);
            t += dt;
        }

        // Critical section for file writing
        #pragma omp critical
        {
            outfile << param << "," << y[0] << "\n";
        }
    }
    
    std::cout << "Parallel sweep complete." << std::endl;
    return 0;
}
