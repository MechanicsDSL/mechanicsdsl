/*
 * MechanicsDSL - Python Extension (pybind11)
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>

namespace py = pybind11;

// {{PARAMETERS}}
const int DIM = {{STATE_DIM}};

void equations(const std::vector<double>& y, std::vector<double>& dydt, double t) {
{{STATE_UNPACK}}
{{EQUATIONS}}
}

void rk4_step(std::vector<double>& y, double t, double dt) {
    // ... Standard RK4 ...
    std::vector<double> k1(DIM), k2(DIM), k3(DIM), k4(DIM), temp_y(DIM), dydt(DIM);
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

std::vector<std::vector<double>> simulate(double t_max, double dt) {
    std::vector<double> y = { {{INITIAL_CONDITIONS}} };
    double t = 0.0;
    std::vector<std::vector<double>> results;
    
    while(t < t_max) {
        rk4_step(y, t, dt);
        t += dt;
        results.push_back(y); // Store state
    }
    return results;
}

PYBIND11_MODULE({{SYSTEM_NAME}}_ext, m) {
    m.doc() = "Fast C++ simulation for {{SYSTEM_NAME}}";
    m.def("simulate", &simulate, "Run simulation",
          py::arg("t_max") = 10.0, py::arg("dt") = 0.01);
}
