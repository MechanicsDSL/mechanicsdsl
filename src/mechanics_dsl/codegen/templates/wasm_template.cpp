/*
 * MechanicsDSL - WebAssembly Export
 * Build with: emcc simulation.cpp -o simulation.js -s EXPORTED_FUNCTIONS='["_init", "_step", "_get_state"]' -s EXPORTED_RUNTIME_METHODS='["ccall", "cwrap"]'
 */
#include <vector>
#include <cmath>
#include <emscripten.h>

// Use standard math functions
using std::sin; using std::cos; using std::tan; 
using std::exp; using std::log; using std::sqrt;
using std::pow; using std::abs;

// {{PARAMETERS}}

const int DIM = {{STATE_DIM}};
std::vector<double> state(DIM);
double t = 0.0;

// Equations
void equations(const std::vector<double>& y, std::vector<double>& dydt, double t) {
{{STATE_UNPACK}}
{{EQUATIONS}}
}

void rk4_step(std::vector<double>& y, double t, double dt) {
    // ... (Compact RK4 implementation) ...
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

extern "C" {

EMSCRIPTEN_KEEPALIVE
void init() {
    std::vector<double> y0 = { {{INITIAL_CONDITIONS}} };
    state = y0;
    t = 0.0;
}

EMSCRIPTEN_KEEPALIVE
void step(double dt) {
    rk4_step(state, t, dt);
    t += dt;
}

EMSCRIPTEN_KEEPALIVE
double get_state(int index) {
    if(index >= 0 && index < DIM) return state[index];
    return 0.0;
}

EMSCRIPTEN_KEEPALIVE
double get_time() {
    return t;
}

}
