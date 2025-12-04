"""
C++ Code Generator for MechanicsDSL
"""
import os
import sympy as sp
from sympy.printing.cxx import cxxcode
from typing import Dict, List
from ..utils import logger

class CppGenerator:
    """Generates C++ simulation code from symbolic equations"""
    
    def __init__(self, system_name: str, coordinates: List[str], 
                 parameters: Dict[str, float], initial_conditions: Dict[str, float],
                 equations: Dict[str, sp.Expr],
                 fluid_particles: List[dict] = None,
                 boundary_particles: List[dict] = None):
        
        self.system_name = system_name
        self.coordinates = coordinates
        self.parameters = parameters
        self.initial_conditions = initial_conditions
        self.equations = equations or {}
        self.fluid_particles = fluid_particles or []
        self.boundary_particles = boundary_particles or []
        
        # Load template
        self.template_path = os.path.join(os.path.dirname(__file__), 'templates', 'solver_template.cpp')
        
        # LOGIC: Choose the template content
        if self.fluid_particles:
            # Case A: Fluids -> Use embedded SPH template
            logger.info("CppGenerator: Using SPH Fluid Template")
            self.template_content = self._get_sph_template()
        else:
            # Case B: Rigid Body -> Use standard file template
            logger.info("CppGenerator: Using Standard Solver Template")
            if not os.path.exists(self.template_path):
                self.template_content = self._get_default_template()
            else:
                with open(self.template_path, 'r') as f:
                    self.template_content = f.read()

    def generate(self, output_file: str = "simulation.cpp"):
        logger.info(f"Generating C++ code for {self.system_name}")
        
        # 1. Generate Parameters
        param_str = "// Physical Parameters\n"
        for name, val in self.parameters.items():
            param_str += f"const double {name} = {val};\n"
            
        # 2. Generate State Unpacking (For Rigid Bodies)
        unpack_str = "// Unpack state variables\n"
        idx = 0
        for coord in self.coordinates:
            unpack_str += f"    double {coord} = y[{idx}];\n"
            unpack_str += f"    double {coord}_dot = y[{idx+1}];\n"
            idx += 2
            
        # 3. Generate Equations (For Rigid Bodies)
        eq_str = "// Computed Derivatives\n"
        idx = 0
        for coord in self.coordinates:
            accel_key = f"{coord}_ddot"
            eq_str += f"    dydt[{idx}] = {coord}_dot;\n"
            if accel_key in self.equations:
                expr = self.equations[accel_key]
                cpp_expr = cxxcode(expr, standard='c++17')
                eq_str += f"    dydt[{idx+1}] = {cpp_expr};\n"
            else:
                eq_str += f"    dydt[{idx+1}] = 0.0;\n"
            idx += 2

        # 4. Initial Conditions Vector
        init_vals = []
        for coord in self.coordinates:
            pos = self.initial_conditions.get(coord, 0.0)
            vel = self.initial_conditions.get(f"{coord}_dot", 0.0)
            init_vals.append(str(pos))
            init_vals.append(str(vel))
        init_str = ", ".join(init_vals)
        
        # 5. CSV Header
        header_parts = ["t"]
        for coord in self.coordinates:
            header_parts.append(coord)
            header_parts.append(f"{coord}_dot")
        header_str = ",".join(header_parts)

        # 6. Particle Initialization (For Fluids)
        particle_init_str = ""
        if self.fluid_particles:
            for p in self.fluid_particles:
                particle_init_str += f"    particles.push_back({{ {p['x']}, {p['y']}, 0, 0, 0, 0, 0, 0, 0 }});\n"
            for p in self.boundary_particles:
                particle_init_str += f"    particles.push_back({{ {p['x']}, {p['y']}, 0, 0, 0, 0, 0, 0, 1 }});\n"

        # Fill Template
        code = self.template_content
        code = code.replace("{{SYSTEM_NAME}}", self.system_name)
        code = code.replace("{{PARAMETERS}}", param_str)
        code = code.replace("{{STATE_DIM}}", str(len(self.coordinates) * 2))
        code = code.replace("{{STATE_UNPACK}}", unpack_str)
        code = code.replace("{{EQUATIONS}}", eq_str)
        code = code.replace("{{INITIAL_CONDITIONS}}", init_str)
        code = code.replace("{{CSV_HEADER}}", header_str)
        code = code.replace("{{PARTICLE_INIT}}", particle_init_str) # Inject particles
        
        with open(output_file, 'w') as f:
            f.write(code)
            
        logger.info(f"Successfully wrote {output_file}")
        return output_file

    def _get_default_template(self):
        # Fallback rigid body template
        return r"""
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>

using std::sin; using std::cos; using std::tan; 
using std::exp; using std::log; using std::sqrt;
using std::pow; using std::abs;

// {{PARAMETERS}}
const int DIM = {{STATE_DIM}};

void equations(const std::vector<double>& y, std::vector<double>& dydt, double t) {
{{STATE_UNPACK}}
{{EQUATIONS}}
}

void rk4_step(std::vector<double>& y, double t, double dt) {
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

int main() {
    std::vector<double> y = { {{INITIAL_CONDITIONS}} };
    double t = 0.0;
    double dt = 0.001;
    double t_end = 10.0;
    int steps = static_cast<int>(t_end / dt);
    int log_interval = 10;

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
"""

    def _get_sph_template(self):
        # SPH Fluid Template
        return r"""
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>

using std::sin; using std::cos; using std::tan; 
using std::exp; using std::log; using std::sqrt;
using std::pow; using std::abs;

// --- Parameters ---
// {{PARAMETERS}}
const double H = h; 
const double MASS = 0.02; 
const double DT = 0.002;

// SPH Constants
const double PI = 3.14159265358979323846;
const double POLY6 = 315.0 / (64.0 * PI * pow(H, 9));
const double SPIKY_GRAD = -45.0 / (PI * pow(H, 6));
const double VISC_LAP = 45.0 / (PI * pow(H, 6));
const double GAS_CONST = 2000.0; 
const double REST_DENS = 1000.0;
const double VISCOSITY = 2.5;

struct Particle {
    double x, y;
    double vx, vy;
    double fx, fy;
    double rho, p;
    int type; // 0 = Fluid, 1 = Boundary
};

class SpatialHash {
public:
    double cell_size;
    int table_size;
    std::vector<int> head;
    std::vector<int> next;

    SpatialHash(int n, double h) : cell_size(h), table_size(2*n) {
        head.resize(table_size, -1);
        next.resize(n, -1);
    }

    int hash(double x, double y) {
        int i = static_cast<int>(x / cell_size);
        int j = static_cast<int>(y / cell_size);
        return (abs(i * 92837111) ^ abs(j * 689287499)) % table_size;
    }

    void build(const std::vector<Particle>& p) {
        std::fill(head.begin(), head.end(), -1);
        for(int i=0; i<p.size(); i++) {
            int h = hash(p[i].x, p[i].y);
            next[i] = head[h];
            head[h] = i;
        }
    }
    
    template<typename Func>
    void query(const std::vector<Particle>& p, int i, Func f) {
        int cx = static_cast<int>(p[i].x / cell_size);
        int cy = static_cast<int>(p[i].y / cell_size);
        
        for(int dx=-1; dx<=1; dx++) {
            for(int dy=-1; dy<=1; dy++) {
                int h = (abs((cx+dx) * 92837111) ^ abs((cy+dy) * 689287499)) % table_size;
                int j = head[h];
                while(j != -1) {
                    if(i != j) f(j);
                    j = next[j];
                }
            }
        }
    }
};

std::vector<Particle> particles;

void compute_density_pressure(SpatialHash& grid) {
    for(int i=0; i<particles.size(); i++) {
        particles[i].rho = 0;
        grid.query(particles, i, [&](int j) {
            double dx = particles[i].x - particles[j].x;
            double dy = particles[i].y - particles[j].y;
            double r2 = dx*dx + dy*dy;
            if(r2 < H*H) {
                particles[i].rho += MASS * POLY6 * pow(H*H - r2, 3);
            }
        });
        particles[i].rho = std::max(REST_DENS, particles[i].rho);
        particles[i].p = GAS_CONST * (pow(particles[i].rho / REST_DENS, 7) - 1);
    }
}

void compute_forces(SpatialHash& grid) {
    for(int i=0; i<particles.size(); i++) {
        particles[i].fx = 0;
        particles[i].fy = -9.81 * MASS; 
        
        if(particles[i].type == 1) continue; 
        
        grid.query(particles, i, [&](int j) {
            double dx = particles[i].x - particles[j].x;
            double dy = particles[i].y - particles[j].y;
            double r = sqrt(dx*dx + dy*dy);
            
            if(r > 0 && r < H) {
                double f_press = -MASS * (particles[i].p + particles[j].p) / (2 * particles[j].rho) * SPIKY_GRAD * pow(H-r, 2);
                double f_visc = VISCOSITY * MASS * VISC_LAP * (H-r) / particles[j].rho;
                
                double dir_x = dx/r;
                double dir_y = dy/r;
                
                particles[i].fx += f_press * dir_x + f_visc * (particles[j].vx - particles[i].vx);
                particles[i].fy += f_press * dir_y + f_visc * (particles[j].vy - particles[i].vy);
            }
        });
    }
}

void integrate() {
    for(auto& p : particles) {
        if(p.type == 0) {
            p.vx += (p.fx / p.rho) * DT;
            p.vy += (p.fy / p.rho) * DT;
            p.x += p.vx * DT;
            p.y += p.vy * DT;
            
            if(p.y < -0.2) { p.y = -0.2; p.vy *= -0.5; }
            if(p.x < -0.2) { p.x = -0.2; p.vx *= -0.5; }
            if(p.x > 2.0)  { p.x = 2.0;  p.vx *= -0.5; }
        }
    }
}

int main() {
    // {{PARTICLE_INIT}}
    
    SpatialHash grid(particles.size(), H);
    
    std::ofstream file("{{SYSTEM_NAME}}_sph.csv");
    file << "t,id,x,y,rho\n";
    
    std::cout << "Simulating " << particles.size() << " particles..." << std::endl;
    
    double t = 0;
    for(int step=0; step<2000; step++) {
        grid.build(particles);
        compute_density_pressure(grid);
        compute_forces(grid);
        integrate();
        
        if(step % 10 == 0) {
            for(int i=0; i<particles.size(); i++) {
                if(particles[i].type == 0)
                    file << t << "," << i << "," << particles[i].x << "," << particles[i].y << "," << particles[i].rho << "\n";
            }
        }
        t += DT;
    }
    std::cout << "Done. Output written to {{SYSTEM_NAME}}_sph.csv" << std::endl;
    return 0;
}
"""
