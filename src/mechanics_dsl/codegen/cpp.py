"""
C++ Code Generator for MechanicsDSL

This module handles the translation of symbolic expressions and simulation parameters
into optimized C++ code using a template-based approach.
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
                 equations: Dict[str, sp.Expr]):
        self.system_name = system_name
        self.coordinates = coordinates
        self.parameters = parameters
        self.initial_conditions = initial_conditions
        self.equations = equations
        
        # Load template
        self.template_path = os.path.join(os.path.dirname(__file__), 'templates', 'solver_template.cpp')
        if not os.path.exists(self.template_path):
            # Fallback for when installed as package without templates
            self.template_content = self._get_default_template()
        else:
            with open(self.template_path, 'r') as f:
                self.template_content = f.read()

    def generate(self, output_file: str = "simulation.cpp"):
        """
        Generate C++ source file
        
        Args:
            output_file: Path to write the .cpp file
        """
        logger.info(f"Generating C++ code for {self.system_name}")
        
        # 1. Generate Parameters
        param_str = "// Physical Parameters\n"
        for name, val in self.parameters.items():
            param_str += f"const double {name} = {val};\n"
            
        # 2. Generate State Unpacking
        # y[0] -> q1, y[1] -> q1_dot, y[2] -> q2 ...
        unpack_str = "// Unpack state variables\n"
        idx = 0
        for coord in self.coordinates:
            unpack_str += f"    double {coord} = y[{idx}];\n"
            unpack_str += f"    double {coord}_dot = y[{idx+1}];\n"
            idx += 2
            
        # 3. Generate Equations
        eq_str = "// Computed Derivatives\n"
        idx = 0
        
        # Custom printer settings for C++
        settings = {'standard': 'c++17'}
        
        for coord in self.coordinates:
            accel_key = f"{coord}_ddot"
            
            # d(pos)/dt = vel
            eq_str += f"    dydt[{idx}] = {coord}_dot;\n"
            
            # d(vel)/dt = accel
            if accel_key in self.equations:
                expr = self.equations[accel_key]
                # Convert SymPy expression to C++ code
                cpp_expr = cxxcode(expr, standard='c++17')
                eq_str += f"    dydt[{idx+1}] = {cpp_expr};\n"
            else:
                eq_str += f"    dydt[{idx+1}] = 0.0; // No equation found for {accel_key}\n"
                
            idx += 2

        # 4. Initial Conditions Vector
        # We need to map the dictionary to the vector order [q1, v1, q2, v2...]
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

        # Fill Template
        code = self.template_content
        code = code.replace("{{SYSTEM_NAME}}", self.system_name)
        code = code.replace("{{PARAMETERS}}", param_str)
        code = code.replace("{{STATE_DIM}}", str(len(self.coordinates) * 2))
        code = code.replace("{{STATE_UNPACK}}", unpack_str)
        code = code.replace("{{EQUATIONS}}", eq_str)
        code = code.replace("{{INITIAL_CONDITIONS}}", init_str)
        code = code.replace("{{CSV_HEADER}}", header_str)
        
        with open(output_file, 'w') as f:
            f.write(code)
            
        logger.info(f"Successfully wrote {output_file}")
        return output_file

    def _get_default_template(self):
        """Fallback template if file is missing"""
        return r"""
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
    std::vector<double> y = { {{INITIAL_CONDITIONS}} };
    double t = 0.0;
    double dt = 0.001; // Fixed step size
    double t_end = 10.0;
    int steps = static_cast<int>(t_end / dt);
    int log_interval = 10; // Log every 10 steps

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
