const EXAMPLES = {
    'pendulum': { name: 'Simple Pendulum', code: `\\system{simple_pendulum}\n\\parameter{m}{1.0}{kg}\n\\parameter{l}{1.0}{m}\n\\parameter{g}{9.81}{m/s^2}\n\\defvar{theta}{angle}{rad}\n\\lagrangian{\\frac{1}{2} m l^2 \\dot{\\theta}^2 - m g l (1 - \\cos(\\theta))}\n\\initial{theta=0.8, theta_dot=0}`, params: { type: 'pendulum', theta: 0.8 } },
    'double-pendulum': { name: 'Double Pendulum', code: `\\system{double_pendulum}\n\\parameter{m1}{1.0}{kg}\n\\parameter{m2}{1.0}{kg}\n\\parameter{l1}{1.0}{m}\n\\parameter{l2}{1.0}{m}\n\\parameter{g}{9.81}{m/s^2}\n\\defvar{theta1}{angle1}{rad}\n\\defvar{theta2}{angle2}{rad}\n\\lagrangian{...}\n\\initial{theta1=2.5, theta2=2.0}`, params: { type: 'double-pendulum', theta1: 2.5, theta2: 2.0 } },
    'spring': { name: 'Spring Oscillator', code: `\\system{spring}\n\\parameter{k}{10}{N/m}\n\\parameter{m}{1}{kg}\n\\parameter{b}{0.1}{Ns/m}\n\\defvar{x}{displacement}{m}\n\\lagrangian{\\frac{1}{2}m\\dot{x}^2 - \\frac{1}{2}kx^2}\n\\initial{x=1.5}`, params: { type: 'spring', x: 1.5 } },
    'orbital': { name: 'Orbital Mechanics', code: `\\system{orbit}\n\\parameter{GM}{1000}{m^3/s^2}\n\\defvar{x}{position_x}{m}\n\\defvar{y}{position_y}{m}\n\\lagrangian{\\frac{1}{2}(\\dot{x}^2+\\dot{y}^2)+\\frac{GM}{r}}\n\\initial{x=100, y=0, x_dot=0, y_dot=8}`, params: { type: 'orbital', r: 100 } },
    'rigid-body': { name: 'Rigid Body', code: `\\system{rigid_body}\n% Quaternion dynamics\n\\initial{q0=0.99, q1=0.1}`, params: { type: 'double-pendulum', theta1: 0.2, omega2: 10 } },
    'sph': { name: 'Fluid SPH', code: `\\system{dam_break}\n\\parameter{h}{0.04}{m}\n\\parameter{rho0}{1000}{kg/m^3}\n\\initial{particles=dam_break}`, params: { type: 'sph', n: 150 } }
};

const CODE_TEMPLATES = {
    cpp: (n) => `// C++ code for ${n}\n#include <cmath>\nconst double g = 9.81, l = 1.0;\nvoid derivatives(double* y, double* dy) {\n    dy[0] = y[1];\n    dy[1] = -g/l * sin(y[0]);\n}`,
    python: (n) => `# Python code for ${n}\nimport numpy as np\nfrom scipy.integrate import solve_ivp\ng, l = 9.81, 1.0\ndef f(t, y): return [y[1], -g/l*np.sin(y[0])]\nsol = solve_ivp(f, [0,10], [0.5,0])`,
    cuda: (n) => `// CUDA code for ${n}\n__device__ void derivatives(float* y, float* dy) {\n    dy[0] = y[1];\n    dy[1] = -9.81f/1.0f * sinf(y[0]);\n}`,
    rust: (n) => `// Rust code for ${n}\nconst G: f64 = 9.81;\nconst L: f64 = 1.0;\nfn derivatives(y: &[f64;2]) -> [f64;2] {\n    [y[1], -G/L * y[0].sin()]\n}`,
    julia: (n) => `# Julia code for ${n}\nusing DifferentialEquations\nf(u,p,t) = [u[2], -9.81/1.0*sin(u[1])]\nprob = ODEProblem(f, [0.5, 0.0], (0.0, 10.0))`
};

window.EXAMPLES = EXAMPLES;
window.CODE_TEMPLATES = CODE_TEMPLATES;
