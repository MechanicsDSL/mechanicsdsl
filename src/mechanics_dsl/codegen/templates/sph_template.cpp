/*
 * MechanicsDSL - SPH Fluid Engine
 * System: {{SYSTEM_NAME}}
 * Features: Spatial Hashing, Velocity Verlet, Tait EOS
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>

// --- Parameters ---
// {{PARAMETERS}}
const double H = h; // Smoothing length from parameters
const double MASS = 0.02; // Default mass if not specified
const double DT = 0.002;  // Fixed time step for stability

// SPH Constants
const double POLY6 = 315.0 / (64.0 * M_PI * pow(H, 9));
const double SPIKY_GRAD = -45.0 / (M_PI * pow(H, 6));
const double VISC_LAP = 45.0 / (M_PI * pow(H, 6));
const double GAS_CONST = 2000.0; // Stiffness
const double REST_DENS = 1000.0;
const double VISCOSITY = 2.5;

struct Particle {
    double x, y;
    double vx, vy;
    double fx, fy;
    double rho, p;
    int type; // 0 = Fluid, 1 = Boundary
};

// Spatial Hash for O(N) neighbor search
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
    
    // Iterate over 3x3 neighborhood
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

// --- SPH Kernels ---

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
        
        // Tait EOS
        particles[i].rho = std::max(REST_DENS, particles[i].rho);
        particles[i].p = GAS_CONST * (pow(particles[i].rho / REST_DENS, 7) - 1);
    }
}

void compute_forces(SpatialHash& grid) {
    for(int i=0; i<particles.size(); i++) {
        particles[i].fx = 0;
        particles[i].fy = -9.81 * MASS; // Gravity
        
        if(particles[i].type == 1) continue; // Boundaries don't move
        
        grid.query(particles, i, [&](int j) {
            double dx = particles[i].x - particles[j].x;
            double dy = particles[i].y - particles[j].y;
            double r = sqrt(dx*dx + dy*dy);
            
            if(r > 0 && r < H) {
                // Pressure Force
                double f_press = -MASS * (particles[i].p + particles[j].p) / (2 * particles[j].rho) * SPIKY_GRAD * pow(H-r, 2);
                
                // Viscosity Force
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
            // Velocity Verlet (Semi-implicit Euler for performance)
            p.vx += (p.fx / p.rho) * DT;
            p.vy += (p.fy / p.rho) * DT;
            p.x += p.vx * DT;
            p.y += p.vy * DT;
            
            // Crude Boundary Enforcement (failsafe)
            if(p.y < -0.1) { p.y = -0.1; p.vy *= -0.5; }
        }
    }
}

int main() {
    // 1. Initialize Particles
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
