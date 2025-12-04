
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
// // Physical Parameters
const double h = 0.04;
const double g = 9.81;

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
    //     particles.push_back({ 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.04, 0.0, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.08, 0.0, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.12, 0.0, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.16, 0.0, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.2, 0.0, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.24, 0.0, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.28, 0.0, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.32, 0.0, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.36, 0.0, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.0, 0.04, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.04, 0.04, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.08, 0.04, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.12, 0.04, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.16, 0.04, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.2, 0.04, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.24, 0.04, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.28, 0.04, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.32, 0.04, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.36, 0.04, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.0, 0.08, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.04, 0.08, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.08, 0.08, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.12, 0.08, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.16, 0.08, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.2, 0.08, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.24, 0.08, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.28, 0.08, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.32, 0.08, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.36, 0.08, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.0, 0.12, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.04, 0.12, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.08, 0.12, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.12, 0.12, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.16, 0.12, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.2, 0.12, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.24, 0.12, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.28, 0.12, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.32, 0.12, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.36, 0.12, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.0, 0.16, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.04, 0.16, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.08, 0.16, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.12, 0.16, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.16, 0.16, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.2, 0.16, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.24, 0.16, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.28, 0.16, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.32, 0.16, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.36, 0.16, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.0, 0.2, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.04, 0.2, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.08, 0.2, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.12, 0.2, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.16, 0.2, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.2, 0.2, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.24, 0.2, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.28, 0.2, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.32, 0.2, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.36, 0.2, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.0, 0.24, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.04, 0.24, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.08, 0.24, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.12, 0.24, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.16, 0.24, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.2, 0.24, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.24, 0.24, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.28, 0.24, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.32, 0.24, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.36, 0.24, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.0, 0.28, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.04, 0.28, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.08, 0.28, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.12, 0.28, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.16, 0.28, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.2, 0.28, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.24, 0.28, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.28, 0.28, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.32, 0.28, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.36, 0.28, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.0, 0.32, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.04, 0.32, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.08, 0.32, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.12, 0.32, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.16, 0.32, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.2, 0.32, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.24, 0.32, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.28, 0.32, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.32, 0.32, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.36, 0.32, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.0, 0.36, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.04, 0.36, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.08, 0.36, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.12, 0.36, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.16, 0.36, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.2, 0.36, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.24, 0.36, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.28, 0.36, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.32, 0.36, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.36, 0.36, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.0, 0.4, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.04, 0.4, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.08, 0.4, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.12, 0.4, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.16, 0.4, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.2, 0.4, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.24, 0.4, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.28, 0.4, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.32, 0.4, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.36, 0.4, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.0, 0.44, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.04, 0.44, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.08, 0.44, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.12, 0.44, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.16, 0.44, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.2, 0.44, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.24, 0.44, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.28, 0.44, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.32, 0.44, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.36, 0.44, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.0, 0.48, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.04, 0.48, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.08, 0.48, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.12, 0.48, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.16, 0.48, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.2, 0.48, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.24, 0.48, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.28, 0.48, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.32, 0.48, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.36, 0.48, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.0, 0.52, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.04, 0.52, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.08, 0.52, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.12, 0.52, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.16, 0.52, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.2, 0.52, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.24, 0.52, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.28, 0.52, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.32, 0.52, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.36, 0.52, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.0, 0.56, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.04, 0.56, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.08, 0.56, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.12, 0.56, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.16, 0.56, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.2, 0.56, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.24, 0.56, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.28, 0.56, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.32, 0.56, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.36, 0.56, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.0, 0.6, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.04, 0.6, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.08, 0.6, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.12, 0.6, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.16, 0.6, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.2, 0.6, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.24, 0.6, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.28, 0.6, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.32, 0.6, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.36, 0.6, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.0, 0.64, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.04, 0.64, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.08, 0.64, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.12, 0.64, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.16, 0.64, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.2, 0.64, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.24, 0.64, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.28, 0.64, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.32, 0.64, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.36, 0.64, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.0, 0.68, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.04, 0.68, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.08, 0.68, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.12, 0.68, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.16, 0.68, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.2, 0.68, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.24, 0.68, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.28, 0.68, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.32, 0.68, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.36, 0.68, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.0, 0.72, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.04, 0.72, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.08, 0.72, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.12, 0.72, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.16, 0.72, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.2, 0.72, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.24, 0.72, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.28, 0.72, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.32, 0.72, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.36, 0.72, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.0, 0.76, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.04, 0.76, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.08, 0.76, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.12, 0.76, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.16, 0.76, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.2, 0.76, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.24, 0.76, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.28, 0.76, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.32, 0.76, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ 0.36, 0.76, 0, 0, 0, 0, 0, 0, 0 });
    particles.push_back({ -0.05, 0.0, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.02, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.04, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.06, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.08, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.1, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.12, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.14, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.16, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.18, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.2, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.22, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.24, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.26, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.28, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.3, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.32, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.34, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.36, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.38, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.4, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.42, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.44, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.46, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.48, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.5, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.52, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.54, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.56, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.58, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.6, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.62, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.64, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.66, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.68, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.7000000000000001, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.72, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.74, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.76, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.78, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.8, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.8200000000000001, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.84, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.86, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.88, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.9, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.92, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.9400000000000001, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.96, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 0.98, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 1.0, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 1.02, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 1.04, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 1.06, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 1.08, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 1.1, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 1.12, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 1.1400000000000001, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 1.16, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 1.18, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 1.2, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 1.22, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 1.24, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 1.26, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 1.28, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 1.3, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 1.32, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 1.34, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 1.36, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 1.3800000000000001, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 1.4000000000000001, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 1.42, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 1.44, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 1.46, 0, 0, 0, 0, 0, 0, 1 });
    particles.push_back({ -0.05, 1.48, 0, 0, 0, 0, 0, 0, 1 });

    
    SpatialHash grid(particles.size(), H);
    
    std::ofstream file("dam_break_sph.csv");
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
    std::cout << "Done. Output written to dam_break_sph.csv" << std::endl;
    return 0;
}
