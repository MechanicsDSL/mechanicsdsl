/*******************************************************************************************
*
* MechanicsDSL - Raylib Visualization
* System: {{SYSTEM_NAME}}
*
********************************************************************************************/
#include "raylib.h"
#include <vector>
#include <cmath>
#include <string>
#include <cstdio>

using std::sin; using std::cos; using std::tan; 
using std::exp; using std::log; using std::sqrt;
using std::pow; using std::abs;

// --- Physics Parameters ---
// {{PARAMETERS}}

const int DIM = {{STATE_DIM}};
const int SCALE = 100; // Pixels per meter

// --- Equations of Motion ---
void equations(const std::vector<double>& y, std::vector<double>& dydt, double t) {
{{STATE_UNPACK}}
    
{{EQUATIONS}}
}

// --- Solver ---
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

    for(int i=0; i<DIM; i++) y[i] += (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]) / 6.0;
}

int main(void)
{
    const int screenWidth = 800;
    const int screenHeight = 600;

    InitWindow(screenWidth, screenHeight, "MechanicsDSL: {{SYSTEM_NAME}}");

    // Initial Conditions
    std::vector<double> y = { {{INITIAL_CONDITIONS}} };
    double t = 0.0;
    
    // Simulation speed
    int sub_steps = 10; 
    double dt = 0.01 / sub_steps;

    SetTargetFPS(60);

    // Trail buffer
    const int trail_len = 200;
    std::vector<Vector2> trail;

    while (!WindowShouldClose())
    {
        // 1. Update Physics
        for(int i=0; i<sub_steps; i++) {
            rk4_step(y, t, dt);
            t += dt;
        }

        // 2. Map coordinates to screen (Generic mapping: x=y[0], y=y[2] usually)
        // Defaulting to first coordinate pair for visualization
        double sim_x = y[0]; 
        double sim_y = (DIM > 2) ? y[2] : 0.0; 

        // Center origin on screen
        Vector2 screen_pos = { 
            (float)(screenWidth/2 + sim_x * SCALE), 
            (float)(screenHeight/2 - sim_y * SCALE) // Flip Y for screen coords
        };

        // Update Trail
        trail.push_back(screen_pos);
        if(trail.size() > trail_len) trail.erase(trail.begin());

        // 3. Draw
        BeginDrawing();
            ClearBackground(RAYWHITE);
            
            DrawText("MechanicsDSL Real-Time Sim", 10, 10, 20, DARKGRAY);
            char time_str[32];
            sprintf(time_str, "Time: %.2f s", t);
            DrawText(time_str, 10, 30, 20, GRAY);

            // Draw Axis
            DrawLine(screenWidth/2, 0, screenWidth/2, screenHeight, LIGHTGRAY);
            DrawLine(0, screenHeight/2, screenWidth, screenHeight/2, LIGHTGRAY);

            // Draw Trail
            for(size_t i = 1; i < trail.size(); i++) {
                DrawLineV(trail[i-1], trail[i], RED);
            }

            // Draw Object
            DrawCircleV(screen_pos, 10, MAROON);
            
            // Draw Origin connection (like a pendulum arm)
            DrawLine(screenWidth/2, screenHeight/2, (int)screen_pos.x, (int)screen_pos.y, BLACK);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
