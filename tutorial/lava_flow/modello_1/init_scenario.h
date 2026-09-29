#pragma once
#include "particles.h"
#include <json.hpp>
#include <cmath>
#include <iostream>

inline particles_t initialize_crater(const quadgrid_t<vector_t<real_t>>& qg, const nlohmann::json& j) {
    using idx_t = particles_t::idx_t;

    // 1. Dynamic extraction from JSON
    real_t x_vent = j["eruption_parameters"]["x_vent"];
    real_t y_vent = j["eruption_parameters"]["y_vent"];
    real_t radius_vent = j["eruption_parameters"]["radius_vent"];
    real_t h_initial_lava = j["eruption_parameters"]["h_initial_lava"];
    int part_x = j["eruption_parameters"]["particles_per_cell_x"];
    int part_y = j["eruption_parameters"]["particles_per_cell_y"];

    vector_t<real_t> xv;
    vector_t<real_t> yv;

    // Estrazione diretta delle proprietà della griglia dal JSON per bypassare qg
    real_t hx = j["grid_properties"]["hx"];
    real_t hy = j["grid_properties"]["hy"];
    real_t x_min = j["grid_properties"]["x_min"];
    real_t y_min = j["grid_properties"]["y_min"];
    idx_t nrows = j["grid_properties"]["ny"];
    idx_t ncols = j["grid_properties"]["nx"];
    
    real_t step_x = hx / part_x;
    real_t step_y = hy / part_y;

    // 2. Positions generation
    for (idx_t i = 0; i < nrows; ++i) {
        for (idx_t j_col = 0; j_col < ncols; ++j_col) {
            real_t cell_x = x_min + j_col * hx + hx / 2.0;
            real_t cell_y = y_min + i * hy + hy / 2.0;

            if (std::hypot(cell_x - x_vent, cell_y - y_vent) <= radius_vent) {
                for (int py = 0; py < part_y; ++py) {
                    for (int px = 0; px < part_x; ++px) {
                        xv.push_back(x_min + j_col * hx + step_x * (px + 0.5));
                        yv.push_back(y_min + i * hy + step_y * (py + 0.5));
                    }
                }
            }
        }
    }

    idx_t num_particles = xv.size();
    std::cerr << "Initialized " << num_particles << " particles in the vent.\n";

    // 3. Allocation using custom vectors
    vector_t<std::string> ipropnames = {};
    vector_t<std::string> dpropnames = {"Vp", "ux", "uy"}; 

    particles_t p(num_particles, ipropnames, dpropnames, qg, xv, yv);

    p.build_mass();
    p.init_particle_mesh();

    // 4. Lagrangian volume assignment
    real_t area_per_particle = (hx * hy) / (part_x * part_y);
    real_t vp_initial = h_initial_lava * area_per_particle;

    for (idx_t ip = 0; ip < num_particles; ++ip) {
        p.dp("Vp", ip) = vp_initial;
        p.dp("ux", ip) = 0.0;
        p.dp("uy", ip) = 0.0;
    }

    p.memcpy_host_to_device();
    
    return p;
}