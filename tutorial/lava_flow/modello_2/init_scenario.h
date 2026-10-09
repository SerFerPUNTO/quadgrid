#pragma once
#include "particles.h"
#include <json.hpp>
#include <cmath>
#include <iostream>
#include <algorithm> // Necessario per std::shuffle
#include <random>    // Necessario per std::mt19937
#include <numeric>   // Necessario per std::iota

inline particles_t initialize_crater(const quadgrid_t<vector_t<real_t>>& qg, const nlohmann::json& j) {
    using idx_t = particles_t::idx_t;

    real_t x_vent = j["eruption_parameters"]["x_vent"];
    real_t y_vent = j["eruption_parameters"]["y_vent"];
    real_t radius_vent = j["eruption_parameters"]["radius_vent"];
    int total_desired_particles = j["eruption_parameters"]["total_particles"];

    // Vettori temporanei per la generazione
    vector_t<real_t> xv_temp;
    vector_t<real_t> yv_temp;

    real_t spacing = radius_vent * std::sqrt(M_PI / total_desired_particles);

    for (real_t x = x_vent - radius_vent + spacing / 2.0; x <= x_vent + radius_vent; x += spacing) {
        for (real_t y = y_vent - radius_vent + spacing / 2.0; y <= y_vent + radius_vent; y += spacing) {
            if (std::hypot(x - x_vent, y - y_vent) <= radius_vent) {
                xv_temp.push_back(x);
                yv_temp.push_back(y);
            }
        }
    }

    idx_t num_particles = xv_temp.size();

    // ---------------------------------------------------------
    // FIX SPATIAL BIAS: Mescoliamo le particelle casualmente
    // ---------------------------------------------------------
    std::vector<idx_t> indices(num_particles);
    std::iota(indices.begin(), indices.end(), 0); // Riempe con 0, 1, 2, ..., N-1
    std::mt19937 gen(42); // Usiamo un seed fisso (42) per riproducibilità scientifica
    std::shuffle(indices.begin(), indices.end(), gen);

    vector_t<real_t> xv(num_particles);
    vector_t<real_t> yv(num_particles);
    
    // Riapplichiamo il mescolamento a coordinate X e Y mantenendole appaiate
    for(idx_t i = 0; i < num_particles; ++i) {
        xv[i] = xv_temp[indices[i]];
        yv[i] = yv_temp[indices[i]];
    }
    // ---------------------------------------------------------

    std::cerr << "Initialized " << num_particles << " dormant particles in the vent.\n";

    vector_t<std::string> ipropnames = {};
    vector_t<std::string> dpropnames = {"Vp", "vx", "vy"}; 

    particles_t p(num_particles, ipropnames, dpropnames, qg, xv, yv);

    p.build_mass();
    p.init_particle_mesh();

    for (idx_t ip = 0; ip < num_particles; ++ip) {
        p.dp("Vp", ip) = 0.0;  
        p.dp("vx", ip) = 0.0;
        p.dp("vy", ip) = 0.0;
    }

    p.memcpy_host_to_device();
    
    return p;
}