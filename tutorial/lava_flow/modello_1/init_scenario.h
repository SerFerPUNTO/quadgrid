#pragma once
#include "particles.h"
#include <json.hpp>
#include <cmath>
#include <iostream>

inline particles_t initialize_crater(const quadgrid_t<vector_t<real_t>>& qg, const nlohmann::json& j) {
    using idx_t = particles_t::idx_t;

    // 1. Estrazione dinamica dei parametri dal JSON
    real_t x_vent = j["eruption_parameters"]["x_vent"];
    real_t y_vent = j["eruption_parameters"]["y_vent"];
    real_t radius_vent = j["eruption_parameters"]["radius_vent"];
    
    // Numero TOTALE di particelle desiderate nel bacino
    int total_desired_particles = j["eruption_parameters"]["total_particles"];

    vector_t<real_t> xv;
    vector_t<real_t> yv;

    // 2. Calcolo della spaziatura uniforme (reticolo quadrato inscritto nel cerchio)
    real_t spacing = radius_vent * std::sqrt(M_PI / total_desired_particles);

    // 3. Generazione posizioni uniformi (bacino inesauribile nel cratere)
    // Sfasiamo di spacing/2 per centrare il reticolo
    for (real_t x = x_vent - radius_vent + spacing / 2.0; x <= x_vent + radius_vent; x += spacing) {
        for (real_t y = y_vent - radius_vent + spacing / 2.0; y <= y_vent + radius_vent; y += spacing) {
            
            // Se la particella cade dentro il cerchio del cratere, la salviamo
            if (std::hypot(x - x_vent, y - y_vent) <= radius_vent) {
                xv.push_back(x);
                yv.push_back(y);
            }
        }
    }

    idx_t num_particles = xv.size();
    std::cerr << "Initialized " << num_particles << " dormant particles in the vent.\n";

    // 4. Allocazione vettori lagrangiani
    vector_t<std::string> ipropnames = {};
    vector_t<std::string> dpropnames = {"Vp", "vx", "vy"}; 

    particles_t p(num_particles, ipropnames, dpropnames, qg, xv, yv);

    p.build_mass();
    p.init_particle_mesh();

    // 5. Inizializzazione Lagrangiana a volume zero (Particle Activation)
    for (idx_t ip = 0; ip < num_particles; ++ip) {
        p.dp("Vp", ip) = 0.0;  // Nascono spente
        p.dp("vx", ip) = 0.0;
        p.dp("vy", ip) = 0.0;
    }

    p.memcpy_host_to_device();
    
    return p;
}