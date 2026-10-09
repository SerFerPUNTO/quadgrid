#ifndef CHANNEL_H
#define CHANNEL_H

#include <json.hpp>
#include <particles.h>
 
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <vector>
 
#include "counter.h"
#include <timer.h>
#include <quadgrid_config.h>

struct safe_divide {
  HOST DEVICE
  real_t operator()(real_t num, real_t den) const {
    return den > 0.0 ? num / den : 0.0;
  }
};

struct abs_no_nan {
  HOST DEVICE
  real_t operator()(real_t value) const {
    return value == value ? (value < 0.0 ? -value : value) : 0.0;
  }
};


template<typename PVAR_t>
class stepper {
private :
  PVAR_t x;
  PVAR_t y;
  const PVAR_t ux;
  const PVAR_t uy;
  PVAR_t Vp;
  PVAR_t Tp; 
  PVAR_t Ep; 
  real_t dt; 
  real_t x_min;
  real_t x_max;
  real_t y_min;
  real_t y_max;

public :
  stepper (PVAR_t x_, PVAR_t y_,
	   const PVAR_t ux_, const PVAR_t uy_, PVAR_t Vp_, PVAR_t Tp_, PVAR_t Ep_,
	   real_t dt_, real_t x_min_, real_t x_max_, real_t y_min_, real_t y_max_)
    : x(x_), y(y_), ux(ux_), uy(uy_), Vp(Vp_), Tp(Tp_), Ep(Ep_), dt(dt_), 
      x_min(x_min_), x_max(x_max_), y_min(y_min_), y_max(y_max_) { }

  DEVICE
  void operator() (int ind) {
    // Muove solo se la particella ha massa (è attiva)
    if (Vp[ind] > 0.0) {
        x[ind] += ux[ind] * dt; 
        y[ind] += uy[ind] * dt; 
    }
      
    // Rete di sicurezza RIGOROSA (nota i >= e <=)
    if (x[ind] != x[ind] || y[ind] != y[ind] || 
        x[ind] <= x_min || x[ind] >= x_max || 
        y[ind] <= y_min || y[ind] >= y_max) {
        
        x[ind] = x_min + 0.001; // La parcheggia in modo sicuro all'interno della cella 0
        y[ind] = y_min + 0.001; 
        Vp[ind] = 0.0; // Massa annullata
        Tp[ind] = 0.0; // Temperatura annullata
        Ep[ind] = 0.0; // Energia annullata
    }
  }
  real_t get_dt() const { return dt; }
  void set_dt(real_t new_dt) { dt = new_dt; }
};

//Injection term
template<typename PVAR_t>
class inject_source {
  // Variabili membro private di default
  PVAR_t Vp;
  PVAR_t Tp;
  PVAR_t Ep;
  const real_t Vp_active;
  const real_t Te;

public:
  inject_source(PVAR_t Vp_, PVAR_t Tp_, PVAR_t Ep_, real_t Vp_active_, real_t Te_)
    : Vp(Vp_), Tp(Tp_), Ep(Ep_), Vp_active(Vp_active_), Te(Te_) {}

  DEVICE
  void
  operator()(int ind) {
      Vp[ind] = Vp_active;
      Tp[ind] = Te;
      Ep[ind] = Vp_active * Te;
  }
};

// Gradient of the eulerian field
template<typename GVAR_t>
class evaluate_grad {
  using idx_t = particles_t::idx_t;
  
  GVAR_t grad_Zx;
  GVAR_t grad_Zy;
  const GVAR_t Z;
  const idx_t nrows;
  const idx_t ncols;
  const real_t hx;
  const real_t hy;
  
public :
  evaluate_grad (GVAR_t grad_Zx_, GVAR_t grad_Zy_, const GVAR_t Z_, 
                 const idx_t nrows_, const idx_t ncols_, 
                 const real_t hx_, const real_t hy_)
    : grad_Zx(grad_Zx_), grad_Zy(grad_Zy_), Z(Z_), 
      nrows(nrows_), ncols(ncols_), hx(hx_), hy(hy_) {};
  
  DEVICE
  void
  operator()(idx_t ind) {
    using qgt = quadgrid_t<real_t>;
    idx_t nodes_y = nrows + 1;

    idx_t r = qgt::gind2row(ind, nodes_y);
    idx_t c = qgt::gind2col(ind, nodes_y);

    if (c == 0) {
        grad_Zx[ind] = (Z[ind + nodes_y] - Z[ind]) / hx;
    } 
    else if (c == ncols) {
        grad_Zx[ind] = (Z[ind] - Z[ind - nodes_y]) / hx;
    } 
    else {
        grad_Zx[ind] = (Z[ind + nodes_y] - Z[ind - nodes_y]) / (2.0 * hx);
    }

    if (r == 0) {
        grad_Zy[ind] = (Z[ind + 1] - Z[ind]) / hy;
    } 
    else if (r == nrows) {
        grad_Zy[ind] = (Z[ind] - Z[ind - 1]) / hy;
    } 
    else {
        grad_Zy[ind] = (Z[ind + 1] - Z[ind - 1]) / (2.0 * hy);
    }
  }  
};

// Divergence of the Eulerian field (Upwind Scheme)
template<typename GVAR_t>
class evaluate_div {
  using idx_t = particles_t::idx_t;
  
  GVAR_t div_x;
  GVAR_t div_y;
  const GVAR_t Fxx;
  const GVAR_t Fxy;
  const GVAR_t Fyy;
  const GVAR_t Ux; // Necessario per leggere la direzione del vento X
  const GVAR_t Uy; // Necessario per leggere la direzione del vento Y
  const idx_t nrows;
  const idx_t ncols;
  const real_t hx;
  const real_t hy;
  
public :
  evaluate_div (GVAR_t div_x_, GVAR_t div_y_, 
                const GVAR_t Fxx_, const GVAR_t Fxy_, const GVAR_t Fyy_, 
                const GVAR_t Ux_, const GVAR_t Uy_,
                const idx_t nrows_, const idx_t ncols_, 
                const real_t hx_, const real_t hy_)
    : div_x(div_x_), div_y(div_y_), 
      Fxx(Fxx_), Fxy(Fxy_), Fyy(Fyy_), 
      Ux(Ux_), Uy(Uy_),
      nrows(nrows_), ncols(ncols_), hx(hx_), hy(hy_) {};
  
  DEVICE
  void
  operator()(idx_t ind) {
    using qgt = quadgrid_t<real_t>;
    idx_t nodes_y = nrows + 1;
    idx_t r = qgt::gind2row(ind, nodes_y);
    idx_t c = qgt::gind2col(ind, nodes_y);

    real_t ux = Ux[ind];
    real_t uy = Uy[ind];

    real_t dFxx_dx = 0.0, dFxy_dx = 0.0;
    real_t dFxy_dy = 0.0, dFyy_dy = 0.0;

    // Derivate lungo X (trasporto guidato da Ux)
    if (ux >= 0.0) { // Upwind all'indietro
        if (c > 0) {
            dFxx_dx = (Fxx[ind] - Fxx[ind - nodes_y]) / hx;
            dFxy_dx = (Fxy[ind] - Fxy[ind - nodes_y]) / hx;
        } else { // Fallback se siamo sul bordo sinistro
            dFxx_dx = (Fxx[ind + nodes_y] - Fxx[ind]) / hx; 
            dFxy_dx = (Fxy[ind + nodes_y] - Fxy[ind]) / hx;
        }
    } else { // Upwind in avanti
        if (c < ncols) { // Assicuriamoci di poter saltare in avanti
            dFxx_dx = (Fxx[ind + nodes_y] - Fxx[ind]) / hx;
            dFxy_dx = (Fxy[ind + nodes_y] - Fxy[ind]) / hx;
        } else { // Fallback se siamo sul bordo destro
            dFxx_dx = (Fxx[ind] - Fxx[ind - nodes_y]) / hx; 
            dFxy_dx = (Fxy[ind] - Fxy[ind - nodes_y]) / hx;
        }
    }

    // Derivate lungo Y (trasporto guidato da Uy)
    if (uy >= 0.0) { // Upwind all'indietro
        if (r > 0) {
            dFxy_dy = (Fxy[ind] - Fxy[ind - 1]) / hy;
            dFyy_dy = (Fyy[ind] - Fyy[ind - 1]) / hy;
        } else { // Fallback se siamo sul bordo superiore
            dFxy_dy = (Fxy[ind + 1] - Fxy[ind]) / hy;
            dFyy_dy = (Fyy[ind + 1] - Fyy[ind]) / hy;
        }
    } else { // Upwind in avanti
        if (r < nrows) { // Assicuriamoci di poter saltare in avanti
            dFxy_dy = (Fxy[ind + 1] - Fxy[ind]) / hy;
            dFyy_dy = (Fyy[ind + 1] - Fyy[ind]) / hy;
        } else { // Fallback se siamo sul bordo inferiore
            dFxy_dy = (Fxy[ind] - Fxy[ind - 1]) / hy;
            dFyy_dy = (Fyy[ind] - Fyy[ind - 1]) / hy;
        }
    }

    div_x[ind] = dFxx_dx + dFxy_dy;
    div_y[ind] = dFxy_dx + dFyy_dy;
    // if (div_x[ind] != 0) {
    //   printf("div_x[%d] = %e\n", (int)ind, div_x[ind]);
    // }
    // if (div_y[ind] != 0) {
    //   printf("div_y[%d] = %e\n", (int)ind, div_y[ind]);
    // }
  }
};

// Formula for U
template<typename GVAR_t>
class evaluate_U {
  using idx_t = particles_t::idx_t;
  
  GVAR_t Ux;
  GVAR_t Uy;
  const GVAR_t grad_Zx;
  const GVAR_t grad_Zy;
  const GVAR_t grad_hx;
  const GVAR_t grad_hy;
  const GVAR_t h;
  const GVAR_t div_x;
  const GVAR_t div_y;
  const GVAR_t T;
  real_t dt;
  real_t c_grad_Zh;
  const real_t g;
  
  // Nuovi parametri reologici per Modello 3
  const real_t nu_r;
  const real_t b;
  const real_t Tr;
  
public :
  evaluate_U (GVAR_t Ux_, GVAR_t Uy_, 
              const GVAR_t grad_Zx_, const GVAR_t grad_Zy_, 
              const GVAR_t grad_hx_, const GVAR_t grad_hy_, 
              const GVAR_t h_, const GVAR_t div_x_, const GVAR_t div_y_, const GVAR_t T_,
              real_t dt_, real_t g_, real_t nu_r_, real_t b_, real_t Tr_)
    : Ux(Ux_), Uy(Uy_), 
      grad_Zx(grad_Zx_), grad_Zy(grad_Zy_), 
      grad_hx(grad_hx_), grad_hy(grad_hy_), 
      h(h_), div_x(div_x_), div_y(div_y_), T(T_),
      g(g_), nu_r(nu_r_), b(b_), Tr(Tr_) 
  {
      set_dt(dt_);
  };
  
  void set_dt(real_t new_dt) {
      dt = new_dt;
      c_grad_Zh = -g * dt;
  }

  DEVICE
  void
  operator()(idx_t ind) {
    if (h[ind] < 1e-1) {
        Ux[ind] = 0.0;
        Uy[ind] = 0.0;
    } else {
        // 1. Calcolo del gamma locale basato sulla temperatura del paper
        real_t gamma_local = (3.0 * nu_r / h[ind]) * exp(-b * (T[ind] - Tr));

        // 2. Update ESPLICITO puramente convettivo e di pressione (senza l'attrito)
        real_t Ux_exp = Ux[ind] - dt * div_x[ind] + c_grad_Zh * h[ind] * (grad_Zx[ind] + grad_hx[ind]);
        real_t Uy_exp = Uy[ind] - dt * div_y[ind] + c_grad_Zh * h[ind] * (grad_Zy[ind] + grad_hy[ind]);

        // 3. Update SEMI-IMPLICITO dell'attrito per stabilità assoluta
        // L'equazione U_new = U_exp - dt * gamma * (U_new / h) diventa:
        real_t friction_factor = 1.0 + dt * (gamma_local / h[ind]);
        
        Ux[ind] = Ux_exp / friction_factor;
        Uy[ind] = Uy_exp / friction_factor;
    }
  }
};

// Compute (U/h beta U) Tensor
template<typename GVAR_t>
class compute_convective_flux {
  using idx_t = particles_t::idx_t;

  GVAR_t Fxx;
  GVAR_t Fxy;
  GVAR_t Fyy;
  const GVAR_t Ux;
  const GVAR_t Uy;
  const GVAR_t h;
  real_t beta;

public:
  compute_convective_flux(GVAR_t Fxx_, GVAR_t Fxy_, GVAR_t Fyy_, 
                          const GVAR_t Ux_, const GVAR_t Uy_, const GVAR_t h_, 
                          real_t beta_)
    : Fxx(Fxx_), Fxy(Fxy_), Fyy(Fyy_), Ux(Ux_), Uy(Uy_), h(h_), beta(beta_) {};

  DEVICE
  void
  operator()(idx_t ind) const {
    real_t hi = h[ind];
    
    if (hi > 1e-6) {
        real_t factor = beta / hi;
        Fxx[ind] = factor * Ux[ind] * Ux[ind];
        Fxy[ind] = factor * Ux[ind] * Uy[ind]; // Componente mista simmetrica
        Fyy[ind] = factor * Uy[ind] * Uy[ind]; 
    } else {
        Fxx[ind] = 0.0;
        Fxy[ind] = 0.0;
        Fyy[ind] = 0.0;
    }
  }
};

template<typename GVAR_t>
class extract_velocity {
  using idx_t = particles_t::idx_t;
  
  GVAR_t Vx;
  GVAR_t Vy;
  const GVAR_t Ux;
  const GVAR_t Uy;
  const GVAR_t h;
  
public:
  extract_velocity (GVAR_t Vx_, GVAR_t Vy_, const GVAR_t Ux_, const GVAR_t Uy_, const GVAR_t h_)
    : Vx(Vx_), Vy(Vy_), Ux(Ux_), Uy(Uy_), h(h_) {};
 
  DEVICE
  void
  operator()(idx_t ind) {
    if (h[ind] > 1e-1) {
        Vx[ind] = Ux[ind] / h[ind];
        Vy[ind] = Uy[ind] / h[ind];
    } else {
        Vx[ind] = 0.0;
        Vy[ind] = 0.0;
    }
  }  
};

#endif