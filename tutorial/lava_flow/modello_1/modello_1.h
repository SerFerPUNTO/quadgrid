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
  real_t dt; 
  real_t x_max;
  real_t y_max;

public :
  stepper (PVAR_t x_, PVAR_t y_,
	   const PVAR_t ux_, const PVAR_t uy_, PVAR_t Vp_,
	   real_t dt_, real_t x_max_, real_t y_max_)
    : x(x_), y(y_), ux(ux_), uy(uy_), Vp(Vp_), dt(dt_), x_max(x_max_), y_max(y_max_) { }

  DEVICE
  void operator() (int n) {
    x[n] += ux[n] * dt; 
    y[n] += uy[n] * dt; 
      
    if (x[n] < 0.0 || x[n] > x_max || y[n] < 0.0 || y[n] > y_max) {
        Vp[n] = 0.0; 
    }
  } 
  real_t get_dt() const { return dt; }
  void set_dt(real_t new_dt) { dt = new_dt; }
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

// Formula for U (con costanti pre-calcolate nel costruttore)
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
  const real_t c_grad_Z;
  const real_t c_grad_h;
  
public:
  evaluate_U (GVAR_t Ux_, GVAR_t Uy_, 
              const GVAR_t grad_Zx_, const GVAR_t grad_Zy_, 
              const GVAR_t grad_hx_, const GVAR_t grad_hy_, 
              const GVAR_t h_,
              const real_t tau_, const real_t gamma_, const real_t g_)
    : Ux(Ux_), Uy(Uy_), 
      grad_Zx(grad_Zx_), grad_Zy(grad_Zy_), 
      grad_hx(grad_hx_), grad_hy(grad_hy_), 
      h(h_), 
      c_grad(-tau_ * g_),      // Costante comune per -tau * g * h * grad(Z + h)
      c_gamma(-tau_ * gamma_)  // Costante per il termine di attrito -tau * gamma * grad(h)
  {};
 
  DEVICE
  void
  operator()(idx_t ind) {
    // -tau * g * h * (grad_Z + grad_h)  +  (-tau * gamma) * grad_h
    Ux[ind] = c_grad * h[ind] * (grad_Zx[ind] + grad_hx[ind]) + c_gamma * grad_hx[ind];
    Uy[ind] = c_grad * h[ind] * (grad_Zy[ind] + grad_hy[ind]) + c_gamma * grad_hy[ind];
  }  
};

#endif