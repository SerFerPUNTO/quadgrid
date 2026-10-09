#include "modello_1.h"
#include "init_scenario.h"
#include <thrust/extrema.h>
#include <thrust/iterator/transform_iterator.h>

#ifndef THRUST_CPU
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuSetDevice hipSetDevice
#define gpuGetDevice hipGetDevice
#else
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuSetDevice cudaSetDevice
#define gpuGetDevice cudaGetDevice
#endif
#endif

int main(){

#ifndef THRUST_CPU
 int num_gpus;
 auto err = gpuGetDeviceCount (&num_gpus); if (err) return err;
 std::cerr << "num_gpus=" << num_gpus <<std::endl;
 int device;
 err = gpuSetDevice (num_gpus - 1); if (err) return err;
 err = gpuGetDevice (&device); if (err) return err;
 std::cerr << "running on gpu n. " << device << std::endl;
#endif
 
 using idx_t = particles_t::idx_t;

 cdf::timer::timer_t timer;
 
 // 1. Apre e legge il file JSON
 std::ifstream ifs("lava_config.json");
 if (!ifs.is_open()) {
     std::cerr << "Errore: impossibile aprire il file JSON." << std::endl;
     return 1;
 }
 nlohmann::json j;
 ifs >> j;
 ifs.close();

 // 2. Costruisce la griglia Euleriana usando il nodo "grid_properties"
 quadgrid_t<vector_t<real_t>> qg(j["grid_properties"]);

 // 3. Estrae i campi nodali iniziali (inclusa la topografia Z)
 std::map<std::string, vector_t<real_t>> vars = 
     j["grid_vars"].get<std::map<std::string, vector_t<real_t>>>();

 // 4. Inizializza le particelle lagrangiane nel cratere
 particles_t p = initialize_crater(qg, j);

 // 5. Carica la topografia Z e altre eventuali variabili iniziali sulla GPU
 for (const auto & g : vars) {
     p.device_grid_vars[g.first] = g.second;
 }

 // Estrazione delle proprietà della griglia e dei parametri dal JSON
 idx_t nrows = j["grid_properties"]["ny"];
 idx_t ncols = j["grid_properties"]["nx"];
 real_t hx = j["grid_properties"]["hx"];
 real_t hy = j["grid_properties"]["hy"];

 // FORZIAMO L'ORIGINE LOCALE A 0.0 (Quadgrid ignora i valori geografici)
 real_t x_min = 0.0;
 real_t y_min = 0.0;

 real_t dt = j["physics_parameters"]["dt"];
 real_t max_dt = j["physics_parameters"]["max_dt"]; 
 real_t tau = j["physics_parameters"]["tau"];
 real_t gamma = j["physics_parameters"]["gamma"];
 real_t g = j["physics_parameters"]["g"];
 real_t tmax = j["physics_parameters"]["t_max"];
 int nsave = j["physics_parameters"]["nsave"];

 // I bordi massimi sono semplicemente nx * hx e ny * hy
 real_t x_max_domain = ncols * hx;
 real_t y_max_domain = nrows * hy;

stepper step(
   thrust::raw_pointer_cast(p.device_x.data()), 
   thrust::raw_pointer_cast(p.device_y.data()), 
   thrust::raw_pointer_cast(p.device_dprops["vx"].data()), 
   thrust::raw_pointer_cast(p.device_dprops["vy"].data()), 
   thrust::raw_pointer_cast(p.device_dprops["Vp"].data()), 
   dt, x_min, x_max_domain, y_min, y_max_domain // Passati i nuovi bordi
 );

 idx_t num_nodi = (nrows + 1) * (ncols + 1);

 // Allocazione memoria in VRAM per gradienti e campi ausiliari
 p.device_grid_vars["grad_Zx"].resize(num_nodi, 0.0);
 p.device_grid_vars["grad_Zy"].resize(num_nodi, 0.0);
 p.device_grid_vars["grad_hx"].resize(num_nodi, 0.0);
 p.device_grid_vars["grad_hy"].resize(num_nodi, 0.0);
  
 p.device_grid_vars["h"].resize(num_nodi, 0.0);
 p.device_grid_vars["Ux"].resize(num_nodi, 0.0); //portata
 p.device_grid_vars["Uy"].resize(num_nodi, 0.0);
 p.device_grid_vars["Vx"].resize(num_nodi, 0.0); //velocità
 p.device_grid_vars["Vy"].resize(num_nodi, 0.0);

 evaluate_grad eval_grad_Z(
   thrust::raw_pointer_cast(p.device_grid_vars["grad_Zx"].data()),
   thrust::raw_pointer_cast(p.device_grid_vars["grad_Zy"].data()),
   thrust::raw_pointer_cast(p.device_grid_vars["Z"].data()),
   nrows, ncols, hx, hy
 );

 idx_t num_nodes = (nrows + 1) * (ncols + 1);
 thrust::counting_iterator<idx_t> first_p(0);
 thrust::counting_iterator<idx_t> last_p(p.num_particles);
 thrust::counting_iterator<idx_t> first_node(0);
 thrust::counting_iterator<idx_t> last_node(num_nodi);

 // Calcolo iniziale del gradiente della topografia (Z)
 thrust::for_each(first_node, last_node, eval_grad_Z);

 evaluate_grad eval_grad_h(
   thrust::raw_pointer_cast(p.device_grid_vars["grad_hx"].data()),
   thrust::raw_pointer_cast(p.device_grid_vars["grad_hy"].data()),
   thrust::raw_pointer_cast(p.device_grid_vars["h"].data()),
   nrows, ncols, hx, hy
 );

 evaluate_U eval_U(
   thrust::raw_pointer_cast(p.device_grid_vars["Ux"].data()),
   thrust::raw_pointer_cast(p.device_grid_vars["Uy"].data()),
   thrust::raw_pointer_cast(p.device_grid_vars["grad_Zx"].data()),
   thrust::raw_pointer_cast(p.device_grid_vars["grad_Zy"].data()),
   thrust::raw_pointer_cast(p.device_grid_vars["grad_hx"].data()),
   thrust::raw_pointer_cast(p.device_grid_vars["grad_hy"].data()),
   thrust::raw_pointer_cast(p.device_grid_vars["h"].data()),
   tau, gamma, g
);

extract_velocity get_kinematics(
   thrust::raw_pointer_cast(p.device_grid_vars["Vx"].data()),
   thrust::raw_pointer_cast(p.device_grid_vars["Vy"].data()),
   thrust::raw_pointer_cast(p.device_grid_vars["Ux"].data()),
   thrust::raw_pointer_cast(p.device_grid_vars["Uy"].data()),
   thrust::raw_pointer_cast(p.device_grid_vars["h"].data())
 );
 
 // Inizializzazione del cratere
 real_t Q_vent = j["eruption_parameters"]["Q"];
 real_t total_volume_to_erupt = Q_vent * tmax;
 real_t Vp_active = total_volume_to_erupt / static_cast<real_t>(p.num_particles);
 int next_particle_to_activate = 0; //indice della particella da attivare

 double dtsave = tmax / nsave;
 double t = 0.;

 // Lambda function per centralizzare la logica di I/O e salvataggio file
 auto save_state = [&](int step_num) {
     p.p2g(p.device_grid_vars, {"Vp"}, {"h"}, false);
     thrust::transform(p.device_grid_vars["h"].begin(), p.device_grid_vars["h"].end(),
                       p.device_grid_M.begin(), p.device_grid_vars["h"].begin(),
                       safe_divide());
     p.memcpy_device_to_host();
     
     vars["h"].resize(num_nodes);
     thrust::copy(p.device_grid_vars["h"].cbegin(), p.device_grid_vars["h"].cend(), vars.at("h").begin());

     const std::string numfile = std::string(".") + std::to_string(step_num);
     std::ofstream outbuf("particle" + numfile + ".csv");  
     p.print<particles_t::output_format::csv>(outbuf);
     outbuf.close();
     
     qg.vtk_export(("grid" + numfile + ".vts").c_str(), vars);
 };
 
 // Ciclo di salvataggio principale
 for(int isave = 0; isave < nsave; ++isave){
  
  save_state(isave);

  // Time Stepping Loop
  while(t < dtsave * (isave + 1)){  
      thrust::fill(p.device_grid_vars["h"].begin(), p.device_grid_vars["h"].end(), 0.0);
      // Nota: Ux e Uy non richiedono thrust::fill poiché vengono sovrascritti interamente da evaluate_U

      // INIEZIONE CONTINUA (PARTICLE ACTIVATION)
      // Calcola quante particelle DEVONO essere accese a questo istante di tempo
      real_t expected_volume = Q_vent * t; //Quanto ce ne aspettiamo in totale
      int expected_active = std::min(static_cast<int>(expected_volume / Vp_active), p.num_particles); //Quante particelle
      
      int n_to_activate = expected_active - next_particle_to_activate; //Rendiamo un delta
      
      // Se è il momento di attivare un nuovo pacchetto, accendiamole in VRAM
      if (n_to_activate > 0) {
          thrust::fill(
              p.device_dprops["Vp"].begin() + next_particle_to_activate,
              p.device_dprops["Vp"].begin() + next_particle_to_activate + n_to_activate,
              Vp_active //Fissiamo il volume delle particelle da attivare a Vp_active
          );
          next_particle_to_activate += n_to_activate; //Aggiorniamo per lo step successivo
      }

      // P2G of Vp
      timer.tic("p2g");
      p.p2g(p.device_grid_vars, {"Vp"}, {"h"}, false); 

      // Safe Divide: h = Vol_nodo / M_nodo
      thrust::transform(p.device_grid_vars["h"].begin(), p.device_grid_vars["h"].end(),
                        p.device_grid_M.begin(),
                        p.device_grid_vars["h"].begin(),
                        safe_divide());
      timer.toc("p2g");
  
      // Eval Ux and Uy
      thrust::for_each(first_node, last_node, eval_grad_h);
      thrust::for_each(first_node, last_node, eval_U);
      
      // Eval Vx and Vy
      thrust::for_each(first_node, last_node, get_kinematics);
      // G2P of Vx and Vy
      timer.tic("g2p");
      p.g2p(p.device_grid_vars, {"Vx", "Vy"}, {"vx", "vy"});
      timer.toc("g2p");

      // Compute dt (CFL condition)
      timer.tic("dt");
      auto abs_vx = thrust::make_transform_iterator(p.device_dprops["vx"].begin(), abs_no_nan());
      auto abs_vy = thrust::make_transform_iterator(p.device_dprops["vy"].begin(), abs_no_nan());
      real_t maxvx = *thrust::max_element(abs_vx, abs_vx + p.num_particles);
      real_t maxvy = *thrust::max_element(abs_vy, abs_vy + p.num_particles);
         
      real_t dtx = maxvx > 0.0 ? .5 * hx / maxvx : max_dt;
      real_t dty = maxvy > 0.0 ? .5 * hy / maxvy : max_dt;
      
      // Prendi il più piccolo tra dtx, dty e il tetto massimo
      step.set_dt(fmin(fmin(dtx, dty), max_dt));
      
      if (t + step.get_dt() > dtsave * (isave + 1)) {
          step.set_dt(dtsave * (isave + 1) - t);
      }
      timer.toc("dt");

      // Moving Particles
      timer.tic("move particles");
      thrust::for_each(thrust::device, first_p, last_p, step);
      p.update_ptcl_to_grd<particles_t::update_ptcl_to_grd_device>();
      timer.toc("move particles");
      
      t += step.get_dt();
  }
 }
  
 // Salvataggio finale dello stato
 save_state(nsave);
 
 timer.print_report();
 return 0; 
}