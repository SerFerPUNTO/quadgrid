import pyvista as pv
import numpy as np
import pandas as pd
import json
import os

# 1. Configurazione del sottocampionamento
# step = 1 -> Risoluzione originale (4096 x 4096 celle)
# step = 4 -> Risoluzione media (1024 x 1024 celle)
# step = 8 -> Risoluzione bassa (512 x 512 celle - OTTIMA PER DEBUG)
step = 8 # funziona solo con potenze di 2

# 2. Lettura del file originale
print("Reading VTU file...")
mesh = pv.read('/home/federico/quadgrid/tutorial/lava_flow/etna/elevazione.vtu')
points = mesh.points
z_raw = mesh.point_data['Z']

# 3. Ordinamento spaziale e pulizia dei nodi
print("Building DataFrame...")
df = pd.DataFrame({'x': points[:, 0], 'y': points[:, 1], 'z': z_raw})

# 1. Rimuoviamo i nodi sovrapposti (duplicati geometrici)
punti_prima = len(df)
df = df.drop_duplicates(subset=['x', 'y'], keep='first')
punti_dopo = len(df)
print(f"Removed {punti_prima - punti_dopo} duplicated VTU nodes.")

# 2. Ordiniamo rigorosamente riga per riga
print("Nodes spatial ordering...")
df = df.sort_values(by=['y', 'x']).reset_index(drop=True)

x_unique = np.unique(df['x'])
y_unique = np.unique(df['y'])
nodes_x_orig = len(x_unique)
nodes_y_orig = len(y_unique)

print(f"Unique nodes: {nodes_x_orig} x {nodes_y_orig} = {nodes_x_orig * nodes_y_orig}")
print(f"Dataset nodes: {len(df)}")

if len(df) != nodes_x_orig * nodes_y_orig:
    # Se questo dovesse scattare, significa che la mesh originale 
    # non è un rettangolo ma ha dei "buchi" (nodi mancanti).
    raise ValueError("Error: The nodes don't create a rectangular grid")

# 3. Rimodelliamo in matrice 2D in totale sicurezza
Z_2D = df['z'].values.reshape((nodes_y_orig, nodes_x_orig))

# 4. Sottocampionamento (Coarsening)
# Estraiamo un nodo ogni 'step' lungo entrambe le direzioni
Z_coarse = Z_2D[::step, ::step]

# Calcoliamo le nuove dimensioni
nodes_y_new, nodes_x_new = Z_coarse.shape
ncols_new = nodes_x_new - 1
nrows_new = nodes_y_new - 1

# Ricalcoliamo i passi hx e hy
hx_new = (np.max(x_unique) - np.min(x_unique)) / ncols_new
hy_new = (np.max(y_unique) - np.min(y_unique)) / nrows_new

print(f"New generated grid: {ncols_new} x {nrows_new} celle")
print(f"New hx: {hx_new:.2f} m | New hy: {hy_new:.2f} m")

# Calcolo del centro della mappa per posizionare il cratere in modo sicuro
x_center = float(np.min(x_unique) + (ncols_new * hx_new) / 2.0)
y_center = float(np.min(y_unique) + (nrows_new * hy_new) / 2.0)

# 5. Generazione del JSON
sim_data = {
    "grid_properties": {
        "nx": int(ncols_new),
        "ny": int(nrows_new),
        "hx": float(hx_new),
        "hy": float(hy_new),
        "x_min": float(np.min(x_unique)),
        "y_min": float(np.min(y_unique))
    },
    "physics_parameters": {
        "g": 9.81,
        "tau": 1000.0,
        "gamma": 0.5,
        "dt": 0.01,
        "t_max": 0.02,
        "nsave": 10
    },
    "eruption_parameters": {
        "x_vent": x_center,  # Ora usa il centro calcolato dinamicamente
        "y_vent": y_center,  # Ora usa il centro calcolato dinamicamente
        "radius_vent": 500.0, # Aumentato leggermente per compensare griglie coarse
        "h_initial_lava": 5.0,
        "particles_per_cell_x": 2,
        "particles_per_cell_y": 2
    },
    "grid_vars": {
        "Z": Z_coarse.flatten().tolist() 
    }
}

# --- IMPOSTA LA TUA CARTELLA QUI ---
cartella_destinazione = "/home/federico/quadgrid/tutorial/lava_flow/modello_1"

# Crea la cartella se non esiste già
os.makedirs(cartella_destinazione, exist_ok=True)

# Sostituito lava_config_step{step}.json con lava_config.json per interfacciarsi con il C++
filename = os.path.join(cartella_destinazione, 'lava_config.json')

with open(filename, 'w') as f:
    json.dump(sim_data, f, indent=4)

print(f"File succesfully generated and saved in:\n{filename}")