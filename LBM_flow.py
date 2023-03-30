"""Latice-Boltzman simulation methods """

import timeit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, writers
from alive_progress import alive_bar
import yaml
import os
import shutil

from lattice import *



starttime = timeit.default_timer()

# Load simulation parameters as dictionary
with open("inputs.yaml", "r") as f:
    inputs = yaml.load(f, Loader=yaml.FullLoader)
    
Nx = inputs['Nx']
Ny = inputs['Ny']
Nt = inputs['Nt']
NL = inputs['NL']
flow_speed = inputs['flow_speed']
flow_direct = inputs['flow_direct']
tau = inputs['tau']
scen = inputs['scen']
path_array = inputs['path_array']
exec_name = inputs['exec_name']
save_exec = inputs['save_exec']
path_to_exec = inputs['path_to_exec']

D = 30
Re = (flow_speed * D) / tau  # Reynold number


# Create lattice object
lat = Lattice(Nx, Ny, tau, scen)

# Initialize density function
F = np.ones((Ny,Nx,NL))
F += 0.01*np.random.randn(Ny,Nx,NL) # add randomness to init conditions

# Initial flow direction (towards right)
F[:,:,3] = flow_speed


# =====================================
# 			  Add boundary
# =====================================

# Vortex Von Karman
lat.add_bd_circle(150, 150, D)

# # Cercles avec carre
# lat.add_bd_rectangle([20,30], [45,55])
# lat.add_bd_rectangle([40,50], [65,75])
# lat.add_bd_rectangle([60,70], [85,95])

# lat.add_bd_rectangle([80,90], [65,75])
# lat.add_bd_rectangle([100,110], [45,55])

# lat.add_bd_rectangle([80,90], [35,45])
# lat.add_bd_rectangle([60,70], [15,25])
# lat.add_bd_rectangle([40,50], [35,45])

# # Deux gros blocs
# lat.add_bd_rectangle([20,60], [0,40])
# lat.add_bd_rectangle([100, 140], [60,100])




# Save boundary
np.save(f'{path_array}/boundary', lat.bound)

ux_arr = np.memmap(f'{path_array}/memmapped_ux.dat', dtype=np.float32,
              		mode='w+', shape=(Nt, Ny, Nx))
uy_arr = np.memmap(f'{path_array}/memmapped_uy.dat', dtype=np.float32,
              		mode='w+', shape=(Nt, Ny, Nx))


# Simulation Main Loop
print(f"\nParameters: Grid:{Nx}x{Ny}, Steps:{Nt}, Re:{Re:.2f}")
print('\nStarting Simulation...')
with alive_bar(Nt) as bar:
	for it in range(Nt):

		# Zo-hoe absorbing boundary conditions
		F[:, -1, [6,7,8]] = F[:, -2, [6,7,8]]
		F[:,  0, [2,3,4]] = F[:,  1, [2,3,4]]
	
		# Drift
		for i, cx, cy in zip(lat.idxs, lat.cxs, lat.cys):
			F[:,:,i] = np.roll(F[:,:,i], cx, axis=1)
			F[:,:,i] = np.roll(F[:,:,i], cy, axis=0)


		# Boundary collisions, bounce towards the opposite nods
		bndryF = F[lat.bound, :]
		bndryF = bndryF[:, lat.opp_nodes]


		# Calculate fluid variables
		rho = np.sum(F, 2)
		lat.ux  = np.sum(F*lat.cxs, 2) / rho
		lat.uy  = np.sum(F*lat.cys, 2) / rho

		# Apply boundary
		F[lat.bound,:] = bndryF
		lat.ux[lat.bound] = 0
		lat.uy[lat.bound] = 0


		# Apply collision
		Feq = np.zeros(F.shape)
		for j, cx, cy, w in zip(range(lat.Q), lat.cxs, lat.cys, lat.weight):
			Feq[:,:,j] = rho*w*(1 + 3*(cx*lat.ux+cy*lat.uy) + 9*(cx*lat.ux+cy*lat.uy)**2 / 2 - 3*(lat.ux**2+lat.uy**2)/2 )
		F = F + -(1.0/lat.tau) * (F - Feq) 

		# Add new value to the array
		ux_arr[it] = lat.ux
		uy_arr[it] = lat.uy

		bar() # Update the cool progress bar!


# Saving simulation execution
if save_exec:

    if not os.path.exists(f'{path_to_exec}/{exec_name}'):
        os.makedirs(f'{path_to_exec}/{exec_name}')
        
    shutil.copyfile('inputs.yaml', f'{path_to_exec}/{exec_name}/inputs.yaml')
    shutil.copytree(f'{path_array}', f'{path_to_exec}/{exec_name}',
                    dirs_exist_ok=True)
        
# Clean memory
del ux_arr
del uy_arr

print(f"\nTime execution: {(timeit.default_timer()-starttime)/60:.4f} min")