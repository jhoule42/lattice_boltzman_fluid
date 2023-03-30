""" Animate the plot functions """

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, writers
import matplotlib.animation as animation
from alive_progress import alive_bar
import yaml


# Load simulation parameters as dictionary
print('Loading parameters.')
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
path_to_save = inputs['path_to_save']
exec_name = inputs['exec_name']

plot='curl'



# Load numpy array
print('Loading arrays.')
ux_arr = np.memmap(f'{path_array}/memmapped_ux.dat', dtype=np.float32,
                    shape=(Nt, Ny, Nx))
uy_arr = np.memmap(f'{path_array}/memmapped_uy.dat', dtype=np.float32,
                    shape=(Nt, Ny, Nx))
boundary = np.load('arr/boundary.npy')



# Save specific frames to compare simulations
frame_nb = 1999
ux_arr[frame_nb][boundary] = 0
uy_arr[frame_nb][boundary] = 0

curl = ux_arr[frame_nb]**2 + uy_arr[frame_nb]**2
curl[boundary] = np.nan
curl = np.ma.array(curl, mask=boundary)

fig, ax = plt.subplots()
ax.imshow(curl, cmap='inferno')
ax = plt.gca()
ax.invert_yaxis()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)	
ax.set_aspect('equal')	
plt.savefig(f'Figures/{exec_name}_frame', bbox_inches='tight')




# Creation de l'animation
fig, ax = plt.subplots()
ims = []

with alive_bar(Nt//20) as bar:
    for i in range(0,ux_arr.shape[0], 20):

        ux_arr[i][boundary] = 0
        uy_arr[i][boundary] = 0
        
        if plot =='vorticity':
            vorticity = (np.roll(ux_arr[i], -1, axis=0) - np.roll(ux_arr[i], 1, axis=0)) - (np.roll(ux_arr[i], -1, axis=1) - np.roll(ux_arr[i], 1, axis=1))
            vorticity[boundary] = np.nan
            vorticity = np.ma.array(vorticity, mask=boundary)

            ax.clim(-.1, .1)
            im = ax.imshow(vorticity, animated=True, cmap='bwr')
            ims.append([im])
            

        elif plot =='curl':
            curl = ux_arr[i]**2 + uy_arr[i]**2
            curl[boundary] = np.nan
            curl = np.ma.array(curl, mask=boundary)
            
            im = ax.imshow(curl, animated=True, cmap='inferno')
            ims.append([im])


        ax = plt.gca()
        ax.invert_yaxis()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)	
        ax.set_aspect('equal')	
        # plt.pause(0.0001)

        bar() # Update the cool progress bar!

     
ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True, repeat=False)
ani.save(f'{path_to_save}/{exec_name}.mp4')
plt.show()