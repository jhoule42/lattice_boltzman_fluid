"""Lid Driven Cavity simulation """

import numpy as np
import matplotlib.pyplot as plt
import time
import yaml
from alive_progress import alive_bar
import matplotlib.animation as animation


# Load simulation parameters as dictionary
with open("inputs.yaml", "r") as f:
    inputs = yaml.load(f, Loader=yaml.FullLoader)
    

path_array = inputs['path_array']
exec_name = inputs['exec_name']
save_exec = inputs['save_exec']
path_to_exec = inputs['path_to_exec']


# Parameters
gsize = 100
n, m = gsize, gsize
f = np.zeros((9, n+1, m+1))
feq = np.zeros((9, n+1, m+1))
rho = np.ones((n+1, m+1))
w = [ 4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36 ]
cx = [ 0, 1, 0, -1, 0, 1, -1, -1, 1 ]
cy = [ 0, 0, 1, 0, -1, 1, 1, -1, -1 ]
u = np.zeros((n+1, m+1))
v = np.zeros((n+1, m+1))


uo = 0.4
rho *= 1.00
dx = 1.0
dy = dx
Re = 200 # Reynolds Number
nu = uo * m / Re # dynamic viscosity
tau = (3*nu + 0.5)
omega = 1/tau
Ma = (dx/(m*np.sqrt(3)))*(omega-0.5)*Re
print(f"Re = {Re}, Ma = {Ma}, Tau = {tau}, nu = {nu}, Grid size = {n+1} x {m+1}")
mstep = 2000
errmax = 1e-6

u[:,m] = uo # Moving wall BC





def collision(u, v, f, feq, rho, omega, w, cx, cy, n, m):
    
    t1 = u[:,:]**2 + v[:,:]**2
    for k in range(9):
        
        t2 = u[:,:]*cx[k] + v[:,:]*cy[k]
        feq[k,:,:] = rho[:,:]*w[k]*(1.0 + 3.0*t2 + 4.50*t2*t2 - 1.50*t1)
        f[k,:,:] = omega*feq[k,:,:] + (1. - omega)*f[k,:,:]
    
    return feq, f
        

def streaming(f, n, m):

    # RIGHT TO LEFT
    for i in reversed(range(1, n+1)):
        f[1,i,:] = f[1,i-1,:]
    
    # LEFT TO RIGHT
    for i in range(n):
        f[3,i,:] = f[3, i+1, :]
    
    # TOP TO BOTTOM
    for j in reversed(range(1,m+1)):
        
        f[2,:,j] = f[2,:,j-1]

        for i in reversed(range(1, n+1)):
            f[5,i,j] = f[5,i-1,j-1]
        
        for i in range(n):
            f[6,i,j] = f[6, i+1, j-1]
    
    # BOTTOM TO TOP
    for j in range(m):
        
        f[4,:,j] = f[4,:,j+1]
        
        for i in range(n):
            f[7,i,j] = f[7, i+1, j+1]
        
        #for i in range(n, 0, -1): #n,0,-1 prevoiusly
        for i in reversed(range(1,n+1)):
            f[8,i,j] = f[8, i-1, j+1]
    
    return f
          
          
            
def sfbound(f, n, m, uo):
    
    # Bounce back on WEST boundary
    f[1,0,:] = f[3,0,:]
    f[5,0,:] = f[7,0,:]
    f[8,0,:] = f[6,0,:]
    
    # Bounce back on EAST boundary
    f[3,n,:] = f[1,n,:]
    f[7,n,:] = f[5,n,:]
    f[6,n,:] = f[8,n,:]
    
    # Bounce back on SOUTH boundary
    f[2,:,0] = f[4,:,0]
    f[5,:,0] = f[7,:,0]
    f[6,:,0] = f[8,:,0]
        
    # Moving lid, NORTH boundary
    rhon = f[0,:,m] + f[1,:,m] + f[3,:,m] + 2*(
            f[2,:,m] + f[6,:,m] + f[5,:,m])
        
    f[4,:,m] = f[2,:,m]
    f[8,:,m] = f[6,:,m] + rhon*uo/6 
    f[7,:,m] = f[5,:,m] - rhon*uo/6 
        
    return f



def rhouv(f, rho, u, v, cx, cy, n, m):    

    rho[:,:] = np.sum(f, axis=0) # sum over all k
    
    # North boundary
    rho[:,m] = f[0,:,m] + f[1,:,m] + f[3,:,m] + 2*(
            f[2,:,m] + f[6,:,m] + f[5,:,m])
    
    usum = np.zeros((n+1,m))
    vsum = np.zeros((n+1,m+1))
    
    for k in range(9):
        
        usum += f[k, :, :m]*cx[k]
        vsum += f[k, :, :]*cy[k]
    
    u[:, :m] = usum/rho[:,:m]
    v[:, :] = vsum/rho[:,:]
       
    return rho, u, v
           
           
           
           
# Save array
ux_arr = np.memmap(f'{path_array}/memmapped_ux.dat', dtype=np.float32,
              		mode='w+', shape=(mstep+1, gsize+1, gsize+1))
uy_arr = np.memmap(f'{path_array}/memmapped_uy.dat', dtype=np.float32,
              		mode='w+', shape=(mstep+1, gsize+1, gsize+1))


start_time = time.time()
du_plus_dv = np.zeros((n+1, m+1))


# Main Simulation Loop
print('\nStarting Simulation...')
with alive_bar(mstep) as bar:
    for it in range(1, mstep+1):

        du_plus_dv = -(u + v) # begin error computation
        
        # Execute the algorithm
        feq, f = collision(u, v, f, feq, rho, omega, w, cx, cy, n, m)
        f = streaming(f, n, m)
        f = sfbound(f, n, m, uo) 
        rho, u, v = rhouv(f, rho, u, v, cx, cy, n, m)
        
        du_plus_dv += (u + v) # continue error computation
        err = np.amax(abs(du_plus_dv))
        
        ux_arr[it] = u
        uy_arr[it] = v
        
        if err <= errmax:
            break
        
        bar() # Update the cool progress bar!

end_time = time.time()
print(f"Iterations = {it}, Time = {np.round(end_time-start_time,2)} s, error = {err}")


# Clean memory
del ux_arr
del uy_arr




# Load numpy array
print('Loading arrays.')
ux_arr = np.memmap(f'{path_array}/memmapped_ux.dat', dtype=np.float32,
                    shape=(mstep+1, gsize+1, gsize+1))
uy_arr = np.memmap(f'{path_array}/memmapped_uy.dat', dtype=np.float32,
                    shape=(mstep+1, gsize+1, gsize+1))



# # ANIMATION
# fig, ax = plt.subplots()
# ims = []
# xs = np.linspace(0, 1, m+1)
# ys = np.linspace(0, 1, m+1)
# x, y = np.meshgrid(xs, ys)

# for i in range(0,ux_arr.shape[0], 50):
#     print(i)
#     ux = np.rot90(ux_arr[i])
#     vy = np.rot90(uy_arr[i])
#     ux2 = ux_arr[i]
#     vy2 = uy_arr[i] 

#     im = ax.imshow(ux**2, animated=True, cmap='turbo', extent = [0, 1, 0, 1])
#     im = ax.quiver(y[::5], x[::5], ux2[::5], vy2[::5], color='black', animated=True)
#     ims.append([im])    
    
#     ax = plt.gca()
#     ax.set_aspect('equal')

# ani = animation.ArtistAnimation(fig, ims, interval=20, repeat=False, blit=False)
# ani.save('test_quiver.mp4')
# # plt.show()


u = np.rot90(u)
v = np.rot90(v)

xs = np.linspace(0, 1, m+1)
ys = np.linspace(0, 1, m+1)
x, y = np.meshgrid(xs, ys)


# Generate Figures
# Plot final ux and uy
plt.figure()
plt.imshow(u, cmap='turbo', extent=[0, 1, 0, 1])
plt.colorbar()
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title(r"$x$ velcocity")
plt.savefig('Figures/plot_ux', bbox_inches='tight')


# Plot final v
plt.figure()
plt.imshow(v, cmap='turbo', extent=[0, 1, 0, 1])
plt.colorbar()
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title(r"$y$ velcocity")
plt.savefig('Figures/plot_uy', bbox_inches='tight')


# Plot streamlines
plt.figure()
plt.streamplot(x, y, np.flipud(u), np.flipud(v), color='black')  #quiver
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title(r"Vector Flow Streamlines")
plt.savefig('Figures/plot_streamlines', bbox_inches='tight')


N = gsize + 1
half = int(np.floor(N/2))
vertical = u[:,half]
horizontal = v[half,:]

if N%2 == 0:
    vertical = (u[:,half] + u[:,half-1])/2
    horizontal = (v[half,:] + v[half-1,:])/2
    
# Vertical centerline at num half
plt.figure()
plt.plot(ys, vertical, 'r')
plt.plot(ys, vertical, 'k.')
plt.title(r"$v_x$ along the the vertical centerline")
plt.xlabel(r"$y$")
plt.ylabel(r"$v_x$")
plt.savefig('Figures/graph_u', bbox_inches='tight')


print(f"u min is {min(vertical)}")
plt.figure()
plt.plot(xs, horizontal, 'r')
plt.plot(xs, horizontal, 'k.')
plt.title(r"$v_x$ along the the horizontal centerline")
plt.xlabel(r"$x$")
plt.ylabel(r"$v_y$")
plt.savefig('Figures/graph_v', bbox_inches='tight')

    