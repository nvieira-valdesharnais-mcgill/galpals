#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 14:13:27 2018

@authors: Nicholas Vieira and Valérie Desharnais
Physics Hackathon 2018

"""

# Imports
# Third-party
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

# Gala
from gala.mpl_style import mpl_style
plt.style.use(mpl_style)
import gala.dynamics as gd
import gala.potential as gp
from gala.units import galactic

# Animation
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D

# ** DEFAULTS **
# clockwise 1, widdershins -1, angular thing
c1 = 1.0
c2 = -1.0
w1 = 1.0
w2 = 1.0 

# Milky Way and Andromeda
M_MW = 1.0E11 * u.Msun
M_AND = 1.23E12 * u.Msun
SIZE_MW = 36.8 # kpc
SIZE_AND = 33.7 # kpc
MW_AND_DIST = 778.0/sqrt(3)
MW_AND_VEL = -110.0/sqrt(3)

POT_MW = gp.MilkyWayPotential()
POT_AND = gp.HernquistPotential(M_AND.value,0,units=galactic)
ICS_MW = gd.PhaseSpacePosition(pos=[0,0,0] * u.kpc, vel=[0,0,0] * u.km/u.s)
ICS_AND = gd.PhaseSpacePosition(pos=[MW_AND_DIST,MW_AND_DIST,MW_AND_DIST] * u.kpc, 
                                vel=[MW_AND_VEL,MW_AND_VEL,MW_AND_VEL] * (u.km/u.s))

# Lörg and Smøl Magellanic Clouds
M_LORG = 1.0E10 * u.M_sun
SIZE_LORG = 4.3 # u.kpc
MW_LORG_DIST = 49.97 # wrt MW, in kpc
MW_LORG_VEL= 321.0 # wrt MW, in km/s

POT_LORG = gp.HernquistPotential(M_LORG.value,0,units=galactic)
ICS_LORG = gd.PhaseSpacePosition(pos=[MW_LORG_DIST,MW_LORG_DIST,MW_LORG_DIST] * u.kpc, 
                                vel=[-MW_LORG_VEL,0,0] * (u.km/u.s))

M_smol = 7.0E9 * u.Msun
SIZE_smol = 2.15 # kpc
MW_smol_DIST = 60.6
MW_smol_VEL= 217.0

POT_smol = gp.HernquistPotential(M_smol.value,0,units=galactic)
ICS_smol = gd.PhaseSpacePosition(pos=[MW_smol_DIST,MW_smol_DIST,MW_smol_DIST] * u.kpc, 
                                vel=[-MW_smol_VEL,0,0] * (u.km/u.s))

# ** SET POTENTIAL **

# Potentials to play with
# Last 2 involve dark matter
pot1_mw = gp.MilkyWayPotential()
pot2_mw = gp.MilkyWayPotential()
pot1_nfw = gp.NFWPotential.from_circular_velocity(v_c=200*u.km/u.s, 
                                             r_s=10.*u.kpc,units=galactic)
pot2_nfw = gp.NFWPotential.from_circular_velocity(v_c=200*u.km/u.s, 
                                             r_s=10.*u.kpc,units=galactic)
pot1_ls = gp.LeeSutoTriaxialNFWPotential(v_c=200*u.km/u.s, r_s=1000.*u.kpc, 
                                      a=100*u.kpc, b=70*u.kpc, c=40*u.kpc,
                                      units=galactic)
pot2_ls = gp.LeeSutoTriaxialNFWPotential(v_c=200*u.km/u.s, r_s=1000.*u.kpc, 
                                      a=100*u.kpc, b=70*u.kpc, c=40*u.kpc,
                                      units=galactic)

# ** INITIAL CONDITIONS **
# initial conditions for each galaxy

# ask user what they want to see 
decision = input("What do you want to see? \n"+
                 "Enter AND to see the Milky Way and Andromeda.\n"+
                 "Enter ANDCLOSE to see the Milky Way and Andromeda, half as far apart.\n"+
                 "Enter LMC to see the the LÖRG Magellanic Cloud and the Milky Way.\n"+
                 "Enter SMC to see the the smøl Magellanic Cloud and the Milky Way.\n"+
                 "Hit enter to see 2 Milky Way-like galaxies. Used for testing.\n\n>>> ")
decision2 = float(input("Are you tilted bro? Give an x angle in degrees.\n\n>>> "))
decision3 = float(input("What about in y bro? You titled? Degrees pls.\n\n>>> "))

if (decision in ["AND","and","a","A"]):
    pot1 = POT_MW
    pot2 = POT_AND
    ics1 = ICS_MW
    ics2 = ICS_AND
    std_dev1 = SIZE_MW/3.0
    std_dev2 = SIZE_AND/3.0
elif (decision in ["ANDCLOSE","ac"]):
    pot1 = POT_MW
    pot2 = POT_AND
    ics1 = ICS_MW
    ics2 = gd.PhaseSpacePosition(pos=[MW_AND_DIST/2.0,
                                         MW_AND_DIST/2.0,
                                         MW_AND_DIST/2.0] * u.kpc, 
                                vel=[MW_AND_VEL,MW_AND_VEL,MW_AND_VEL] * (u.km/u.s))
    std_dev1 = SIZE_MW/3.0
    std_dev2 = SIZE_AND/3.0
elif (decision in ["LMC","lmc","LORG","lorg","l","L","LÖRG"]):
    pot1 = POT_MW
    pot2 = POT_LORG
    ics1 = ICS_MW
    ics2 = ICS_LORG
    std_dev1 = SIZE_MW/3.0
    std_dev2 = SIZE_LORG/3.0
elif (decision in ["SMC","smc","SMOL","smol","s","S","smøl"]):
    pot1 = POT_MW
    pot2 = POT_smol
    ics1 = ICS_MW
    ics2 = ICS_smol
    std_dev1 = SIZE_MW/3.0
    std_dev2 = SIZE_smol/3.0
else:
    pot1 = pot1_mw #+ pot1_ls
    pot2 = pot2_mw #+ pot2_ls
    ics1 = gd.PhaseSpacePosition(pos=[0,0,0] * u.kpc, vel=[0,0,0] * u.km/u.s)
    rand_xyz = list(np.random.randint(0,200,2))
    rand_xyz.append(0)
    rand_vxyz = list(np.random.randint(-200,200,2))
    rand_vxyz.append(np.random.randint(-20,20))
    ics2 = gd.PhaseSpacePosition(pos=rand_xyz * u.kpc, 
                             vel=rand_vxyz * u.km/u.s)
    std_dev1 = 10.0
    std_dev2 = 10.0

N1 = 1000 # the no. of stars in galaxy 1
N2 = N1 # the no. of stars in galaxy 2

# gaussian distribute points/velocities about galaxy 1
gaussboipos1 = np.random.normal(ics1.pos.xyz.value, std_dev1, 
                            size=(N1,3)).T*u.kpc
gaussboivel1 =  np.random.normal(ics1.vel.d_xyz.to(u.km/u.s).value, 1., 
                                size=(N1,3)).T * u.km/u.s
new_ics1 = gd.PhaseSpacePosition(pos=gaussboipos1, vel=gaussboivel1)

x1_vel_shifts = ([c1*w1*(gaussboipos1[0][i].value - ics1.x.value) for 
                  i in range(N1)]* u.km/u.s).to(u.kpc/u.Myr)
y1_vel_shifts = ([-w1*(gaussboipos1[1][i].value - ics1.y.value) for 
                  i in range(N1)]* u.km/u.s).to(u.kpc/u.Myr)
new_x1_vels = new_ics1.vel.d_x + x1_vel_shifts
new_y1_vels = new_ics1.vel.d_y + y1_vel_shifts
new_vels1 = [new_x1_vels, new_y1_vels, new_ics1.vel.d_z] * (u.km/u.s)
new_ics1 = gd.PhaseSpacePosition(pos=gaussboipos1, vel=new_vels1)

# gaussian distribute points/velocities about galaxy 2
gaussboipos2 = np.random.normal(ics2.pos.xyz.value, std_dev2, 
                            size=(N2,3)).T*u.kpc
gaussboivel2 =  np.random.normal(ics2.vel.d_xyz.to(u.km/u.s).value, 1., 
                                size=(N2,3)).T * u.km/u.s
new_ics2 = gd.PhaseSpacePosition(pos=gaussboipos2, vel=gaussboivel2)

x2_vel_shifts = ([c2*w2*(gaussboipos2[0][i].value - ics2.x.value) for 
                  i in range(N2)]* u.km/u.s).to(u.kpc/u.Myr)
y2_vel_shifts = ([-w2*(gaussboipos2[1][i].value - ics2.y.value) for 
                  i in range(N2)]* u.km/u.s).to(u.kpc/u.Myr)
new_x2_vels = new_ics2.vel.d_x + x2_vel_shifts
new_y2_vels = new_ics2.vel.d_y + y2_vel_shifts
new_vels2 = [new_x2_vels, new_y2_vels, new_ics2.vel.d_z] * (u.km/u.s)
new_ics2 = gd.PhaseSpacePosition(pos=gaussboipos2, vel=new_vels2)

# ** GALACTIC TILT **  
thx = decision2 * np.pi/180.0
thy = decision3 * np.pi/180.0

new_x_2, new_y_2, new_z_2 = [], [], []
new_vx_2, new_vy_2, new_vz_2 = [], [], []
for n in  range(N1):
    #positions
    new_z_1 = (ics2.pos.y.value+np.sin(thx)*(new_ics2.pos[n].y.value-ics2.pos.y.value))+(ics2.pos.z.value+np.cos(thx)*(new_ics2.pos[n].z.value-ics2.pos.z.value))
    new_x_2.append((ics2.pos.x.value+np.cos(thy)*(new_ics2.pos[n].x.value-ics2.pos.x.value))+(ics2.pos.z.value+np.sin(thy)*(new_z_1-ics2.pos.z.value)))
    new_y_2.append((ics2.pos.y.value+np.cos(thx)*(new_ics2.pos[n].y.value-ics2.pos.y.value))+(ics2.pos.z.value-np.sin(thx)*(new_ics2.pos[n].z.value-ics2.pos.z.value)))
    new_z_2.append((ics2.pos.x.value-np.sin(thy)*(new_ics2.pos[n].x.value-ics2.pos.x.value))+(ics2.pos.z.value+np.cos(thy)*(new_z_1-ics2.pos.z.value)))
    #velocities
    new_vz_1 = (ics2.vel.d_y.value+np.sin(thx)*(new_ics2.vel[n].d_y.value-ics2.vel.d_y.value))+(ics2.vel.d_z.value+np.cos(thx)*(new_ics2.vel[n].d_z.value-ics2.vel.d_z.value))
    new_vx_2.append((ics2.vel.d_x.value+np.cos(thy)*(new_ics2.vel[n].d_x.value-ics2.vel.d_x.value))+(ics2.vel.d_z.value+np.sin(thy)*(new_vz_1-ics2.vel.d_z.value)))
    new_vy_2.append((ics2.vel.d_y.value+np.cos(thx)*(new_ics2.vel[n].d_y.value-ics2.vel.d_y.value))+(ics2.vel.d_z.value-np.sin(thx)*(new_ics2.vel[n].d_z.value-ics2.vel.d_z.value)))
    new_vz_2.append((ics2.vel.d_x.value-np.sin(thy)*(new_ics2.vel[n].d_x.value-ics2.vel.d_x.value))+(ics2.vel.d_z.value+np.cos(thy)*(new_vz_1-ics2.vel.d_z.value)))

new_ics2 = gd.PhaseSpacePosition(pos=[new_x_2,new_y_2,new_z_2]*u.kpc, vel=[new_vx_2,new_vy_2,new_vz_2]*(u.km/u.s))

all_x = []
all_y = []
all_z = []

# ** DYNAMICS ** 
steps_per_iter = 1 # time steps per orbit iteration
timestep = 1.0 # time step in Myr
end = 3000  # number of times to iterate orbits
for i in range(end):
    
    # integrate orbit for galaxy 1 in potential 1
    orbits1 = gp.Hamiltonian(pot1).integrate_orbit(new_ics1, dt=timestep, 
                           n_steps=steps_per_iter) 
    new_ics1 = gd.PhaseSpacePosition(orbits1[-1].pos,orbits1[-1].vel)
    
    # integrate orbit for galaxy 2 in potential 1
    orbits2 = gp.Hamiltonian(pot1).integrate_orbit(new_ics2, dt=timestep, 
                           n_steps=steps_per_iter) 
    new_ics2 = gd.PhaseSpacePosition(orbits2[-1].pos,orbits2[-1].vel)
    
    # compute relative motion of galaxy 2 wrt galaxy 1
    # positions
    x_coords2 = (orbits2[-1].pos.xyz.value[0]) * u.kpc
    x_mean2 = np.mean(x_coords2)
    y_coords2 = (orbits2[-1].pos.xyz.value[1]) * u.kpc
    y_mean2 = np.mean(y_coords2)
    z_coords2 = (orbits2[-1].pos.xyz.value[2]) * u.kpc
    z_mean2 = np.mean(z_coords2)
    
    # velocities
    vx_coords2 = (orbits2[-1].vel.d_xyz.value[0]) * u.kpc/u.Myr
    vx_mean2 = np.mean(vx_coords2)
    vy_coords2 = (orbits2[-1].vel.d_xyz.value[1]) * u.kpc/u.Myr
    vy_mean2 = np.mean(vy_coords2)
    vz_coords2 = (orbits2[-1].vel.d_xyz.value[2]) * u.kpc/u.Myr
    vz_mean2 = np.mean(vz_coords2)
    
    # move galaxy 1 away from origin according to distance to galaxy 2
    new_pos = (orbits1[-1].pos.xyz.value + [[x_mean2.value]*N1, 
                       [y_mean2.value]*N1, 
                       [z_mean2.value]*N1]) * u.kpc
    
    new_vel = (orbits1[-1].vel.d_xyz.value + [[vx_mean2.value]*N1, 
                       [vy_mean2.value]*N1, 
                       [vz_mean2.value]*N1]) * u.kpc/u.Myr
    new_ics1 = gd.PhaseSpacePosition(new_pos,new_vel)
    
    # integrate orbit for galaxy 1 in potential 2
    orbits1 = gp.Hamiltonian(pot2).integrate_orbit(new_ics1, dt=timestep, 
                           n_steps=steps_per_iter)
    new_ics1 = gd.PhaseSpacePosition(orbits1[-1].pos,orbits1[-1].vel) 
    
    # move galaxy 1 back to origin
    new_pos = (orbits1[-1].pos.xyz.value - [[x_mean2.value]*N1, 
                       [y_mean2.value]*N1, 
                       [z_mean2.value]*N1]) * u.kpc
    new_vel = (orbits1[-1].vel.d_xyz.value - [[vx_mean2.value]*N1, 
                       [vy_mean2.value]*N1, 
                       [vz_mean2.value]*N1]) * u.kpc/u.Myr
    new_ics1 = gd.PhaseSpacePosition(new_pos,new_vel)
     
    # reintegrate orbit for galaxy 1 in potential 1
    orbits1 = gp.Hamiltonian(pot1).integrate_orbit(new_ics1, dt=2., 
                           n_steps=steps_per_iter) 
    new_ics1 = gd.PhaseSpacePosition(orbits1[-1].pos,orbits1[-1].vel) 

    # compute relative motion of galaxy 1 wrt galaxy 2
    # positions
    x_coords1 = (new_ics1.xyz.value[0]) * u.kpc
    x_mean1 = np.mean(x_coords1)
    y_coords1 = (new_ics1.xyz.value[1]) * u.kpc
    y_mean1 = np.mean(y_coords1)
    z_coords1 = (new_ics1.xyz.value[2]) * u.kpc
    z_mean1 = np.mean(z_coords1)
    # velocities
    vx_coords1 = (new_ics1.vel.d_xyz.value[0]) * u.kpc/u.Myr
    vx_mean1 = np.mean(vx_coords1)
    vy_coords1 = (new_ics1.vel.d_xyz.value[1]) * u.kpc/u.Myr
    vy_mean1 = np.mean(vy_coords1)
    vz_coords1 = (new_ics1.vel.d_xyz.value[2]) * u.kpc/u.Myr
    vz_mean1 = np.mean(vz_coords1)
    
    # move galaxy 2 towards origin according to distance to galaxy 1
    new_pos = (orbits2[-1].pos.xyz.value - [[x_mean1.value]*N2, 
                       [y_mean1.value]*N2, 
                       [z_mean1.value]*N2]) * u.kpc
    new_vel = (orbits2[-1].vel.d_xyz.value - [[vx_mean1.value]*N2, 
                       [vy_mean1.value]*N2, 
                       [vz_mean1.value]*N2]) * u.kpc/u.Myr
    new_ics2 = gd.PhaseSpacePosition(new_pos,new_vel)
    
    # integrate orbit for galaxy 2 in potential 2
    orbits2 = gp.Hamiltonian(pot2).integrate_orbit(new_ics2, dt=timestep, 
                           n_steps=steps_per_iter) 
    new_ics2 = gd.PhaseSpacePosition(orbits2[-1].pos,orbits2[-1].vel)
    
    # move galaxy 2 back to original location
    new_pos = (orbits2[-1].pos.xyz.value + [[x_mean1.value]*N2, 
                       [y_mean1.value]*N2, 
                       [z_mean1.value]*N2]) * u.kpc
    new_vel = (orbits2[-1].vel.d_xyz.value + [[vx_mean1.value]*N2, 
                       [vy_mean1.value]*N2, 
                       [vz_mean1.value]*N2]) * u.kpc/u.Myr
    new_ics2 = gd.PhaseSpacePosition(new_pos,new_vel)
    
    # reintegrate orbit for galaxy 2 in potential 1
    orbits2 = gp.Hamiltonian(pot1).integrate_orbit(new_ics2, dt=2., 
                           n_steps=steps_per_iter) 
    new_ics2 = gd.PhaseSpacePosition(orbits2[-1].pos,orbits2[-1].vel)
    
    x_coords1 = (new_ics1.xyz.value[0]) * u.kpc
    x_mean1 = np.mean(x_coords1)
    y_coords1 = (new_ics1.xyz.value[1]) * u.kpc
    y_mean1 = np.mean(y_coords1)
    z_coords1 = (new_ics1.xyz.value[2]) * u.kpc
    z_mean1 = np.mean(z_coords1)
    
    x_coords2 = (new_ics2.xyz.value[0]) * u.kpc
    x_mean2 = np.mean(x_coords2)
    y_coords2 = (new_ics2.xyz.value[1]) * u.kpc
    y_mean2 = np.mean(y_coords2)
    z_coords2 = (new_ics2.xyz.value[2]) * u.kpc
    z_mean2 = np.mean(z_coords2)
    
    all_x.append(list(x_coords1.value) + list(x_coords2.value))
    all_y.append(list(y_coords1.value) + list(y_coords2.value))
    all_z.append(list(z_coords1.value) + list(z_coords2.value))
    
    
# ** 3D ANIMATION **
plt.ion()
def update_graph(i):
    graph.set_data(all_x[i],all_y[i])
    graph.set_3d_properties(all_z[i])
    plt.draw()
    return graph,

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
fig.gca().patch.set_facecolor('#000000') # bg black
fig.patch.set_facecolor('#000000') # bg black
ax.set_axis_off()
size = np.mean([std_dev1*100, std_dev2*100]) 
ax.set_xlim(-size,size)
ax.set_ylim(-size,size)
ax.set_zlim(-size/10,size/10)    
graph, = ax.plot(all_x[0],all_y[0],all_z[0],markerfacecolor='#FFFFD4',
                 markeredgecolor='#FFFFD4',linestyle="",marker="*",
                 alpha=1.0,markersize=0.1)
ani = anim.FuncAnimation(fig,update_graph,end,interval=1)
FFwriter = anim.FFMpegWriter(fps=30,
                             extra_args = ['-vcodec','libx264'])
                             
plt.show()
ani.save('animation2.mp4',writer=FFwriter) 
            

