""" Lattice class implementation """

import numpy as np


class Lattice:
    
    def __init__(self, nx, ny, tau, scen):
        self.nx = nx
        self.ny = ny
        self.tau = tau
        self.scen = scen
        
        self.bound = np.full((self.ny, self.nx),False)
        self.flow_direct = 0
        self.flow_speed = 0
        
        
        if self.scen == "2DQ9":
            self.Q = 9
            self.idxs = np.arange(9)
            self.F = np.ones((self.ny, self.nx, self.Q)) # Distribution function array
            self.cxs = np.array([0, 0, 1, 1, 1, 0, -1,-1,-1])
            self.cys = np.array([0, 1, 1, 0,-1,-1, -1, 0, 1])
            self.weight = np.array([4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36])
            self.opp_nodes = [0, 5, 6, 7, 8, 1, 2, 3, 4]
            
            self.ux = np.zeros((self.ny, self.nx)) # Velocity in x direction
            self.uy = np.zeros((self.ny, self.nx)) # Velocity in y direction
                


    def initialize(self, flow_direct, flow_speed):
        """Initialize the arrays with the initial values."""      
        
        # Set density func initial conditions
        self.F += 0.01*np.random.randn(self.ny, self.nx, self.Q) # add randomness to init condit
        self.F[:,:,self.flow_direct] = self.flow_speed  # initial flow direction & speed
        
        

    def add_bd_circle(self, cx, cy, r):
        """Create a circular boundary centered at (cx, cy) with radius r."""
        
        for y in range(0, self.ny):
            for x in range(0, self.nx):
                if np.sqrt((x-cx)**2 + (y-cy)**2) < r:
                    self.bound[y][x] = True
                    

    def add_bd_rectangle(self, xcoords, ycoords):
        """Create a rectangle boundary from coordinates"""
        
        self.bound[ycoords[0]:ycoords[1], xcoords[0]:xcoords[1]] = True