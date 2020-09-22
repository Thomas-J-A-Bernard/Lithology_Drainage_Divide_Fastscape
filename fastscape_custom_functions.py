"""
Created on Wed Jan  8 16:47:50 2020
@author: Thomas Bernard
"""

import xsimlab
import numpy as np
from typing import List, Tuple

from fastscape.processes import (BlockUplift, LinearDiffusion, StreamPowerChannel, UniformRectilinearGrid2D, RasterGrid2D, Escarpment)

@xsimlab.process
class SquareBasementIntrusion:
    """Mimic the effect on erosion of an intrusion with a different erodibility
    compare to the surrounding rock. Position of the intrusion scale with the
    grid__shape (not the grid__lenght) !"""
    
    x_origin_position = xsimlab.variable(description='x original position of the basement intrusion')
    y_origin_position = xsimlab.variable(description='y original position of the basement intrusion')
    x_lenght = xsimlab.variable(description='x lenght of the basement intrusion')
    y_lenght = xsimlab.variable(description='y lenght of the basement intrudion')
        
    basement_k_coef = xsimlab.variable(description='erodability coefficient of the granitic basement')
    rock_k_coef = xsimlab.variable(description='erodibility coefficient of the surrounding rocks')
    basement_diff = xsimlab.variable(description='diffusivity of the granitic basement')
    rock_diff = xsimlab.variable(description='diffusivity of the surrounding rocks')
    
    grid_shape = xsimlab.foreign(RasterGrid2D, 'shape')
    
    k_coef = xsimlab.foreign(StreamPowerChannel, 'k_coef', intent='out')
    diffusivity = xsimlab.foreign(LinearDiffusion, 'diffusivity', intent='out')
    
    def run_step(self):
        
        mask = np.zeros((self.grid_shape[0], self.grid_shape[1]), dtype=bool)        
        mask[self.y_origin_position:self.y_origin_position + self.y_lenght, self.x_origin_position:self.x_origin_position + self.x_lenght] = True
        
        self.k_coef = np.where(mask, self.basement_k_coef, self.rock_k_coef)
        self.diffusivity = np.where(mask, self.basement_diff, self.rock_diff)

@xsimlab.process
class CircleBasementIntrusion:
    
    x_origin_position = xsimlab.variable(description='x original position of the basement intrusion')
    y_origin_position = xsimlab.variable(description='y original position of the basement intrusion')
    radius = xsimlab.variable(description='radius of the basement intrusion')

    basement_k_coef = xsimlab.variable(description='erodability coefficient of the granitic basement')
    rock_k_coef = xsimlab.variable(description='erodibility coefficient of the surrounding rocks')
    basement_diff = xsimlab.variable(description='diffusivity of the granitic basement')
    rock_diff = xsimlab.variable(description='diffusivity of the surrounding rocks')
    
    grid_shape = xsimlab.foreign(RasterGrid2D, 'shape')
    
    k_coef = xsimlab.foreign(StreamPowerChannel, 'k_coef', intent='out')
    diffusivity = xsimlab.foreign(LinearDiffusion, 'diffusivity', intent='out')
    
    def run_step(self):
        
        def grid_circle(x, y, sigma):
            """ create a small collection of points in a neighborhood of some point 
            """
            neighborhood = []
    
            X = int(sigma)
            for i in range(-X, X + 1):
                Y = int(pow(sigma * sigma - i * i, 1/2))
                for j in range(-Y, Y + 1):
                    neighborhood.append((y + i, x + j))
    
            return neighborhood
        
        points = grid_circle(self.x_origin_position, self.y_origin_position, self.radius)
        
        mask = np.zeros((self.grid_shape[0], self.grid_shape[1]), dtype=bool)
        for i in range(0, np.shape(points)[0]):
            mask[points[i]] = True
        
        self.k_coef = np.where(mask, self.basement_k_coef, self.rock_k_coef)
        self.diffusivity = np.where(mask, self.basement_diff, self.rock_diff)

@xsimlab.process
class VariableUplift:
    """Compute a linear gradient uplift rate along the x or y axis"""
    
    uplift_rate = xsimlab.variable(description='uplift rate apply to the grid')
    coeff = xsimlab.variable(description='coefficient for the linear uplift gradient')
    axis = xsimlab.variable(description='axis (x or y) where the gradient is applied')
    
    grid_shape = xsimlab.foreign(RasterGrid2D, 'shape')
    
    rate = xsimlab.foreign(BlockUplift, 'rate', intent='out')
    
    def run_step(self):
        
        mask = np.ones((self.grid_shape[0], self.grid_shape[1])) * self.uplift_rate *self.coeff
        
        if self.axis ==0:
            self.rate = mask
        
        if self.axis == 1:
            gradient = np.linspace(0, 1, self.grid_shape[0])
            self.rate = mask * gradient[:,None]
        
        if self.axis == 2:
            gradient = np.linspace(0, 1, self.grid_shape[0])[::-1]
            self.rate = mask * gradient[:,None]
        
        if self.axis == 3:
            gradient = np.linspace(0, 1, self.grid_shape[1])
            self.rate = mask * gradient
            
        if self.axis == 4:
            gradient = np.linspace(0, 1, self.grid_shape[1])[::-1]
            self.rate = mask * gradient
        
@xsimlab.process
class VariableTwoBlockUplift:
    """compute diferent linear gradient uplift rate along the x or y axis 
    for two blocks separated by a clip plane"""  
    
    x_position = xsimlab.variable(description='position of the clip plane along the x-axis')
        
    uplift_rate_left = xsimlab.variable(description='uplift rate of the left box')
    uplift_rate_right = xsimlab.variable(description='uplift rate of the right box')
    gradient_left = xsimlab.variable(description='gradient of the left box uplift')
    gradient_right = xsimlab.variable(description='gradient of the right box uplift')
    
    grid_shape = xsimlab.foreign(RasterGrid2D, 'shape')
    
    rate = xsimlab.foreign(BlockUplift, 'rate', intent='out')

    def run_step(self):
        
        mask = np.ones((self.grid_shape[0], self.grid_shape[1]))
        mask[:, 0:self.x_position] = self.uplift_rate_left
        mask[:, self.x_position:self.grid_shape[1]] = self.uplift_rate_right
        
        gradient = np.ones((self.grid_shape[1]))
        gradient[0:self.x_position] = np.linspace(self.gradient_left, 1, self.x_position)
        gradient[self.x_position:self.grid_shape[1]] = np.linspace(self.gradient_right, 1, np.abs(self.grid_shape[1]-self.x_position))
        
        self.rate = mask * gradient
        
@xsimlab.process
class EscarpmentWithPertubation(Escarpment):
    
    y = xsimlab.foreign(UniformRectilinearGrid2D, 'y')
    grid_lenght = xsimlab.foreign(UniformRectilinearGrid2D, 'length')
    
    def initialize(self):
        super(EscarpmentWithPertubation, self).initialize()
        
        perturb = np.cos(self.x/self.grid_lenght[1]*2.*np.pi)
        
        self.elevation += perturb       