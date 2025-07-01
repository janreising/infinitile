from typing import Tuple
import numpy as np
from noise import snoise2, pnoise2

class Tile():
        
    def __init__(self, coord: Tuple[int, int] = (0, 0), size: int = 64, scale: float = 0.06, octaves: int=12, 
                map_seed: Tuple[float, float] = (0, 0)):

        self.size = size
        self.coord = coord
        self.scale = scale
        self.octaves = 12
        self.map_seed = map_seed

        self.grid = self.generate_base_heightmap()

    def generate_base_heightmap(self):

        x_off = self.map_seed[0] + self.coord[0] * self.size 
        y_off = self.map_seed[0] + self.coord[1] * self.size
        
        return np.array([
            [pnoise2((x + x_off) * self.scale, (y + y_off) * self.scale, octaves=self.octaves, repeatx=999999, repeaty=999999)
            for x in range(self.size)]
            for y in range(self.size)
        ])