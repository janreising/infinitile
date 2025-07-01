from typing import Literal, Tuple
import numpy as np
from noise import snoise2, pnoise2
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import logging


class Layer():
    def __init__(self, size: int = 64, logging_level: int = logging.WARNING):
        self.size = size
        
        # Set up logging for this layer instance
        self._setup_logging(logging_level)
        
        self.logger.info(f"Initializing {self.__class__.__name__} with size={size}")
        self.layer = self.generate(size=size)
        self.logger.info(f"Layer generation completed")

    def _setup_logging(self, level: int):
        """Set up logging for this layer instance."""
        # Create a unique logger name for this instance
        logger_name = f"infinitile.{self.__class__.__name__}_{id(self)}"
        self.logger = logging.getLogger(logger_name)
        
        # Set the logging level
        self.logger.setLevel(level)
        
        # Only add handler if none exists to avoid duplicate logs
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
        # Prevent propagation to avoid duplicate messages
        self.logger.propagate = False

    def generate(self, size: int = 64):
        self.logger.error("generate() method must be implemented by subclasses")
        raise NotImplementedError("Subclasses should implement this method")

    def _repr_json_(self) -> dict:
        self.logger.debug("Generating JSON representation")
        stats = {
            "size": self.size,
            "average": np.round(np.mean(self.layer).item(), 2),
            "max": np.round(np.max(self.layer).item(), 2),
            "min": np.round(np.min(self.layer).item(), 2),
            "std": np.round(np.std(self.layer).item(), 2)
        }
        self.logger.debug(f"Layer stats: {stats}")
        return stats
    
    def _figure_data(self, format='png'):
        """Generate figure data for display in various contexts."""
        self.logger.debug(f"Generating figure data in {format} format")
        
        try:
            fig, ax = plt.subplots(figsize=(4, 4))
            
            # Use the appropriate data attribute based on the class
            data_to_plot = getattr(self, 'layer', getattr(self, 'grid', None))
            if data_to_plot is None:
                self.logger.error("No data found to plot (expected 'layer' or 'grid' attribute)")
                raise AttributeError("No data found to plot (expected 'layer' or 'grid' attribute)")
            
            ax.imshow(data_to_plot, cmap='terrain')
            
            # Set title based on available attributes
            if hasattr(self, 'coord'):
                ax.set_title(f"Tile {self.coord}")
            else:
                ax.set_title(f"Layer ({self.size}x{self.size})")
            
            ax.axis('off')
            
            # Convert figure to bytes
            buffer = BytesIO()
            fig.savefig(buffer, format=format, bbox_inches='tight', dpi=100)
            buffer.seek(0)
            
            if format.lower() == 'png':
                # Return raw bytes for PNG (for Jupyter notebooks)
                data = buffer.getvalue()
            else:
                # For other formats, encode as base64
                data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            plt.close(fig)
            buffer.close()
            
            self.logger.debug(f"Successfully generated {format} figure data")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to generate figure data: {e}")
            raise

    def _repr_png_(self):
        return self._figure_data('png')

    def save(self, path: str, cmap='terrain'):
        """Save the layer as an image file."""
        self.logger.info(f"Saving layer to {path} with colormap {cmap}")
        
        try:
            data_to_plot = getattr(self, 'layer', getattr(self, 'grid', None))
            if data_to_plot is None:
                self.logger.error("No data found to save (expected 'layer' or 'grid' attribute)")
                raise AttributeError("No data found to plot (expected 'layer' or 'grid' attribute)")
            
            plt.imsave(path, data_to_plot, cmap=cmap)
            self.logger.info(f"Successfully saved layer to {path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save layer to {path}: {e}")
            raise

class EarthLayer(Layer):
        
    def __init__(self, coord: Tuple[int, int] = (0, 0), size: int = 64, 
                 scale: float = 0.06, octaves: int=12,
                 map_seed: Tuple[float, float] = (0, 0), logging_level: int = logging.WARNING):

        self.coord = coord
        self.scale = scale
        self.octaves = octaves
        self.map_seed = map_seed
        
        # Call parent constructor, which will set up logging and call self.generate()
        super().__init__(size=size, logging_level=logging_level)

    def generate(self, size: int = 64):
        """Generate heightmap using Perlin noise."""
        self.logger.debug(f"Generating heightmap: size={size}, scale={self.scale}, octaves={self.octaves}")
        self.logger.debug(f"Coordinate offset: {self.coord}, Map seed: {self.map_seed}")
        
        try:
            x_off = self.map_seed[0] + self.coord[0] * size 
            y_off = self.map_seed[1] + self.coord[1] * size
            
            self.logger.debug(f"Calculating noise with offsets: x_off={x_off}, y_off={y_off}")
            
            heightmap = np.array([
                [pnoise2((x + x_off) * self.scale, (y + y_off) * self.scale, 
                        octaves=self.octaves, repeatx=999999, repeaty=999999)
                for x in range(size)]
                for y in range(size)
            ])
            
            self.logger.info(f"Generated heightmap: min={heightmap.min():.3f}, max={heightmap.max():.3f}, mean={heightmap.mean():.3f}")
            return heightmap
            
        except Exception as e:
            self.logger.error(f"Failed to generate heightmap: {e}")
            raise
    
    def _repr_json_(self) -> dict:
        """Enhanced JSON representation including coordinate information."""
        base_data = super()._repr_json_()
        base_data["coord"] = self.coord
        base_data["scale"] = self.scale
        base_data["octaves"] = self.octaves
        return base_data
    
    @staticmethod
    def blend_edges(heightmap, north=None, west=None, blend_width=5):
        if north is not None:
            heightmap[0, :] = north
            for i in range(1, blend_width):
                alpha = i / blend_width
                heightmap[i, :] = (1 - alpha) * north + alpha * heightmap[i, :]
        if west is not None:
            heightmap[:, 0] = west
            for i in range(1, blend_width):
                alpha = i / blend_width
                heightmap[:, i] = (1 - alpha) * west + alpha * heightmap[:, i]
        return heightmap

  
class PrecipitationLayer(Layer):
    def __init__(
        self,
        elevation: EarthLayer,
        dew_point: float = 0.3,  # Changed to 0.3 to work better as a relative value
        humidity: float = 1.0,
        wind_direction: Literal['north', 'south', 'east', 'west'] = 'north',
        logging_level: int = logging.WARNING
    ):
        self.elevation = elevation.layer if isinstance(elevation, EarthLayer) else elevation
        self.dew_point = dew_point
        self.initial_humidity = humidity
        self.wind_dir = wind_direction
        
        # Call parent constructor with the elevation size
        super().__init__(size=elevation.size, logging_level=logging_level)

    def generate(self, size: int = 64) -> np.ndarray:
        """Simulate precipitation based on wind direction, elevation, and dew point."""
        def rotate(array, k): return np.rot90(array, k=k)
        def unrotate(array, k): return np.rot90(array, k=-k)

        self.logger.info("Starting precipitation simulation")
        self.logger.debug(f"Parameters: dew_point={self.dew_point}, humidity={self.initial_humidity}, wind={self.wind_dir}")
        self.logger.debug(f"Elevation: shape={self.elevation.shape}, range=[{self.elevation.min():.3f}, {self.elevation.max():.3f}]")

        try:
            # Adjust dew point to work with actual elevation range
            elev_min, elev_max = self.elevation.min(), self.elevation.max()
            elev_range = elev_max - elev_min
            
            if self.dew_point > elev_max:
                adjusted_dew_point = elev_min + (self.dew_point * elev_range)
                self.logger.info(f"Adjusted dew_point from {self.dew_point} to {adjusted_dew_point:.3f} (relative to elevation range)")
            else:
                adjusted_dew_point = self.dew_point
                self.logger.debug(f"Using original dew_point {adjusted_dew_point:.3f}")

            wind_rotation = {'west': 0, 'north': 1, 'east': 2, 'south': 3}.get(self.wind_dir, 0)
            self.logger.debug(f"Wind rotation: {wind_rotation} for direction {self.wind_dir}")

            elev = rotate(self.elevation, wind_rotation)
            H, W = elev.shape
            
            # Initialize humidity array with the parameter value
            humidity = np.full(H, self.initial_humidity)
            precipitation_rot = np.zeros((H, W))

            # Simulation counters
            total_conditions_met = 0
            total_rainfall_events = 0
            max_rainfall_single = 0

            for x in range(W):
                # Calculate slope (elevation difference)
                if x > 0:
                    slope = elev[:, x] - elev[:, x - 1]
                else:
                    slope = np.zeros(H)
                
                rainfall = np.zeros(H)
                for y in range(H):
                    # Check precipitation conditions
                    elev_above_dew = elev[y, x] > adjusted_dew_point
                    positive_slope = slope[y] > 0
                    has_humidity = humidity[y] > 0
                    
                    if elev_above_dew and positive_slope and has_humidity:
                        total_conditions_met += 1
                        
                        # Calculate precipitation factors
                        elevation_factor = (elev[y, x] - adjusted_dew_point) / (elev_max - adjusted_dew_point + 1e-6)
                        slope_factor = np.clip(slope[y], 0, 0.5) / 0.5
                        
                        # Calculate rainfall amount
                        rainfall[y] = elevation_factor * slope_factor * humidity[y] * 0.5
                        
                        if rainfall[y] > 0:
                            total_rainfall_events += 1
                            max_rainfall_single = max(max_rainfall_single, rainfall[y])
                        
                        # Reduce humidity based on rainfall
                        humidity[y] = max(humidity[y] - rainfall[y] * 2.0, 0.0)
                    
                    # Add humidity recovery in valleys
                    elif slope[y] < -0.1 and humidity[y] < self.initial_humidity:
                        humidity[y] = min(humidity[y] + 0.05, self.initial_humidity)
                        
                precipitation_rot[:, x] = rainfall

            self.logger.info(f"Precipitation simulation: {total_conditions_met} conditions met, {total_rainfall_events} rainfall events")
            self.logger.debug(f"Max single rainfall: {max_rainfall_single:.6f}")

            # Apply rotation back
            precipitation = unrotate(precipitation_rot, wind_rotation)
            
            # Apply smoothing for realism
            try:
                from scipy.ndimage import gaussian_filter
                precipitation = gaussian_filter(precipitation, sigma=0.8)
                self.logger.debug("Applied Gaussian smoothing")
            except ImportError:
                self.logger.warning("Scipy not available, skipping smoothing")
            
            # Normalize output
            if precipitation.max() > 0:
                precipitation = precipitation / precipitation.max()
                self.logger.info(f"Normalized precipitation: final range=[{precipitation.min():.3f}, {precipitation.max():.3f}]")
            else:
                self.logger.warning("All precipitation values are zero - check parameters")
            
            return precipitation
            
        except Exception as e:
            self.logger.error(f"Failed to generate precipitation: {e}")
            raise
    
    def _repr_json_(self) -> dict:
        """Enhanced JSON representation including precipitation parameters."""
        base_data = super()._repr_json_()
        base_data["dew_point"] = self.dew_point
        base_data["initial_humidity"] = self.initial_humidity
        base_data["wind_direction"] = self.wind_dir
        return base_data
    
    def _figure_data(self, format='png'):
        """Generate figure data for precipitation display with appropriate colormap."""
        self.logger.debug(f"Generating precipitation figure in {format} format")
        
        try:
            fig, ax = plt.subplots(figsize=(4, 4))
            
            # Use precipitation-appropriate colormap
            ax.imshow(self.layer, cmap='Blues')
            ax.set_title(f"Precipitation Layer ({self.size}x{self.size})")
            ax.axis('off')
            
            # Convert figure to bytes
            buffer = BytesIO()
            fig.savefig(buffer, format=format, bbox_inches='tight', dpi=100)
            buffer.seek(0)
            
            if format.lower() == 'png':
                data = buffer.getvalue()
            else:
                data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            plt.close(fig)
            buffer.close()
            
            self.logger.debug("Successfully generated precipitation figure")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to generate precipitation figure: {e}")
            raise
    
class WaterLayer(Layer):
    def __init__(
        self,
        elevation: EarthLayer,
        precipitation: PrecipitationLayer,
        water_level: float = 0.2,  # Relative value (0-1) for lake formation threshold
        river_threshold: float = 0.3,  # Minimum precipitation to form rivers
        flow_iterations: int = 5,  # Number of iterations for water flow simulation
        logging_level: int = logging.WARNING
    ):
        self.elevation = elevation.layer if isinstance(elevation, EarthLayer) else elevation
        self.precipitation = precipitation.layer if isinstance(precipitation, PrecipitationLayer) else precipitation
        self.water_level = water_level
        self.river_threshold = river_threshold
        self.flow_iterations = flow_iterations
        
        # Call parent constructor
        super().__init__(size=elevation.size, logging_level=logging_level)

    def generate(self, size: int = 64) -> np.ndarray:
        """Generate water layer with lakes and rivers."""
        self.logger.info("Starting water layer generation")
        self.logger.debug(f"Parameters: water_level={self.water_level}, river_threshold={self.river_threshold}, flow_iterations={self.flow_iterations}")
        
        try:
            # Initialize water layer
            water = np.zeros_like(self.elevation)
            
            # 1. Create lakes based on elevation
            self.logger.debug("Creating lakes based on elevation")
            lakes = self._create_lakes()
            water += lakes
            
            # 2. Create rivers based on precipitation flow
            self.logger.debug("Creating rivers based on precipitation")
            rivers = self._create_rivers()
            water += rivers
            
            # 3. Simulate water flow and accumulation
            self.logger.debug(f"Simulating water flow over {self.flow_iterations} iterations")
            water = self._simulate_water_flow(water)
            
            # Normalize to 0-1 range
            if water.max() > 0:
                water = water / water.max()
                self.logger.info(f"Water generation complete: range=[{water.min():.3f}, {water.max():.3f}], lake_volume={lakes.sum():.3f}, river_volume={rivers.sum():.3f}")
            else:
                self.logger.warning("No water features generated - check parameters")
            
            return water
            
        except Exception as e:
            self.logger.error(f"Failed to generate water layer: {e}")
            raise
    
    def _create_lakes(self) -> np.ndarray:
        """Create lakes in low-elevation areas."""
        try:
            # Normalize elevation to 0-1 range
            elev_norm = (self.elevation - self.elevation.min()) / (self.elevation.max() - self.elevation.min())
            
            # Lakes form in areas below the water level threshold
            lakes = np.where(elev_norm < self.water_level, self.water_level - elev_norm, 0)
            lake_cells = np.count_nonzero(lakes)
            
            self.logger.debug(f"Initial lakes: {lake_cells} cells, total volume: {lakes.sum():.3f}")
            
            # Apply smoothing for natural shapes
            try:
                from scipy.ndimage import gaussian_filter
                lakes = gaussian_filter(lakes, sigma=1.0)
                self.logger.debug("Applied Gaussian smoothing to lakes")
            except ImportError:
                self.logger.warning("Scipy not available, no lake smoothing applied")
            
            final_volume = lakes.sum()
            self.logger.info(f"Created {lake_cells} lake cells with volume {final_volume:.3f}")
            return lakes
            
        except Exception as e:
            self.logger.error(f"Failed to create lakes: {e}")
            raise
    
    def _create_rivers(self) -> np.ndarray:
        """Create rivers based on precipitation flow."""
        try:
            # Only consider areas with significant precipitation
            high_precip = self.precipitation > self.river_threshold
            river_sources = np.count_nonzero(high_precip)
            
            self.logger.debug(f"Found {river_sources} high precipitation sources above threshold {self.river_threshold}")
            
            # Initialize river network
            rivers = np.zeros_like(self.elevation)
            
            if not high_precip.any():
                self.logger.warning("No precipitation areas above river threshold - no rivers generated")
                return rivers
            
            # Find high precipitation starting points
            precip_sources = np.where(high_precip, self.precipitation, 0)
            rivers_traced = 0
            
            # Simulate water flow from high precipitation areas
            for i in range(self.elevation.shape[0]):
                for j in range(self.elevation.shape[1]):
                    if precip_sources[i, j] > self.river_threshold:
                        # Trace water flow downhill
                        flow_strength = precip_sources[i, j]
                        self._trace_river_flow(rivers, i, j, flow_strength)
                        rivers_traced += 1
            
            river_volume = rivers.sum()
            self.logger.info(f"Traced {rivers_traced} rivers with total volume {river_volume:.3f}")
            return rivers
            
        except Exception as e:
            self.logger.error(f"Failed to create rivers: {e}")
            raise
    
    def _trace_river_flow(self, rivers: np.ndarray, start_i: int, start_j: int, flow_strength: float) -> None:
        """Trace water flow downhill to create river paths."""
        current_i, current_j = start_i, start_j
        visited = set()
        
        for _ in range(50):  # Maximum flow distance
            if (current_i, current_j) in visited:
                break
            visited.add((current_i, current_j))
            
            # Add water to current position
            rivers[current_i, current_j] += flow_strength * 0.1
            
            # Find steepest downhill direction
            best_drop = 0
            next_i, next_j = current_i, current_j
            
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    
                    ni, nj = current_i + di, current_j + dj
                    if 0 <= ni < self.elevation.shape[0] and 0 <= nj < self.elevation.shape[1]:
                        drop = self.elevation[current_i, current_j] - self.elevation[ni, nj]
                        if drop > best_drop:
                            best_drop = drop
                            next_i, next_j = ni, nj
            
            # If no downhill direction found, stop
            if best_drop <= 0:
                break
            
            # Reduce flow strength as water flows
            flow_strength *= 0.95
            current_i, current_j = next_i, next_j
    
    def _simulate_water_flow(self, initial_water: np.ndarray) -> np.ndarray:
        """Simulate water flow and accumulation over multiple iterations."""
        water = initial_water.copy()
        
        for iteration in range(self.flow_iterations):
            new_water = water.copy()
            
            for i in range(1, water.shape[0] - 1):
                for j in range(1, water.shape[1] - 1):
                    if water[i, j] > 0:
                        # Calculate water surface elevation (terrain + water)
                        current_surface = self.elevation[i, j] + water[i, j]
                        
                        # Check all 8 neighbors
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                if di == 0 and dj == 0:
                                    continue
                                
                                ni, nj = i + di, j + dj
                                neighbor_surface = self.elevation[ni, nj] + water[ni, nj]
                                
                                # If current cell is higher, flow some water downhill
                                if current_surface > neighbor_surface:
                                    flow_amount = min(water[i, j] * 0.1, 
                                                    (current_surface - neighbor_surface) * 0.5)
                                    
                                    new_water[i, j] -= flow_amount
                                    new_water[ni, nj] += flow_amount
            
            water = new_water
            
            # Apply some evaporation/settling
            water *= 0.98
        
        return water
    
    def _repr_json_(self) -> dict:
        """Enhanced JSON representation including water parameters."""
        base_data = super()._repr_json_()
        base_data["water_level"] = self.water_level
        base_data["river_threshold"] = self.river_threshold
        base_data["flow_iterations"] = self.flow_iterations
        return base_data
    
    def _figure_data(self, format='png'):
        """Generate figure data for water display with appropriate colormap."""
        self.logger.debug(f"Generating water figure in {format} format")
        
        try:
            fig, ax = plt.subplots(figsize=(4, 4))
            
            # Use water-appropriate colormap
            ax.imshow(self.layer, cmap='Blues_r')  # Reverse blues for better water visualization
            ax.set_title(f"Water Layer ({self.size}x{self.size})")
            ax.axis('off')
            
            # Convert figure to bytes
            buffer = BytesIO()
            fig.savefig(buffer, format=format, bbox_inches='tight', dpi=100)
            buffer.seek(0)
            
            if format.lower() == 'png':
                data = buffer.getvalue()
            else:
                data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            plt.close(fig)
            buffer.close()
            
            self.logger.debug("Successfully generated water figure")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to generate water figure: {e}")
            raise