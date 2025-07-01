from typing import Literal, Tuple
import numpy as np
from noise import snoise2, pnoise2
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import logging
from skimage.graph import route_through_array


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

class VegetationLayer(Layer):
    def __init__(
        self,
        elevation: EarthLayer,
        precipitation: PrecipitationLayer,
        water: WaterLayer,
        seed_density: float = 0.01,  # Initial probability of vegetation seeds
        growth_steps: int = 20,  # Number of growth simulation iterations
        max_elevation: float = 0.8,  # Maximum elevation for vegetation (relative)
        water_proximity_bonus: float = 0.3,  # Growth bonus near water
        precipitation_factor: float = 0.5,  # How much precipitation affects growth
        logging_level: int = logging.WARNING
    ):
        # Store input layers as numpy arrays
        self.elevation = elevation.layer if isinstance(elevation, EarthLayer) else elevation
        self.precipitation = precipitation.layer if isinstance(precipitation, PrecipitationLayer) else precipitation
        self.water = water.layer if isinstance(water, WaterLayer) else water
        
        # Vegetation parameters
        self.seed_density = seed_density
        self.growth_steps = growth_steps
        self.max_elevation = max_elevation
        self.water_proximity_bonus = water_proximity_bonus
        self.precipitation_factor = precipitation_factor
        
        # Call parent constructor
        super().__init__(size=elevation.size, logging_level=logging_level)

    def generate(self, size: int = 64) -> np.ndarray:
        """Generate vegetation layer through simulation."""
        self.logger.info("Starting vegetation generation")
        self.logger.debug(f"Parameters: seed_density={self.seed_density}, growth_steps={self.growth_steps}, max_elevation={self.max_elevation}")
        
        try:
            # Initialize vegetation layer
            vegetation = np.zeros_like(self.elevation)
            
            # 1. Initialize seeds
            self.logger.debug("Initializing vegetation seeds")
            vegetation = self._initialize_seeds(vegetation)
            
            # 2. Calculate environmental factors once
            self.logger.debug("Calculating environmental factors")
            growth_factors = self._calculate_growth_factors()
            
            # 3. Simulate vegetation growth over multiple steps
            self.logger.debug(f"Simulating {self.growth_steps} growth iterations")
            for step in range(self.growth_steps):
                vegetation = self._simulate_growth_step(vegetation, growth_factors, step)
                
                if step % 5 == 0:  # Log progress every 5 steps
                    current_coverage = np.sum(vegetation > 0) / vegetation.size * 100
                    self.logger.debug(f"Step {step}: vegetation coverage {current_coverage:.1f}%")
            
            # 4. Normalize and finalize
            if vegetation.max() > 0:
                vegetation = np.clip(vegetation, 0, 1)  # Ensure 0-1 range
                final_coverage = np.sum(vegetation > 0.1) / vegetation.size * 100
                avg_density = vegetation.mean()
                self.logger.info(f"Vegetation generation complete: coverage={final_coverage:.1f}%, avg_density={avg_density:.3f}")
            else:
                self.logger.warning("No vegetation generated - check parameters")
            
            return vegetation
            
        except Exception as e:
            self.logger.error(f"Failed to generate vegetation: {e}")
            raise

    def _initialize_seeds(self, vegetation: np.ndarray) -> np.ndarray:
        """Initialize vegetation seeds based on environmental conditions."""
        try:
            # Normalize elevation for threshold calculations
            elev_normalized = (self.elevation - self.elevation.min()) / (self.elevation.max() - self.elevation.min())
            
            # Create suitability mask
            # Vegetation cannot grow: in water, above max elevation
            water_mask = self.water > 0.1  # Areas with significant water
            elevation_mask = elev_normalized > self.max_elevation  # Too high
            unsuitable = water_mask | elevation_mask
            
            # Calculate base seed probability
            # Higher precipitation = more likely to have seeds
            precip_bonus = self.precipitation * 0.5
            base_probability = self.seed_density + precip_bonus
            
            # Add some randomness for natural distribution
            np.random.seed(42)  # For reproducible results
            random_factor = np.random.random(vegetation.shape) * 0.02
            
            # Final seed probability
            seed_probability = base_probability + random_factor
            seed_probability[unsuitable] = 0  # No seeds in unsuitable areas
            
            # Generate initial seeds
            initial_seeds = np.random.random(vegetation.shape) < seed_probability
            vegetation[initial_seeds] = 0.1  # Small initial vegetation value
            
            seed_count = np.sum(initial_seeds)
            unsuitable_count = np.sum(unsuitable)
            self.logger.info(f"Initialized {seed_count} vegetation seeds, {unsuitable_count} unsuitable cells")
            
            return vegetation
            
        except Exception as e:
            self.logger.error(f"Failed to initialize seeds: {e}")
            raise

    def _calculate_growth_factors(self) -> np.ndarray:
        """Calculate environmental growth factors for each cell."""
        try:
            # Normalize elevation
            elev_normalized = (self.elevation - self.elevation.min()) / (self.elevation.max() - self.elevation.min())
            
            # Elevation factor: vegetation prefers mid-elevations
            # Use a bell curve peaking around 0.3-0.4 elevation
            elevation_optimum = 0.35
            elevation_factor = np.exp(-((elev_normalized - elevation_optimum) / 0.3) ** 2)
            
            # Precipitation factor: more precipitation = better growth
            precipitation_factor = self.precipitation * self.precipitation_factor
            
            # Water proximity factor: calculate distance to nearest water
            water_proximity_factor = self._calculate_water_proximity()
            
            # Combine all factors
            growth_factors = (elevation_factor + precipitation_factor + water_proximity_factor) / 3.0
            
            # Ensure no growth in water or too high elevations
            water_mask = self.water > 0.1
            elevation_mask = elev_normalized > self.max_elevation
            unsuitable = water_mask | elevation_mask
            growth_factors[unsuitable] = 0
            
            self.logger.debug(f"Growth factors: mean={growth_factors.mean():.3f}, max={growth_factors.max():.3f}")
            return growth_factors
            
        except Exception as e:
            self.logger.error(f"Failed to calculate growth factors: {e}")
            raise

    def _calculate_water_proximity(self) -> np.ndarray:
        """Calculate water proximity bonus for each cell."""
        try:
            # Find water cells
            water_cells = self.water > 0.1
            
            if not np.any(water_cells):
                self.logger.warning("No water cells found for proximity calculation")
                return np.zeros_like(self.water)
            
            # Calculate distance to nearest water for each cell
            # This is a simplified distance calculation
            proximity = np.zeros_like(self.water)
            
            for i in range(self.water.shape[0]):
                for j in range(self.water.shape[1]):
                    if water_cells[i, j]:
                        continue  # Skip water cells themselves
                    
                    # Find minimum distance to any water cell
                    min_distance = float('inf')
                    for wi in range(max(0, i-10), min(self.water.shape[0], i+11)):
                        for wj in range(max(0, j-10), min(self.water.shape[1], j+11)):
                            if water_cells[wi, wj]:
                                distance = np.sqrt((i - wi)**2 + (j - wj)**2)
                                min_distance = min(min_distance, distance)
                    
                    # Convert distance to proximity bonus (closer = higher bonus)
                    if min_distance < float('inf'):
                        # Exponential decay with distance
                        proximity[i, j] = self.water_proximity_bonus * np.exp(-min_distance / 3.0)
            
            self.logger.debug(f"Water proximity calculated: mean bonus={proximity.mean():.4f}")
            return proximity
            
        except Exception as e:
            self.logger.error(f"Failed to calculate water proximity: {e}")
            # Return zeros if calculation fails
            return np.zeros_like(self.water)

    def _simulate_growth_step(self, vegetation: np.ndarray, growth_factors: np.ndarray, step: int) -> np.ndarray:
        """Simulate one step of vegetation growth."""
        try:
            new_vegetation = vegetation.copy()
            
            # Growth rate decreases over time (slower as vegetation matures)
            step_factor = 1.0 - (step / self.growth_steps) * 0.5
            
            for i in range(1, vegetation.shape[0] - 1):
                for j in range(1, vegetation.shape[1] - 1):
                    current_vegetation = vegetation[i, j]
                    
                    if growth_factors[i, j] > 0:
                        # Calculate growth based on environmental factors
                        base_growth = growth_factors[i, j] * 0.05 * step_factor
                        
                        # Spreading from neighboring vegetation
                        neighbor_vegetation = 0
                        neighbor_count = 0
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                if di == 0 and dj == 0:
                                    continue
                                ni, nj = i + di, j + dj
                                if vegetation[ni, nj] > 0:
                                    neighbor_vegetation += vegetation[ni, nj]
                                    neighbor_count += 1
                        
                        # Spreading factor: vegetation spreads from neighbors
                        if neighbor_count > 0:
                            spreading_factor = (neighbor_vegetation / neighbor_count) * 0.02
                        else:
                            spreading_factor = 0
                        
                        # Total growth
                        total_growth = base_growth + spreading_factor
                        
                        # Apply growth with some randomness
                        if np.random.random() < 0.8:  # 80% chance of growth
                            new_vegetation[i, j] = min(1.0, current_vegetation + total_growth)
                    
                    # Natural decay for sparse vegetation (prevents unrealistic patterns)
                    elif current_vegetation > 0 and current_vegetation < 0.05:
                        if np.random.random() < 0.1:  # 10% chance of decay
                            new_vegetation[i, j] = max(0, current_vegetation - 0.01)
            
            return new_vegetation
            
        except Exception as e:
            self.logger.error(f"Failed to simulate growth step {step}: {e}")
            return vegetation  # Return unchanged vegetation on error

    def _repr_json_(self) -> dict:
        """Enhanced JSON representation including vegetation parameters."""
        base_data = super()._repr_json_()
        base_data["seed_density"] = self.seed_density
        base_data["growth_steps"] = self.growth_steps
        base_data["max_elevation"] = self.max_elevation
        base_data["water_proximity_bonus"] = self.water_proximity_bonus
        base_data["precipitation_factor"] = self.precipitation_factor
        return base_data

    def _figure_data(self, format='png'):
        """Generate figure data for vegetation display with appropriate colormap."""
        self.logger.debug(f"Generating vegetation figure in {format} format")
        
        try:
            fig, ax = plt.subplots(figsize=(4, 4))
            
            # Use vegetation-appropriate colormap (greens)
            ax.imshow(self.layer, cmap='Greens', origin='lower')
            ax.set_title(f"Vegetation Layer ({self.size}x{self.size})")
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
            
            self.logger.debug("Successfully generated vegetation figure")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to generate vegetation figure: {e}")
            raise

class PopulationLayer(Layer):
    """
    Population layer that simulates settlement placement and growth.
    
    Features:
    - Initial seed placement based on environmental heuristics
    - Settlement growth simulation over time
    - Multiple settlement types (villages, towns, cities)
    - Labeled output for road/trade network construction
    """
    
    def __init__(self, 
                 elevation: EarthLayer, vegetation: VegetationLayer, 
                 precipitation: PrecipitationLayer, water: WaterLayer,
                 num_settlements=None, min_settlement_distance=8, 
                 growth_iterations=20, max_settlement_size=50, logging_level=logging.WARNING):
        """
        Initialize PopulationLayer.
        
        Args:
            num_settlements: Number of initial settlements (auto-calculated if None)
            min_settlement_distance: Minimum distance between settlement centers
            growth_iterations: Number of growth simulation steps
            max_settlement_size: Maximum cells per settlement
            logging_level: Logging verbosity level
        """
        
        # Layer parameters
        self.earth_layer = elevation
        size = self.earth_layer.size
        self.vegetation_layer = vegetation
        self.precipitation_layer = precipitation
        self.water_layer = water

        # Settlement parameters
        self.num_settlements = num_settlements or max(2, size // 16)
        self.min_settlement_distance = min_settlement_distance
        self.growth_iterations = growth_iterations
        self.max_settlement_size = max_settlement_size
        
        # Settlement data
        self.settlement_centers = []
        self.settlement_sizes = []
        self.population_map = None
        
        # Call parent constructor
        super().__init__(size=size, logging_level=logging_level)

        self.logger.info(f"PopulationLayer initialized: {self.num_settlements} settlements, "
                        f"min distance {min_settlement_distance}, max size {max_settlement_size}")
    
    def generate(self, size: int = 64) -> np.ndarray:
        """
        Generate population settlements based on environmental factors.
        
        Args:
            earth_layer: EarthLayer for elevation/slope data
            water_layer: WaterLayer for water proximity
            vegetation_layer: VegetationLayer for fertility
            precipitation_layer: PrecipitationLayer for climate
        
        Returns:
            numpy.ndarray: Population map with settlement IDs (0 = uninhabited, 1+ = settlement ID)
        """
        self.logger.info("Starting population generation")
        
        # Initialize population map
        self.population_map = np.zeros((self.size, self.size), dtype=int)
        
        # Calculate suitability map
        suitability = self._calculate_suitability(self.earth_layer, self.water_layer, 
                                                 self.vegetation_layer, self.precipitation_layer)
        
        # Place initial settlement seeds
        self._place_settlement_seeds(suitability)
        
        # Simulate settlement growth
        self._simulate_settlement_growth(self.earth_layer, self.water_layer, self.vegetation_layer)
        
        self.logger.info(f"Population generation complete. {len(self.settlement_centers)} settlements placed")
        
        return self.population_map
    
    def _calculate_suitability(self, earth_layer=None, water_layer=None, 
                              vegetation_layer=None, precipitation_layer=None):
        """Calculate settlement suitability based on environmental factors."""
        self.logger.debug("Calculating settlement suitability map")
        
        suitability = np.ones((self.size, self.size))
        
        # Water proximity factor
        if water_layer is not None:
            water_distance = self._calculate_water_distance(water_layer.layer)
            # Prefer areas close to water but not in water
            water_factor = np.exp(-water_distance / 5.0)
            water_factor[water_layer.layer > 0] = 0.1  # Penalize being in water
            suitability *= water_factor
            self.logger.debug("Applied water proximity factor")
        
        # Elevation and slope factor
        if earth_layer is not None:
            # Prefer moderate elevations (not too high, not too low)
            elevation_norm = earth_layer.layer / np.max(earth_layer.layer)
            elevation_factor = 1.0 - np.abs(elevation_norm - 0.4)  # Peak at 40% of max elevation
            elevation_factor = np.clip(elevation_factor, 0.1, 1.0)
            
            # Calculate slope and penalize steep areas
            slope = self._calculate_slope(earth_layer.layer)
            slope_factor = np.exp(-slope * 3.0)  # Exponential decay with slope
            
            terrain_factor = elevation_factor * slope_factor
            suitability *= terrain_factor
            self.logger.debug("Applied elevation and slope factors")
        
        # Vegetation factor (moderate vegetation is good)
        if vegetation_layer is not None:
            veg_norm = vegetation_layer.layer / np.max(vegetation_layer.layer)
            # Peak fertility around 60% vegetation coverage
            veg_factor = 1.0 - np.abs(veg_norm - 0.6)
            veg_factor = np.clip(veg_factor, 0.2, 1.0)
            suitability *= veg_factor
            self.logger.debug("Applied vegetation factor")
        
        # Precipitation factor (avoid extremes)
        if precipitation_layer is not None:
            precip_norm = precipitation_layer.layer / np.max(precipitation_layer.layer)
            # Prefer moderate precipitation
            precip_factor = 1.0 - np.abs(precip_norm - 0.5)
            precip_factor = np.clip(precip_factor, 0.3, 1.0)
            suitability *= precip_factor
            self.logger.debug("Applied precipitation factor")
        
        # Normalize suitability
        if np.max(suitability) > 0:
            suitability /= np.max(suitability)
        
        self.logger.debug(f"Suitability map calculated, range: {np.min(suitability):.3f} to {np.max(suitability):.3f}")
        return suitability
    
    def _calculate_water_distance(self, water_map):
        """Calculate distance to nearest water source."""
        try:
            from scipy.ndimage import distance_transform_edt
            # Distance transform from water cells
            return distance_transform_edt(water_map == 0)
        except ImportError:
            self.logger.warning("SciPy not available, using simple water distance calculation")
            # Simple distance calculation without scipy
            water_coords = np.where(water_map > 0)
            if len(water_coords[0]) == 0:
                return np.full((self.size, self.size), self.size)
            
            distances = np.full((self.size, self.size), self.size)
            for i in range(self.size):
                for j in range(self.size):
                    min_dist = self.size
                    for wi, wj in zip(water_coords[0], water_coords[1]):
                        dist = np.sqrt((i - wi)**2 + (j - wj)**2)
                        min_dist = min(min_dist, dist)
                    distances[i, j] = min_dist
            return distances
    
    def _calculate_slope(self, elevation_map):
        """Calculate slope magnitude from elevation data."""
        # Calculate gradients
        grad_y, grad_x = np.gradient(elevation_map)
        # Slope magnitude
        slope = np.sqrt(grad_x**2 + grad_y**2)
        return slope
    
    def _place_settlement_seeds(self, suitability):
        """Place initial settlement centers using Poisson disk sampling."""
        self.logger.debug(f"Placing {self.num_settlements} settlement seeds")
        
        # Use rejection sampling with suitability weighting
        attempts = 0
        max_attempts = self.num_settlements * 100
        settlement_id = 1
        
        while len(self.settlement_centers) < self.num_settlements and attempts < max_attempts:
            attempts += 1
            
            # Sample a random location weighted by suitability
            flat_suitability = suitability.flatten()
            if np.sum(flat_suitability) == 0:
                # Fallback to uniform sampling
                idx = np.random.randint(0, len(flat_suitability))
            else:
                probs = flat_suitability / np.sum(flat_suitability)
                idx = np.random.choice(len(flat_suitability), p=probs)
            
            y, x = np.unravel_index(idx, suitability.shape)
            
            # Check minimum distance constraint
            valid = True
            for center_y, center_x in self.settlement_centers:
                distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                if distance < self.min_settlement_distance:
                    valid = False
                    break
            
            if valid:
                self.settlement_centers.append((y, x))
                self.population_map[y, x] = settlement_id
                
                # Determine initial settlement size based on suitability
                base_size = max(1, int(suitability[y, x] * 10))
                self.settlement_sizes.append(base_size)
                
                self.logger.debug(f"Placed settlement {settlement_id} at ({y}, {x}) with base size {base_size}")
                settlement_id += 1
        
        if len(self.settlement_centers) < self.num_settlements:
            self.logger.warning(f"Only placed {len(self.settlement_centers)} of {self.num_settlements} settlements")
    
    def _simulate_settlement_growth(self, earth_layer=None, water_layer=None, vegetation_layer=None):
        """Simulate settlement growth over time."""
        self.logger.debug(f"Simulating settlement growth for {self.growth_iterations} iterations")
        
        for iteration in range(self.growth_iterations):
            new_population_map = self.population_map.copy()
            
            for settlement_id, (center_y, center_x) in enumerate(self.settlement_centers, 1):
                current_size = np.sum(self.population_map == settlement_id)
                
                # Stop growing if settlement reached max size
                if current_size >= self.max_settlement_size:
                    continue
                
                # Calculate growth probability for this settlement
                growth_rate = self._calculate_growth_rate(settlement_id, current_size, 
                                                        earth_layer, water_layer, vegetation_layer)
                
                if np.random.random() > growth_rate:
                    continue
                
                # Find cells adjacent to current settlement
                settlement_cells = np.where(self.population_map == settlement_id)
                adjacent_candidates = []
                
                for sy, sx in zip(settlement_cells[0], settlement_cells[1]):
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        ny, nx = sy + dy, sx + dx
                        if (0 <= ny < self.size and 0 <= nx < self.size and 
                            self.population_map[ny, nx] == 0):
                            
                            # Calculate expansion suitability
                            expansion_score = self._calculate_expansion_suitability(
                                ny, nx, settlement_id, earth_layer, water_layer, vegetation_layer)
                            
                            adjacent_candidates.append((ny, nx, expansion_score))
                
                # Select best expansion candidate
                if adjacent_candidates:
                    adjacent_candidates.sort(key=lambda x: x[2], reverse=True)
                    best_candidate = adjacent_candidates[0]
                    
                    if best_candidate[2] > 0.3:  # Minimum expansion threshold
                        new_population_map[best_candidate[0], best_candidate[1]] = settlement_id
            
            self.population_map = new_population_map
            
            if iteration % 5 == 0:
                total_population = np.sum(self.population_map > 0)
                self.logger.debug(f"Growth iteration {iteration}: {total_population} total population cells")
    
    def _calculate_growth_rate(self, settlement_id, current_size, earth_layer=None, 
                              water_layer=None, vegetation_layer=None):
        """Calculate growth rate for a settlement."""
        # Base growth rate decreases with size
        base_rate = max(0.1, 0.8 - (current_size / self.max_settlement_size) * 0.6)
        
        # Environmental factors
        center_y, center_x = self.settlement_centers[settlement_id - 1]
        
        # Water access bonus
        if water_layer is not None:
            water_distance = self._calculate_water_distance(water_layer.layer)[center_y, center_x]
            water_bonus = max(0, 1.0 - water_distance / 10.0)
            base_rate *= (1.0 + water_bonus * 0.3)
        
        # Vegetation bonus (moderate vegetation is good)
        if vegetation_layer is not None:
            veg_value = vegetation_layer.layer[center_y, center_x]
            veg_norm = veg_value / np.max(vegetation_layer.layer)
            veg_bonus = 1.0 - abs(veg_norm - 0.6)
            base_rate *= (1.0 + veg_bonus * 0.2)
        
        return min(0.9, base_rate)
    
    def _calculate_expansion_suitability(self, y, x, settlement_id, earth_layer=None, 
                                       water_layer=None, vegetation_layer=None):
        """Calculate suitability for settlement expansion to a specific cell."""
        suitability = 0.5  # Base suitability
        
        # Distance from settlement center
        center_y, center_x = self.settlement_centers[settlement_id - 1]
        distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        distance_penalty = min(1.0, distance / 8.0)  # Prefer closer expansion
        suitability *= (1.0 - distance_penalty * 0.4)
        
        # Terrain factors
        if earth_layer is not None:
            slope = self._calculate_slope(earth_layer.layer)[y, x]
            slope_factor = max(0.2, 1.0 - slope * 2.0)
            suitability *= slope_factor
        
        # Water proximity
        if water_layer is not None:
            if water_layer.layer[y, x] > 0:
                suitability *= 0.1  # Don't build in water
            else:
                water_distance = self._calculate_water_distance(water_layer.layer)[y, x]
                water_factor = max(0.5, 1.0 - water_distance / 8.0)
                suitability *= water_factor
        
        # Vegetation factor
        if vegetation_layer is not None:
            veg_value = vegetation_layer.layer[y, x]
            veg_norm = veg_value / np.max(vegetation_layer.layer)
            # Moderate vegetation is good, too dense is bad
            if veg_norm > 0.8:
                suitability *= 0.3  # Dense forest is hard to clear
            else:
                veg_factor = max(0.5, 1.0 - abs(veg_norm - 0.5))
                suitability *= veg_factor
        
        return max(0, suitability)
    
    def get_settlement_info(self):
        """Get information about generated settlements."""
        if self.population_map is None:
            return {}
        
        info = {}
        for settlement_id, (center_y, center_x) in enumerate(self.settlement_centers, 1):
            size = np.sum(self.population_map == settlement_id)
            settlement_type = self._classify_settlement(size)
            
            info[settlement_id] = {
                'center': (center_y, center_x),
                'size': size,
                'type': settlement_type
            }
        
        return info
    
    def _classify_settlement(self, size):
        """Classify settlement type based on size."""
        if size <= 5:
            return 'hamlet'
        elif size <= 15:
            return 'village'
        elif size <= 30:
            return 'town'
        else:
            return 'city'
        
class RoadLayer(Layer):
    """
    Road layer that creates transportation networks connecting settlements.
    
    Features:
    - Connects settlements based on priority (cities > towns > villages > hamlets)
    - Uses terrain-aware pathfinding (Dijkstra's algorithm)
    - Considers elevation gradients, water obstacles, and vegetation density
    - Generates road network with different road types
    """
    
    def __init__(self, elevation: EarthLayer, water: WaterLayer, vegetation: VegetationLayer,
                 population: PopulationLayer, bridge_cost_multiplier: float = 5.0,
                 forest_cost_multiplier: float = 3.0, slope_cost_multiplier: float = 4.0,
                 boundary_penalty: float = 10.0, boundary_width: int = 2,
                 logging_level: int = logging.WARNING):
        """
        Initialize RoadLayer.
        
        Args:
            elevation: EarthLayer for terrain slopes
            water: WaterLayer for water obstacles
            vegetation: VegetationLayer for forest obstacles
            population: PopulationLayer for settlement locations
            bridge_cost_multiplier: Cost multiplier for crossing water
            forest_cost_multiplier: Cost multiplier for going through dense vegetation
            slope_cost_multiplier: Cost multiplier for steep terrain
            boundary_penalty: Cost multiplier for map edges to discourage edge routing
            boundary_width: Width of boundary penalty zone in cells
            logging_level: Logging verbosity level
        """
        # Store input layers
        self.elevation = elevation
        self.water = water
        self.vegetation = vegetation
        self.population = population
        
        # Road construction parameters
        self.bridge_cost_multiplier = bridge_cost_multiplier
        self.forest_cost_multiplier = forest_cost_multiplier
        self.slope_cost_multiplier = slope_cost_multiplier
        self.boundary_penalty = boundary_penalty
        self.boundary_width = boundary_width
        
        # Road network data
        self.road_map = None
        self.road_network = {}  # Graph structure: settlement_id -> [(connected_settlement, road_type)]
        self.cost_map = None
        
        # Call parent constructor
        super().__init__(size=elevation.size, logging_level=logging_level)
        
        self.logger.info(f"RoadLayer initialized with cost multipliers: "
                        f"bridge={bridge_cost_multiplier}, forest={forest_cost_multiplier}, "
                        f"slope={slope_cost_multiplier}, boundary={boundary_penalty}")
    
    def generate(self, size: int = 64) -> np.ndarray:
        """
        Generate road network connecting settlements.
        
        Returns:
            numpy.ndarray: Road map with road types (0 = no road, 1 = path, 2 = road, 3 = highway)
        """
        self.logger.info("Starting road network generation")
        
        # Initialize road map
        self.road_map = np.zeros((self.size, self.size), dtype=int)
        
        # Get settlement information
        settlements = self.population.get_settlement_info()
        if len(settlements) < 2:
            self.logger.warning("Not enough settlements to create roads")
            return self.road_map
        
        # Calculate terrain cost map
        self._calculate_cost_map()
        
        # Build road network by connecting settlements
        self._build_road_network(settlements)
        
        self.logger.info(f"Road network generation complete. Connected {len(settlements)} settlements")
        return self.road_map
    
    def _calculate_cost_map(self):
        """Calculate movement cost for each cell based on terrain factors."""
        self.logger.debug("Calculating terrain cost map")
        
        # Base cost is 1.0 for all cells
        self.cost_map = np.ones((self.size, self.size), dtype=float)
        
        # Add boundary penalties to discourage edge routing
        boundary_width = self.boundary_width
        boundary_penalty = self.boundary_penalty
        
        # Apply boundary penalties
        self.cost_map[:boundary_width, :] *= boundary_penalty  # Top edge
        self.cost_map[-boundary_width:, :] *= boundary_penalty  # Bottom edge
        self.cost_map[:, :boundary_width] *= boundary_penalty  # Left edge
        self.cost_map[:, -boundary_width:] *= boundary_penalty  # Right edge
        
        # Slope cost factor
        slopes = self._calculate_slope(self.elevation.layer)
        slope_normalized = slopes / (np.max(slopes) + 1e-6)  # Normalize to 0-1
        slope_cost = 1.0 + slope_normalized * (self.slope_cost_multiplier - 1.0)
        self.cost_map *= slope_cost
        
        # Water cost factor (bridges are expensive)
        water_mask = self.water.layer > 0.1
        self.cost_map[water_mask] *= self.bridge_cost_multiplier
        
        # Vegetation cost factor (clearing forests is expensive)
        veg_normalized = self.vegetation.layer / (np.max(self.vegetation.layer) + 1e-6)
        # Dense vegetation (>0.7) is expensive to traverse
        dense_veg_mask = veg_normalized > 0.7
        forest_cost = 1.0 + veg_normalized * (self.forest_cost_multiplier - 1.0)
        forest_cost[dense_veg_mask] = self.forest_cost_multiplier
        self.cost_map *= forest_cost
        
        self.logger.debug(f"Cost map calculated: range [{np.min(self.cost_map):.2f}, {np.max(self.cost_map):.2f}]")
        self.logger.debug(f"Applied boundary penalty of {boundary_penalty}x to {boundary_width}-cell border")
    
    def _calculate_slope(self, elevation_map):
        """Calculate slope magnitude from elevation data."""
        grad_y, grad_x = np.gradient(elevation_map)
        slope = np.sqrt(grad_x**2 + grad_y**2)
        return slope
    
    def _build_road_network(self, settlements):
        """Build road network connecting settlements by priority."""
        self.logger.debug("Building road network")
        
        # Sort settlements by priority (size/type)
        settlement_priority = {
            'city': 4, 'town': 3, 'village': 2, 'hamlet': 1
        }
        
        sorted_settlements = sorted(settlements.items(), 
                                  key=lambda x: (settlement_priority.get(x[1]['type'], 0), x[1]['size']),
                                  reverse=True)
        
        connected_settlements = set()
        road_segments = []
        
        for settlement_id, settlement_info in sorted_settlements:
            center_y, center_x = settlement_info['center']
            settlement_type = settlement_info['type']
            
            if not connected_settlements:
                # First settlement becomes the network hub
                connected_settlements.add(settlement_id)
                self.logger.debug(f"Starting network with {settlement_type} {settlement_id} at ({center_y}, {center_x})")
                continue
            
            # Find closest connected settlement to connect to
            min_cost = float('inf')
            best_target = None
            best_path = None
            
            for target_id in connected_settlements:
                target_info = settlements[target_id]
                target_y, target_x = target_info['center']
                
                # Find optimal path using Dijkstra's algorithm
                path, total_cost = self._find_optimal_path((center_y, center_x), (target_y, target_x))
                
                if path and total_cost < min_cost:
                    min_cost = total_cost
                    best_target = target_id
                    best_path = path
            
            if best_path:
                # Determine road type based on settlement importance
                road_type = self._determine_road_type(settlement_info, settlements[best_target])
                
                # Add road to map
                self._add_road_to_map(best_path, road_type)
                
                # Update network graph
                if settlement_id not in self.road_network:
                    self.road_network[settlement_id] = []
                if best_target not in self.road_network:
                    self.road_network[best_target] = []
                
                self.road_network[settlement_id].append((best_target, road_type))
                self.road_network[best_target].append((settlement_id, road_type))
                
                connected_settlements.add(settlement_id)
                road_segments.append((settlement_id, best_target, road_type))
                
                self.logger.debug(f"Connected {settlement_type} {settlement_id} to {settlements[best_target]['type']} {best_target} "
                                f"with {self._road_type_name(road_type)} (cost: {min_cost:.1f})")
        
        self.logger.info(f"Built road network with {len(road_segments)} segments")
    
    def _find_optimal_path(self, start, end):
        try:
            path, cost = route_through_array(
                self.cost_map,
                start=start,
                end=end,
                fully_connected=True
            )
            return path, cost
        except Exception as e:
            self.logger.warning(f"Pathfinding failed between {start} and {end}: {e}")
            return None, float('inf')
    
    def _determine_road_type(self, settlement1, settlement2):
        """Determine road type based on connected settlement types."""
        type_priority = {'city': 4, 'town': 3, 'village': 2, 'hamlet': 1}
        
        priority1 = type_priority.get(settlement1['type'], 1)
        priority2 = type_priority.get(settlement2['type'], 1)
        max_priority = max(priority1, priority2)
        
        if max_priority >= 4:  # At least one city
            return 3  # Highway
        elif max_priority >= 3:  # At least one town
            return 2  # Road
        elif max_priority >= 2:  # At least one village
            return 2  # Road
        else:  # Only hamlets
            return 1  # Path
    
    def _road_type_name(self, road_type):
        """Get human-readable name for road type."""
        names = {1: 'path', 2: 'road', 3: 'highway'}
        return names.get(road_type, 'unknown')
    
    def _add_road_to_map(self, path, road_type):
        """Add road path to the road map."""
        for y, x in path:
            # Only upgrade road type, never downgrade
            current_type = self.road_map[y, x]
            self.road_map[y, x] = max(current_type, road_type)
    
    def get_road_network_info(self):
        """Get information about the road network."""
        if self.road_map is None:
            return {}
        
        total_cells = np.sum(self.road_map > 0)
        road_type_counts = {
            'paths': np.sum(self.road_map == 1),
            'roads': np.sum(self.road_map == 2),
            'highways': np.sum(self.road_map == 3)
        }
        
        return {
            'total_road_cells': total_cells,
            'road_type_distribution': road_type_counts,
            'network_connections': len(self.road_network),
            'network_graph': self.road_network
        }
    
    def _repr_json_(self) -> dict:
        """Enhanced JSON representation including road parameters."""
        base_data = super()._repr_json_()
        base_data["bridge_cost_multiplier"] = self.bridge_cost_multiplier
        base_data["forest_cost_multiplier"] = self.forest_cost_multiplier
        base_data["slope_cost_multiplier"] = self.slope_cost_multiplier
        base_data["boundary_penalty"] = self.boundary_penalty
        base_data["boundary_width"] = self.boundary_width
        
        # Add network statistics if available
        if self.road_map is not None:
            network_info = self.get_road_network_info()
            base_data["road_network_info"] = network_info
        
        return base_data
    
    def _figure_data(self, format='png'):
        """Generate figure data for road display with appropriate colormap."""
        self.logger.debug(f"Generating road figure in {format} format")
        
        try:
            fig, ax = plt.subplots(figsize=(4, 4))
            
            # Use discrete colormap for road types
            # 0 = no road (transparent), 1 = path (light brown), 2 = road (brown), 3 = highway (dark brown)
            road_colors = ['white', 'burlywood', 'saddlebrown', 'maroon']
            from matplotlib.colors import ListedColormap
            road_cmap = ListedColormap(road_colors)
            
            ax.imshow(self.layer, cmap=road_cmap, origin='lower', vmin=0, vmax=3)
            ax.set_title(f"Road Network ({self.size}x{self.size})")
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
            
            self.logger.debug("Successfully generated road figure")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to generate road figure: {e}")
            raise
        
    def get_cost_map(self):
        """Get the calculated cost map for visualization or analysis."""
        return self.cost_map.copy() if self.cost_map is not None else None
    
    def visualize_cost_map(self, figsize: Tuple[int, int] = (8, 8)):
        """Create a visualization of the terrain cost map."""
        if self.cost_map is None:
            raise ValueError("Cost map not calculated. Run generate() first.")
        
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Use log scale for better visualization since costs can vary widely
            cost_log = np.log1p(self.cost_map)  # log(1 + x) to handle zeros
            
            im = ax.imshow(cost_log, cmap='hot', origin='lower')
            ax.set_title("Terrain Cost Map (log scale)")
            ax.axis('off')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Movement Cost (log scale)')
            
            # Mark settlement locations if available
            settlements = self.population.get_settlement_info()
            for settlement_id, info in settlements.items():
                center_y, center_x = info['center']
                ax.scatter(center_x, center_y, marker='o', s=50, 
                          c='cyan', edgecolors='black', linewidth=2, zorder=10)
                ax.annotate(f'{settlement_id}', (center_x, center_y), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, color='white', fontweight='bold')
            
            return fig, ax
            
        except ImportError:
            raise ImportError("Matplotlib is required for cost map visualization")