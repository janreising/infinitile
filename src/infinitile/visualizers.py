"""
Visualization module for InfiniTile terrain layers.

This module provides various rendering approaches for visualizing terrain data:
- Simple 2D visualizations (individual layers, composites)
- 2.5D rendering (heightmaps with shading, isometric views)
- 3D rendering (mesh visualization, interactive plots)
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LightSource
import matplotlib.patches as patches
from matplotlib.collections import PolyCollection
import logging

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    MPL_3D_AVAILABLE = True
except ImportError:
    MPL_3D_AVAILABLE = False

from .generators import Layer, EarthLayer, PrecipitationLayer, WaterLayer, VegetationLayer, PopulationLayer, RoadLayer


class TerrainVisualizer:
    """Main visualization class for terrain layers."""
    
    def __init__(self, logging_level: int = logging.WARNING):
        self.logger = logging.getLogger(f"infinitile.TerrainVisualizer_{id(self)}")
        self.logger.setLevel(logging_level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.logger.propagate = False
        self.logger.info("TerrainVisualizer initialized")
    
    # ========================================
    # 2D VISUALIZATION METHODS
    # ========================================
    
    def plot_single_layer(self, layer: Layer, title: Optional[str] = None, 
                         cmap: Optional[str] = None, figsize: Tuple[int, int] = (8, 8)):
        """Plot a single layer with appropriate colormap."""
        self.logger.debug(f"Plotting single layer: {type(layer).__name__}")
        
        # Auto-select colormap based on layer type
        if cmap is None:
            cmap = self._get_default_colormap(layer)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(layer.layer, cmap=cmap, origin='lower')
        ax.set_title(title or f"{type(layer).__name__}")
        ax.axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        self.logger.info(f"Single layer plot created for {type(layer).__name__}")
        return fig, ax
    
    def plot_layer_grid(self, layers: List[Layer], titles: Optional[List[str]] = None,
                       cmaps: Optional[List[str]] = None, figsize: Tuple[int, int] = (15, 10)):
        """Plot multiple layers in a grid layout."""
        self.logger.debug(f"Creating grid plot for {len(layers)} layers")
        
        n_layers = len(layers)
        cols = min(3, n_layers)
        rows = (n_layers + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # Ensure axes is always a list we can iterate over
        if n_layers == 1:
            axes = [axes]
        elif rows == 1 and cols > 1:
            # When we have one row with multiple columns, axes is already an array
            axes = list(axes) if hasattr(axes, '__len__') else [axes]
        else:
            # Multiple rows - flatten the 2D array
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        for i, layer in enumerate(layers):
            if i >= len(axes):
                break
                
            cmap = (cmaps[i] if cmaps and i < len(cmaps) 
                   else self._get_default_colormap(layer))
            title = (titles[i] if titles and i < len(titles) 
                    else type(layer).__name__)
            
            im = axes[i].imshow(layer.layer, cmap=cmap, origin='lower')
            axes[i].set_title(title)
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i])
        
        # Hide unused subplots
        for i in range(n_layers, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        self.logger.info(f"Grid plot created for {n_layers} layers")
        return fig, axes
    
    def create_composite_map(self, elevation: EarthLayer, water: Optional[WaterLayer] = None,
                           precipitation: Optional[PrecipitationLayer] = None,
                           vegetation: Optional[VegetationLayer] = None,
                           population: Optional[PopulationLayer] = None,
                           roads: Optional[RoadLayer] = None,
                           alpha_water: float = 0.7, alpha_precip: float = 0.5,
                           alpha_vegetation: float = 0.6, alpha_population: float = 0.8,
                           alpha_roads: float = 0.9, figsize: Tuple[int, int] = (10, 10)):
        """Create a composite map combining multiple layers."""
        self.logger.debug("Creating composite map")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Base elevation layer
        im_elev = ax.imshow(elevation.layer, cmap='terrain', origin='lower')
        
        # Overlay vegetation layer (before water/population so they appear on top)
        if vegetation is not None:
            veg_mask = vegetation.layer > 0.1  # Only show significant vegetation
            im_veg = ax.imshow(np.ma.masked_where(~veg_mask, vegetation.layer),
                             cmap='Greens', alpha=alpha_vegetation, origin='lower')
        
        # Overlay water layer
        if water is not None:
            water_mask = water.layer > 0.1  # Only show significant water
            im_water = ax.imshow(np.ma.masked_where(~water_mask, water.layer),
                               cmap='Blues', alpha=alpha_water, origin='lower')
        
        # Overlay precipitation as contours or transparency
        if precipitation is not None:
            precip_mask = precipitation.layer > 0.2
            im_precip = ax.imshow(np.ma.masked_where(~precip_mask, precipitation.layer),
                                cmap='Blues', alpha=alpha_precip, origin='lower')
        
        # Overlay population settlements (on top of everything)
        if population is not None:
            pop_mask = population.layer > 0  # Show all settlements
            im_pop = ax.imshow(np.ma.masked_where(~pop_mask, population.layer),
                             cmap='tab10', alpha=alpha_population, origin='lower')
            
            # Add settlement markers for better visibility
            settlement_info = population.get_settlement_info()
            for settlement_id, info in settlement_info.items():
                center_y, center_x = info['center']
                settlement_type = info['type']
                
                # Different markers for different settlement types
                if settlement_type == 'hamlet':
                    marker, size = 'o', 20
                elif settlement_type == 'village':
                    marker, size = 's', 30
                elif settlement_type == 'town':
                    marker, size = '^', 40
                else:  # city
                    marker, size = '*', 50
                
                ax.scatter(center_x, center_y, marker=marker, s=size, 
                          c='red', edgecolors='black', linewidth=1, zorder=10)
        
        # Overlay road network (on top of everything)
        if roads is not None:
            road_mask = roads.layer > 0  # Show all roads
            # Create custom colormap for roads
            road_colors = ['white', 'burlywood', 'saddlebrown', 'maroon']
            from matplotlib.colors import ListedColormap
            road_cmap = ListedColormap(road_colors)
            
            im_roads = ax.imshow(np.ma.masked_where(~road_mask, roads.layer),
                               cmap=road_cmap, alpha=alpha_roads, origin='lower',
                               vmin=0, vmax=3, zorder=5)
        
        ax.set_title("Composite Terrain Map")
        ax.axis('off')
        
        # Create custom legend
        legend_elements = [patches.Patch(color='brown', label='Terrain')]
        if vegetation is not None:
            legend_elements.append(patches.Patch(color='green', label='Vegetation'))
        if water is not None:
            legend_elements.append(patches.Patch(color='blue', label='Water'))
        if precipitation is not None:
            legend_elements.append(patches.Patch(color='lightblue', label='Precipitation'))
        if population is not None:
            legend_elements.append(patches.Patch(color='red', label='Settlements'))
        if roads is not None:
            legend_elements.append(patches.Patch(color='saddlebrown', label='Roads'))
        
        ax.legend(handles=legend_elements, loc='upper right')
        
        self.logger.info("Composite map created")
        return fig, ax
    
    # ========================================
    # 2.5D VISUALIZATION METHODS
    # ========================================
    
    def plot_heightmap_shaded(self, elevation: EarthLayer, azimuth: float = 315,
                             altitude: float = 45, figsize: Tuple[int, int] = (10, 10)):
        """Create a shaded relief visualization of elevation data."""
        self.logger.debug("Creating shaded heightmap")
        
        if not MPL_3D_AVAILABLE:
            self.logger.warning("3D plotting not available, falling back to 2D")
            return self.plot_single_layer(elevation, "Elevation (2D fallback)")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create light source for shading
        ls = LightSource(azdeg=azimuth, altdeg=altitude)
        
        # Generate shaded relief
        shaded = ls.shade(elevation.layer, cmap=plt.cm.terrain, vert_exag=1.5)
        
        ax.imshow(shaded, origin='lower')
        ax.set_title("Shaded Relief Map")
        ax.axis('off')
        
        self.logger.info("Shaded heightmap created")
        return fig, ax
    
    def plot_isometric_view(self, elevation: EarthLayer, water: Optional[WaterLayer] = None,
                           scale_factor: float = 0.3, figsize: Tuple[int, int] = (12, 8)):
        """Create an isometric-style visualization."""
        self.logger.debug("Creating isometric view")
        
        height, width = elevation.layer.shape
        
        # Create isometric transformation
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sample data for performance (every nth point)
        step = max(1, min(height, width) // 50)
        y_indices, x_indices = np.meshgrid(
            np.arange(0, height, step),
            np.arange(0, width, step),
            indexing='ij'
        )
        
        # Apply isometric transformation
        iso_x = (x_indices - y_indices) * np.cos(np.pi/6)
        iso_y = (x_indices + y_indices) * np.sin(np.pi/6) + elevation.layer[y_indices, x_indices] * scale_factor
        
        # Create color mapping based on elevation
        colors = plt.cm.terrain(elevation.layer[y_indices, x_indices])
        
        # Plot as scatter with varying colors
        ax.scatter(iso_x.flatten(), iso_y.flatten(), c=colors.reshape(-1, 4), s=1, alpha=0.8)
        
        # Add water if available
        if water is not None:
            water_mask = water.layer > 0.1
            if np.any(water_mask[y_indices, x_indices]):
                water_y = iso_y + water.layer[y_indices, x_indices] * scale_factor
                water_points = water_mask[y_indices, x_indices]
                ax.scatter(iso_x[water_points], water_y[water_points], 
                          c='blue', s=2, alpha=0.7, label='Water')
        
        ax.set_title("Isometric Terrain View")
        ax.set_aspect('equal')
        ax.axis('off')
        
        self.logger.info("Isometric view created")
        return fig, ax
    
    def plot_contour_map(self, elevation: EarthLayer, levels: int = 20,
                        water: Optional[WaterLayer] = None,
                        figsize: Tuple[int, int] = (10, 10)):
        """Create a topographic contour map."""
        self.logger.debug("Creating contour map")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create contour lines
        contours = ax.contour(elevation.layer, levels=levels, colors='brown', alpha=0.7, linewidths=0.5)
        contour_filled = ax.contourf(elevation.layer, levels=levels, cmap='terrain', alpha=0.6)
        
        # Add elevation labels
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
        
        # Overlay water
        if water is not None:
            water_contours = ax.contourf(water.layer, levels=10, cmap='Blues', alpha=0.8)
        
        ax.set_title("Topographic Map")
        plt.colorbar(contour_filled, ax=ax, label='Elevation')
        
        self.logger.info("Contour map created")
        return fig, ax
    
    # ========================================
    # 3D VISUALIZATION METHODS
    # ========================================
    
    def plot_3d_surface(self, elevation: EarthLayer, water: Optional[WaterLayer] = None,
                       figsize: Tuple[int, int] = (12, 10), elevation_scale: float = 10):
        """Create a 3D surface plot."""
        self.logger.debug("Creating 3D surface plot")
        
        if not MPL_3D_AVAILABLE:
            self.logger.error("3D plotting not available")
            raise ImportError("matplotlib 3D plotting not available")
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create coordinate grids
        height, width = elevation.layer.shape
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)
        Z = elevation.layer * elevation_scale
        
        # Plot elevation surface
        surf = ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.8, 
                              linewidth=0, antialiased=True)
        
        # Add water surface if available
        if water is not None:
            water_mask = water.layer > 0.1
            if np.any(water_mask):
                Z_water = Z + water.layer * elevation_scale * 0.5
                Z_water = np.ma.masked_where(~water_mask, Z_water)
                water_surf = ax.plot_surface(X, Y, Z_water, cmap='Blues', 
                                           alpha=0.6, linewidth=0)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Elevation')
        ax.set_title('3D Terrain Visualization')
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        self.logger.info("3D surface plot created")
        return fig, ax
    
    def create_interactive_plotly(self, elevation: EarthLayer, 
                                 water: Optional[WaterLayer] = None,
                                 precipitation: Optional[PrecipitationLayer] = None):
        """Create an interactive 3D visualization using Plotly."""
        if not PLOTLY_AVAILABLE:
            self.logger.error("Plotly not available for interactive visualization")
            raise ImportError("Plotly not available. Install with: pip install plotly")
        
        self.logger.debug("Creating interactive Plotly visualization")
        
        height, width = elevation.layer.shape
        x = np.arange(width)
        y = np.arange(height)
        
        # Create base elevation surface
        fig = go.Figure()
        
        fig.add_trace(go.Surface(
            z=elevation.layer,
            x=x,
            y=y,
            colorscale='Earth',
            name='Elevation',
            showscale=True
        ))
        
        # Add water layer if available
        if water is not None:
            water_z = elevation.layer + water.layer * 0.1
            fig.add_trace(go.Surface(
                z=water_z,
                x=x,
                y=y,
                colorscale='Blues',
                opacity=0.7,
                name='Water',
                showscale=False
            ))
        
        fig.update_layout(
            title='Interactive 3D Terrain',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Elevation',
                camera=dict(eye=dict(x=1.2, y=1.2, z=0.6))
            ),
            width=800,
            height=600
        )
        
        self.logger.info("Interactive Plotly visualization created")
        return fig
    
    def create_multi_view_dashboard(self, elevation: EarthLayer, 
                                   water: Optional[WaterLayer] = None,
                                   precipitation: Optional[PrecipitationLayer] = None):
        """Create a comprehensive dashboard with multiple visualization types."""
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not available, creating matplotlib dashboard")
            return self._create_matplotlib_dashboard(elevation, water, precipitation)
        
        self.logger.debug("Creating multi-view dashboard")
        
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "surface"}, {"type": "heatmap"}],
                   [{"type": "contour"}, {"type": "heatmap"}]],
            subplot_titles=['3D Surface', 'Elevation Heatmap', 
                           'Contour Map', 'Water Distribution']
        )
        
        # 3D Surface
        fig.add_trace(go.Surface(z=elevation.layer, colorscale='Earth'), row=1, col=1)
        
        # Elevation heatmap
        fig.add_trace(go.Heatmap(z=elevation.layer, colorscale='Earth'), row=1, col=2)
        
        # Contour map
        fig.add_trace(go.Contour(z=elevation.layer, colorscale='Earth'), row=2, col=1)
        
        # Water distribution
        if water is not None:
            fig.add_trace(go.Heatmap(z=water.layer, colorscale='Blues'), row=2, col=2)
        else:
            fig.add_trace(go.Heatmap(z=np.zeros_like(elevation.layer)), row=2, col=2)
        
        fig.update_layout(
            title='Terrain Analysis Dashboard',
            height=800,
            width=1200
        )
        
        self.logger.info("Multi-view dashboard created")
        return fig
    
    # ========================================
    # UTILITY METHODS
    # ========================================
    
    def _get_default_colormap(self, layer: Layer) -> str:
        """Get appropriate default colormap for layer type."""
        if isinstance(layer, EarthLayer):
            return 'terrain'
        elif isinstance(layer, WaterLayer):
            return 'Blues'
        elif isinstance(layer, PrecipitationLayer):
            return 'Blues'
        elif isinstance(layer, VegetationLayer):
            return 'Greens'
        elif isinstance(layer, PopulationLayer):
            return 'tab10'  # Discrete colormap for settlement IDs
        elif isinstance(layer, RoadLayer):
            return 'copper'  # Brown tones for roads
        else:
            return 'viridis'
    
    def _create_matplotlib_dashboard(self, elevation: EarthLayer,
                                   water: Optional[WaterLayer] = None,
                                   precipitation: Optional[PrecipitationLayer] = None):
        """Create matplotlib-based dashboard when Plotly is not available."""
        self.logger.debug("Creating matplotlib dashboard")
        
        fig = plt.figure(figsize=(16, 12))
        
        # 2D Elevation
        ax1 = plt.subplot(2, 3, 1)
        im1 = ax1.imshow(elevation.layer, cmap='terrain', origin='lower')
        ax1.set_title('Elevation')
        plt.colorbar(im1, ax=ax1)
        
        # Shaded relief
        ax2 = plt.subplot(2, 3, 2)
        if MPL_3D_AVAILABLE:
            ls = LightSource(azdeg=315, altdeg=45)
            shaded = ls.shade(elevation.layer, cmap=plt.cm.terrain)
            ax2.imshow(shaded, origin='lower')
        else:
            ax2.imshow(elevation.layer, cmap='terrain', origin='lower')
        ax2.set_title('Shaded Relief')
        
        # Contour map
        ax3 = plt.subplot(2, 3, 3)
        contours = ax3.contour(elevation.layer, levels=15, colors='brown', alpha=0.7)
        ax3.contourf(elevation.layer, levels=15, cmap='terrain', alpha=0.6)
        ax3.set_title('Contour Map')
        
        # Water layer
        ax4 = plt.subplot(2, 3, 4)
        if water is not None:
            im4 = ax4.imshow(water.layer, cmap='Blues', origin='lower')
            plt.colorbar(im4, ax=ax4)
        ax4.set_title('Water Distribution')
        
        # Precipitation
        ax5 = plt.subplot(2, 3, 5)
        if precipitation is not None:
            im5 = ax5.imshow(precipitation.layer, cmap='Blues', origin='lower')
            plt.colorbar(im5, ax=ax5)
        ax5.set_title('Precipitation')
        
        # Composite
        ax6 = plt.subplot(2, 3, 6)
        ax6.imshow(elevation.layer, cmap='terrain', origin='lower')
        if water is not None:
            water_mask = water.layer > 0.1
            ax6.imshow(np.ma.masked_where(~water_mask, water.layer),
                      cmap='Blues', alpha=0.7, origin='lower')
        ax6.set_title('Composite View')
        
        plt.tight_layout()
        self.logger.info("Matplotlib dashboard created")
        return fig
    
    def save_visualization(self, fig, filename: str, dpi: int = 300, format: str = 'png'):
        """Save visualization to file."""
        self.logger.info(f"Saving visualization to {filename}")
        
        try:
            if hasattr(fig, 'write_image'):  # Plotly figure
                fig.write_image(filename, width=1200, height=800, format=format)
            else:  # Matplotlib figure
                fig.savefig(filename, dpi=dpi, bbox_inches='tight', format=format)
            
            self.logger.info(f"Visualization saved successfully to {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save visualization: {e}")
            raise


# ========================================
# CONVENIENCE FUNCTIONS
# ========================================

def quick_plot(layer: Layer, **kwargs):
    """Quick plotting function for single layers."""
    viz = TerrainVisualizer()
    return viz.plot_single_layer(layer, **kwargs)

def quick_composite(elevation: EarthLayer, water: Optional[WaterLayer] = None,
                   precipitation: Optional[PrecipitationLayer] = None,
                   vegetation: Optional[VegetationLayer] = None,
                   population: Optional[PopulationLayer] = None,
                   roads: Optional[RoadLayer] = None, **kwargs):
    """Quick composite map creation."""
    viz = TerrainVisualizer()
    return viz.create_composite_map(elevation, water, precipitation, vegetation, population, roads, **kwargs)

def quick_3d(elevation: EarthLayer, water: Optional[WaterLayer] = None, **kwargs):
    """Quick 3D visualization."""
    viz = TerrainVisualizer()
    try:
        return viz.plot_3d_surface(elevation, water, **kwargs)
    except ImportError:
        print("3D plotting not available, falling back to 2D")
        return viz.plot_single_layer(elevation)

def interactive_view(elevation: EarthLayer, water: Optional[WaterLayer] = None,
                    precipitation: Optional[PrecipitationLayer] = None):
    """Create interactive visualization if Plotly is available."""
    viz = TerrainVisualizer()
    try:
        return viz.create_interactive_plotly(elevation, water, precipitation)
    except ImportError:
        print("Plotly not available for interactive visualization")
        return viz._create_matplotlib_dashboard(elevation, water, precipitation)

def plot_population_with_info(population: PopulationLayer, figsize: Tuple[int, int] = (10, 8)):
    """
    Create a specialized plot for PopulationLayer with settlement information.
    
    Args:
        population: PopulationLayer to visualize
        figsize: Figure size tuple
        
    Returns:
        Tuple of (figure, axes) for the plot
    """
    viz = TerrainVisualizer()
    fig, ax = viz.plot_single_layer(population, figsize=figsize)
    
    # Add settlement information
    settlement_info = population.get_settlement_info()
    
    for settlement_id, info in settlement_info.items():
        center_y, center_x = info['center']
        settlement_type = info['type']
        size = info['size']
        
        # Different markers for different settlement types
        if settlement_type == 'hamlet':
            marker, marker_size = 'o', 30
        elif settlement_type == 'village':
            marker, marker_size = 's', 40
        elif settlement_type == 'town':
            marker, marker_size = '^', 50
        else:  # city
            marker, marker_size = '*', 60
        
        # Plot settlement center
        ax.scatter(center_x, center_y, marker=marker, s=marker_size, 
                  c='red', edgecolors='black', linewidth=2, zorder=10)
        
        # Add text label
        ax.annotate(f'{settlement_type.title()}\n(ID: {settlement_id}, Size: {size})',
                   (center_x, center_y), xytext=(5, 5), textcoords='offset points',
                   fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_title(f"Population Map - {len(settlement_info)} Settlements")
    
    # Add legend for settlement types
    legend_elements = [
        plt.scatter([], [], marker='o', s=30, c='red', edgecolors='black', label='Hamlet'),
        plt.scatter([], [], marker='s', s=40, c='red', edgecolors='black', label='Village'),
        plt.scatter([], [], marker='^', s=50, c='red', edgecolors='black', label='Town'),
        plt.scatter([], [], marker='*', s=60, c='red', edgecolors='black', label='City')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    return fig, ax

def plot_road_network_with_info(roads: RoadLayer, population: Optional[PopulationLayer] = None, 
                               figsize: Tuple[int, int] = (12, 8)):
    """
    Create a specialized plot for RoadLayer with network information.
    
    Args:
        roads: RoadLayer to visualize
        population: Optional PopulationLayer to show settlements
        figsize: Figure size tuple
        
    Returns:
        Tuple of (figure, axes) for the plot
    """
    viz = TerrainVisualizer()
    
    # Create custom colormap for roads
    road_colors = ['white', 'burlywood', 'saddlebrown', 'maroon']
    from matplotlib.colors import ListedColormap
    road_cmap = ListedColormap(road_colors)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot road network
    im = ax.imshow(roads.layer, cmap=road_cmap, origin='lower', vmin=0, vmax=3)
    ax.set_title("Road Network")
    ax.axis('off')
    
    # Add settlement markers if population layer is provided
    if population is not None:
        settlement_info = population.get_settlement_info()
        
        for settlement_id, info in settlement_info.items():
            center_y, center_x = info['center']
            settlement_type = info['type']
            
            # Different markers for different settlement types
            if settlement_type == 'hamlet':
                marker, marker_size = 'o', 40
            elif settlement_type == 'village':
                marker, marker_size = 's', 50
            elif settlement_type == 'town':
                marker, marker_size = '^', 60
            else:  # city
                marker, marker_size = '*', 70
            
            # Plot settlement center
            ax.scatter(center_x, center_y, marker=marker, s=marker_size, 
                      c='red', edgecolors='white', linewidth=2, zorder=10)
            
            # Add settlement ID label
            ax.annotate(f'{settlement_id}', (center_x, center_y), 
                       xytext=(0, 0), textcoords='offset points',
                       ha='center', va='center', fontsize=8, fontweight='bold',
                       color='white')
    
    # Add colorbar for road types
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['No Road', 'Path', 'Road', 'Highway'])
    cbar.set_label('Road Type')
    
    # Add network statistics
    network_info = roads.get_road_network_info()
    stats_text = f"Total Road Cells: {network_info['total_road_cells']}\n"
    stats_text += f"Paths: {network_info['road_type_distribution']['paths']}\n"
    stats_text += f"Roads: {network_info['road_type_distribution']['roads']}\n"
    stats_text += f"Highways: {network_info['road_type_distribution']['highways']}\n"
    stats_text += f"Connections: {network_info['network_connections']}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Add legend for settlement types if population is provided
    if population is not None:
        legend_elements = [
            plt.scatter([], [], marker='o', s=40, c='red', edgecolors='white', label='Hamlet'),
            plt.scatter([], [], marker='s', s=50, c='red', edgecolors='white', label='Village'),
            plt.scatter([], [], marker='^', s=60, c='red', edgecolors='white', label='Town'),
            plt.scatter([], [], marker='*', s=70, c='red', edgecolors='white', label='City')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    return fig, ax

def plot_cost_map(roads: RoadLayer, figsize: Tuple[int, int] = (10, 8)):
    """
    Visualize the terrain cost map used for pathfinding.
    
    Args:
        roads: RoadLayer with calculated cost map
        figsize: Figure size tuple
        
    Returns:
        Tuple of (figure, axes) for the plot
    """
    return roads.visualize_cost_map(figsize=figsize)


