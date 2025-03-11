from herbie import Herbie
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from herbie.toolbox import EasyMap, pc
import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as PathEffects
import rasterio
import scipy.ndimage as ndimage
import numpy.ma as ma
from datetime import datetime
import os


class PrecipitationMapGenerator:
    """
    A class to generate precipitation maps using RTMA data for a specified date.

    This class handles downloading, processing, and visualizing precipitation data
    with geographic context including elevation contours, cities, and geographic labels.
    """

    def __init__(self, date, model="rtma", product="pcp", dem_file=None, output_dir="./"):
        """
        Initialize the PrecipitationMapGenerator with a specified date and optional parameters.

        Args:
            date (str): Date and time for analysis in format "YYYY-MM-DD HH:MM"
            model (str, optional): Weather model to use. Defaults to "rtma".
            product (str, optional): Product to analyze. Defaults to "pcp" (precipitation).
            dem_file (str, optional): Path to DEM file. If None, elevation contours won't be added.
            output_dir (str, optional): Directory to save output files. Defaults to current directory.
        """
        self.date = self._validate_date(date)
        self.model = model
        self.product = product
        self.dem_file = dem_file
        self.output_dir = output_dir

        # Initialize attributes that will be set later
        self.herbie = None
        self.ds = None
        self.fig = None
        self.ax = None
        self.dem_data = None
        self.x_coords = None
        self.y_coords = None
        self.dem_masked = None
        self.precip = None

        # Default map extent (Utah region)
        self.map_extent = [-112.5, -109, 39.5, 41.5]

        # Default cities to display
        self.cities = {
            'Salt Lake City': (-111.891, 40.761),
            'Park City': (-111.498, 40.646),
            'Vernal': (-109.528, 40.455),
            'Roosevelt': (-109.989, 40.299),
            'Duchesne': (-110.400, 40.163)
        }

        # Default geographic labels
        self.geographic_labels = [
            ("Wasatch Range", -111.75, 40.5),
            ("Uinta Mountains", -110.5, 40.8),
            ("Uinta Basin", -110.0, 40.0)
        ]

    def _validate_date(self, date):
        """
        Validate the date format.

        Args:
            date (str): Date string in format "YYYY-MM-DD HH:MM"

        Returns:
            str: Validated date string

        Raises:
            ValueError: If date format is invalid
        """
        try:
            datetime.strptime(date, "%Y-%m-%d %H:%M")
            return date
        except ValueError:
            raise ValueError("Invalid date format. Use 'YYYY-MM-DD HH:MM' format.")

    def download_data(self):
        """
        Download RTMA precipitation data for the specified date.

        Returns:
            self: For method chaining
        """
        self.herbie = Herbie(self.date, model=self.model, product=self.product)
        self.herbie.download()
        return self

    def load_data(self):
        """
        Load the downloaded data with xarray.

        Returns:
            self: For method chaining
        """
        if self.herbie is None:
            self.download_data()

        self.ds = self.herbie.xarray(engine='cfgrib', backend_kwargs={'decode_timedelta': False})

        # Mask zero or very low precipitation values for clarity
        self.ds['tp'] = self.ds.tp.where(self.ds.tp > 0.1)
        return self

    def load_dem(self):
        """
        Load and process DEM file if provided.

        Returns:
            self: For method chaining
        """
        if self.dem_file is None or not os.path.exists(self.dem_file):
            print("Warning: DEM file not provided or does not exist. Elevation contours will not be added.")
            return self

        try:
            with rasterio.open(self.dem_file) as src:
                # Read the first band of the DEM
                self.dem_data = src.read(1)

                # Get the transformation information
                transform = src.transform

                # Create a regular grid of coordinates in the DEM's projection
                height, width = self.dem_data.shape
                rows, cols = np.mgrid[0:height, 0:width]

                # Convert grid indices to coordinates using the transform
                xs, ys = rasterio.transform.xy(transform, rows.flatten(), cols.flatten())

                # Reshape back to 2D arrays matching the DEM shape
                xs = np.array(xs).reshape(rows.shape)
                ys = np.array(ys).reshape(cols.shape)

                # Get arrays of unique x and y coordinates
                self.x_coords = xs[0, :]  # First row contains all x coordinates
                self.y_coords = ys[:, 0]  # First column contains all y coordinates

                # Create a longitude mask to remove data that spills into Colorado
                lon_mask = xs < -109.05  # Utah-Colorado border

                # Smooth the DEM to reduce noise
                dem_smoothed = ndimage.gaussian_filter(self.dem_data, sigma=2)

                # Apply the mask to remove Colorado data
                self.dem_masked = ma.masked_array(dem_smoothed, mask=~lon_mask)

                # Convert elevation from meters to feet
                self.dem_masked = self.dem_masked * 3.28084
        except Exception as e:
            print(f"Error loading DEM file: {e}")
            self.dem_masked = None

        return self

    def create_map(self, figsize=(10, 8), dpi=300):
        """
        Create the map and set up the figure.

        Args:
            figsize (tuple, optional): Figure size. Defaults to (10, 8).
            dpi (int, optional): DPI for the figure. Defaults to 300.

        Returns:
            self: For method chaining
        """
        if self.ds is None:
            self.load_data()

        # Set up figure with higher DPI for publication quality
        plt.rcParams.update({'font.size': 12})
        self.fig = plt.figure(figsize=figsize, dpi=dpi)

        # Create the map with the same projection as the RTMA data
        self.ax = plt.axes(projection=self.ds.herbie.crs)

        # Set the map extent to focus on the study area
        self.ax.set_extent(self.map_extent, crs=ccrs.PlateCarree())

        return self

    def add_elevation_contours(self, contour_levels=None):
        """
        Add elevation contours to the map.

        Args:
            contour_levels (list, optional): Contour levels in feet.
                                            Defaults to range from 6000 to 11000, step 2000.

        Returns:
            self: For method chaining
        """
        if self.dem_masked is None:
            self.load_dem()

        if self.dem_masked is None:
            return self  # Skip if DEM couldn't be loaded

        if self.ax is None:
            self.create_map()

        if contour_levels is None:
            contour_levels = np.arange(6000, 11000, 2000)  # 2000 ft intervals

        contours = self.ax.contour(
            self.x_coords,
            self.y_coords,
            self.dem_masked,
            levels=contour_levels,
            colors='gray',  # Darker gray for contour lines
            linewidths=0.4,
            alpha=0.6,  # Increased alpha for better visibility
            transform=ccrs.PlateCarree(),
            zorder=5
        )

        return self

    def add_geographic_features(self):
        """
        Add geographic features to the map (state boundaries, rivers).

        Returns:
            self: For method chaining
        """
        if self.ax is None:
            self.create_map()

        # Add state boundaries (subtle)
        self.ax.add_feature(cfeature.STATES.with_scale('10m'), linewidth=0.8,
                            edgecolor='black', alpha=0.5, zorder=10)

        # Keep rivers but with reduced visibility
        self.ax.add_feature(cfeature.RIVERS.with_scale('10m'), linewidth=0.5,
                            edgecolor='lightblue', alpha=0.4, zorder=15)

        return self

    def create_precipitation_colormap(self):
        """
        Create a custom colormap for precipitation.

        Returns:
            LinearSegmentedColormap: Custom colormap for precipitation
        """
        colors = [(1, 1, 1, 0),  # transparent for no precipitation
                  (0.85, 0.9, 0.95, 1),  # very light blue
                  (0.65, 0.85, 0.95, 1),  # light blue
                  (0.4, 0.7, 0.9, 1),  # medium blue
                  (0.2, 0.5, 0.8, 1),  # blue
                  (0.1, 0.3, 0.6, 1),  # dark blue
                  (0.05, 0.15, 0.4, 1)]  # very dark blue
        return LinearSegmentedColormap.from_list('precipitation', colors, N=256)

    def add_precipitation_overlay(self, vmin=0, vmax=5):
        """
        Add precipitation visualization to the map.

        Args:
            vmin (float, optional): Minimum value for precipitation colormap. Defaults to 0.
            vmax (float, optional): Maximum value for precipitation colormap. Defaults to 5.

        Returns:
            self: For method chaining
        """
        if self.ds is None:
            self.load_data()

        if self.ax is None:
            self.create_map()

        precip_cmap = self.create_precipitation_colormap()

        # Plot precipitation with improved colormap and smooth shading
        self.precip = self.ax.pcolormesh(
            self.ds.longitude,
            self.ds.latitude,
            self.ds.tp,
            cmap=precip_cmap,
            transform=pc,
            shading='gouraud',
            vmin=vmin,
            vmax=vmax,
            zorder=20
        )

        # Add precipitation contours with darker lines
        precip_contour_levels = [0.5, 1, 2, 3, 4]  # Original contour levels
        precip_contours = self.ax.contour(
            self.ds.longitude,
            self.ds.latitude,
            self.ds.tp,
            levels=precip_contour_levels,
            colors=['#002244'],  # Darker blue for better visibility
            linewidths=0.7,  # Slightly thicker
            alpha=0.9,  # More opaque
            transform=pc,
            zorder=25
        )

        return self

    def add_cities(self, cities=None):
        """
        Add city markers and labels to the map.

        Args:
            cities (dict, optional): Dictionary of city names and coordinates.
                                    If None, uses default cities.

        Returns:
            self: For method chaining
        """
        if self.ax is None:
            self.create_map()

        cities = cities or self.cities

        # Add each city as a point with a label
        for city, (lon, lat) in cities.items():
            # Add city marker
            self.ax.plot(lon, lat, 'ko', markersize=4, transform=pc, zorder=30)

            # Add city label with white outline for visibility
            txt = self.ax.text(lon, lat + 0.05, city, fontsize=9, ha='center', transform=pc, zorder=30)
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='white')])

        return self

    def add_geographic_labels(self, labels=None):
        """
        Add geographic feature labels to the map.

        Args:
            labels (list, optional): List of tuples with (name, lon, lat).
                                    If None, uses default labels.

        Returns:
            self: For method chaining
        """
        if self.ax is None:
            self.create_map()

        labels = labels or self.geographic_labels

        # Add each geographic label with styling
        for name, lon, lat in labels:
            txt = self.ax.text(lon, lat, name, fontsize=9,
                               fontstyle='italic', ha='center',
                               transform=pc, zorder=35,
                               path_effects=[PathEffects.withStroke(linewidth=2, foreground='white')])

        return self

    def add_grid_and_formatting(self):
        """
        Add grid labels and format the map.

        Returns:
            self: For method chaining
        """
        if self.ax is None:
            self.create_map()

        # Simplify grid - only labels, no gridlines
        gl = self.ax.gridlines(crs=pc, draw_labels=True, linewidth=0.0, alpha=0.0)
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 9, 'color': 'black'}
        gl.ylabel_style = {'size': 9, 'color': 'black'}

        return self

    def add_colorbar(self, label="Precipitation (mm/hr)"):
        """
        Add colorbar to the map.

        Args:
            label (str, optional): Label for the colorbar. Defaults to "Precipitation (mm/hr)".

        Returns:
            self: For method chaining
        """
        if not hasattr(self, 'precip') or self.precip is None:
            self.add_precipitation_overlay()

        if self.fig is None:
            self.create_map()

        # Add colorbar
        cbar_ax = self.fig.add_axes([0.15, 0.08, 0.7, 0.03])  # [left, bottom, width, height]
        cb = plt.colorbar(self.precip, cax=cbar_ax, orientation='horizontal')
        cb.set_label(label, fontsize=12)
        cb.ax.tick_params(labelsize=10)

        return self

    def add_titles(self, main_title=None, subtitle="Precipitation Gradient near Wasatch and Uinta Basin"):
        """
        Add titles to the map.

        Args:
            main_title (str, optional): Main title. If None, uses date.
            subtitle (str, optional): Subtitle. Defaults to specified string.

        Returns:
            self: For method chaining
        """
        if self.fig is None:
            self.create_map()

        if main_title is None:
            main_title = f"RTMA Precipitation: {self.date}"

        plt.suptitle(main_title, fontsize=16, y=0.95)
        self.ax.set_title(subtitle, fontsize=14, pad=10)

        return self

    def save_map(self, filename_prefix="rtma_precipitation_map"):
        """
        Save the map to PNG and PDF files.

        Args:
            filename_prefix (str, optional): Prefix for the filenames.
                                           Defaults to "rtma_precipitation_map".

        Returns:
            self: For method chaining
        """
        if self.fig is None:
            print("Warning: No map to save. Call generate() first.")
            return self

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        png_path = os.path.join(self.output_dir, f"{filename_prefix}.png")
        pdf_path = os.path.join(self.output_dir, f"{filename_prefix}.pdf")

        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')  # Vector format

        print(f"Saved map to {png_path} and {pdf_path}")

        return self

    def show(self):
        """
        Display the map.

        Returns:
            self: For method chaining
        """
        if self.fig is None:
            print("Warning: No map to show. Call generate() first.")
            return self

        plt.show()
        return self

    def generate(self):
        """
        Generate the complete map using all components.

        This method orchestrates the entire map generation process.

        Returns:
            self: For method chaining
        """
        self.download_data()
        self.load_data()
        self.load_dem()
        self.create_map()
        self.add_elevation_contours()
        self.add_geographic_features()
        self.add_precipitation_overlay()
        self.add_cities()
        self.add_geographic_labels()
        self.add_grid_and_formatting()
        self.add_colorbar()
        self.add_titles()

        return self


# Example usage
if __name__ == "__main__":
    # Path to your DEM file
    dem_file = 'merged_dem.tif'

    # Create a precipitation map generator for February 22, 2023 at 08:00
    generator = PrecipitationMapGenerator(
        date="2023-02-22 08:00",
        dem_file=dem_file,
        output_dir="./output"
    )

    # Generate and display the map
    generator.generate().save_map().show()

    # Example: Generate maps for multiple dates
    dates = [
        "2023-02-22 08:00",
        "2023-02-23 08:00",
        "2023-02-24 08:00"
    ]

    for date in dates:
        print(f"Generating map for {date}")
        PrecipitationMapGenerator(
            date=date,
            dem_file=dem_file,
            output_dir="./output"
        ).generate().save_map(
            filename_prefix=f"rtma_precipitation_map_{date.replace(' ', '_').replace(':', '')}"
        )